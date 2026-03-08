from langchain_openai import ChatOpenAI
try:
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema import StrOutputParser
except ImportError:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

class Generator:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        """
        Supports both Local LLMs (via OpenAI-compatible API) and OpenAI.
        For interviews: Highlight that the model choice is decoupled from the retrieval logic.
        """
        self.local_mode = os.getenv("LOCAL_LLM_MODE", "").strip().lower()
        self.use_local_hf = self.local_mode == "hf"
        self.is_local = self.use_local_hf or "localhost" in (base_url or os.getenv("OPENAI_API_BASE", "")) or "127.0.0.1" in (base_url or os.getenv("OPENAI_API_BASE", ""))

        if self.use_local_hf:
            local_model_id = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
            local_model_source, local_files_only = self._resolve_local_model_source(local_model_id)
            attn_impl = os.getenv("LOCAL_ATTENTION_IMPL", "eager")
            tokenizer = AutoTokenizer.from_pretrained(local_model_source, local_files_only=local_files_only)
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_source,
                    dtype=torch.float16,
                    device_map="auto",
                    attn_implementation=attn_impl,
                    local_files_only=local_files_only
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_source,
                    dtype=torch.float32,
                    attn_implementation=attn_impl,
                    local_files_only=local_files_only
                )
            model.config.output_attentions = True
            model.eval()
            self.local_model = model
            self.local_tokenizer = tokenizer
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key or os.getenv("OPENAI_API_KEY", "not-needed") if self.is_local else os.getenv("OPENAI_API_KEY"),
                openai_api_base=base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                temperature=0.1
            )
        
        # Academic-style Prompt with strict context grounding
        self.prompt_template = PromptTemplate.from_template("""
你是一名资深的学术研究导师。请根据以下提供的检索内容，回答用户关于学术论文的问题。

规则：
1. 必须优先使用提供的【检索内容】进行回答。
2. 如果【检索内容】中没有相关信息，请明确告知用户“检索内容未包含该信息”，不要编造。
3. 回答必须紧扣【用户问题】的关键词，不要泛化到无关背景。
4. 优先给出可执行结论，再给出简短依据。
5. 结构清晰，分点回答，控制在 3-6 条以内。

【检索内容】：
{context}

【用户问题】：
{question}

【学术回答】：
""")

    def generate(self, query, context_docs):
        return "".join(self.stream_generate(query, context_docs)).strip()

    def stream_generate(self, query, context_docs):
        context_text, _ = self._format_context(context_docs)
        if self.use_local_hf:
            prompt = self.prompt_template.format(context=context_text, question=query)
            inputs = self.local_tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}
            streamer = TextIteratorStreamer(self.local_tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": int(os.getenv("LOCAL_MAX_NEW_TOKENS", "192")),
                "do_sample": False,
                "pad_token_id": self.local_tokenizer.eos_token_id
            }
            thread = threading.Thread(target=self.local_model.generate, kwargs=generate_kwargs)
            thread.start()
            for token_text in streamer:
                if token_text:
                    yield token_text
            thread.join()
            return

        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(query)
        if answer:
            yield answer

    def analyze_attention(self, query, context_docs, top_n=3):
        result = self.analyze_attention_struct(query, context_docs, top_n=top_n)
        if not result.get("ok"):
            return result.get("message", "注意力分析失败。")
        ranked = result.get("ranked", [])
        lines = [f"注意力焦点（集中度={result.get('concentration', 0.0):.3f}）："]
        for i, item in enumerate(ranked, start=1):
            lines.append(f"{i}. 证据 {item['index']} | 归一化权重 {item['weight']:.3f}")
            lines.append(item["text"][:260].replace("\n", " "))
        return "\n".join(lines)

    def analyze_attention_struct(self, query, context_docs, top_n=3):
        if not self.use_local_hf:
            return {"ok": False, "message": "当前不是本地 HuggingFace 模式，无法提取注意力权重。"}
        context_text, segments = self._format_context(context_docs)
        if not context_text.strip() or not segments:
            return {"ok": False, "message": "当前没有可分析的检索片段。"}
        prompt = self.prompt_template.format(context=context_text, question=query)
        inputs = self.local_tokenizer(prompt, return_tensors="pt")
        model_inputs = inputs.to(self.local_model.device) if torch.cuda.is_available() else inputs
        with torch.no_grad():
            outputs = self.local_model(**model_inputs, output_attentions=True, use_cache=False, return_dict=True)
        attentions = outputs.attentions if hasattr(outputs, "attentions") else None
        if not attentions:
            with torch.no_grad():
                gen_out = self.local_model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            attentions = getattr(gen_out, "attentions", None)
            if isinstance(attentions, tuple) and len(attentions) > 0 and isinstance(attentions[0], tuple):
                attentions = attentions[0]
        if not attentions:
            return {"ok": False, "message": "模型未返回注意力权重。可在 .env 中设置 LOCAL_ATTENTION_IMPL=eager 后重试。"}
        last_attn = attentions[-1][0].mean(dim=0)
        focus = last_attn[-1]
        token_count = int(focus.shape[-1])
        segment_scores = []
        for seg in segments:
            token_ids = self._char_span_to_token_indices(prompt, seg["start"], seg["end"], token_count)
            if not token_ids:
                continue
            score = float(focus[token_ids].mean().item())
            segment_scores.append({"index": seg["index"], "score": score, "text": seg["text"]})
        if not segment_scores:
            return {"ok": False, "message": "未能匹配到注意力与证据片段的对应关系。"}
        ranked_all = sorted(segment_scores, key=lambda x: x["score"], reverse=True)
        total_all = sum(max(x["score"], 0.0) for x in ranked_all) or 1.0
        for item in ranked_all:
            item["weight"] = max(item["score"], 0.0) / total_all
        ranked = ranked_all[:max(1, int(top_n))]
        entropy = -sum(item["weight"] * torch.log(torch.tensor(max(item["weight"], 1e-9))).item() for item in ranked_all)
        max_entropy = torch.log(torch.tensor(max(len(ranked_all), 1))).item() if ranked_all else 1.0
        concentration = 1.0 - (entropy / max(max_entropy, 1e-9))
        return {
            "ok": True,
            "message": "",
            "ranked": ranked,
            "all_scores": ranked_all,
            "top1_index": ranked_all[0]["index"],
            "concentration": float(concentration)
        }

    @staticmethod
    def _resolve_local_model_source(model_id):
        if os.path.isdir(model_id):
            return model_id, True
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir_from_env = os.path.join(project_root, model_id)
        if os.path.isdir(model_dir_from_env):
            return model_dir_from_env, True
        models_root = os.path.join(project_root, "models")
        if os.path.isdir(models_root):
            for name in [model_id.split("/")[-1], "local-qwen2.5-0.5b-instruct"]:
                candidate = os.path.join(models_root, name)
                if os.path.isdir(candidate):
                    return candidate, True
        return model_id, False

    @staticmethod
    def _format_context(context_docs):
        doc_char_limit = int(os.getenv("LOCAL_CONTEXT_DOC_CHARS", "700"))
        total_char_limit = int(os.getenv("LOCAL_CONTEXT_TOTAL_CHARS", "2200"))
        pieces = []
        segments = []
        total = 0
        running_pos = 0
        for i, doc in enumerate(context_docs):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            chunk = text[:doc_char_limit]
            line = f"Source [{i+1}]: {chunk}"
            if total + len(line) > total_char_limit:
                remaining = total_char_limit - total
                if remaining > 40:
                    clipped = line[:remaining]
                    pieces.append(clipped)
                    segments.append({
                        "index": i + 1,
                        "text": clipped,
                        "start": running_pos,
                        "end": running_pos + len(clipped)
                    })
                break
            pieces.append(line)
            segments.append({
                "index": i + 1,
                "text": line,
                "start": running_pos,
                "end": running_pos + len(line)
            })
            total += len(line)
            running_pos += len(line) + 2
        return "\n\n".join(pieces), segments

    def _char_span_to_token_indices(self, prompt, span_start, span_end, token_count):
        token_ids = []
        st = self.local_tokenizer(prompt[:span_start], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]
        ed = self.local_tokenizer(prompt[:span_end], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]
        for token_idx in range(token_count):
            if st <= token_idx < ed:
                token_ids.append(token_idx)
        return token_ids
