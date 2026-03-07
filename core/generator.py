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

class Generator:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        """
        Supports both Local LLMs (via OpenAI-compatible API) and OpenAI.
        For interviews: Highlight that the model choice is decoupled from the retrieval logic.
        """
        # Determine API source for interview explanation
        self.is_local = "localhost" in (base_url or os.getenv("OPENAI_API_BASE", "")) or "127.0.0.1" in (base_url or os.getenv("OPENAI_API_BASE", ""))
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY", "not-needed") if self.is_local else os.getenv("OPENAI_API_KEY"),
            openai_api_base=base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            temperature=0.1 # Lower temperature for academic precision
        )
        
        # Academic-style Prompt with strict context grounding
        self.prompt_template = PromptTemplate.from_template("""
你是一名资深的学术研究导师。请根据以下提供的检索内容，回答用户关于学术论文的问题。

规则：
1. 必须优先使用提供的【检索内容】进行回答。
2. 如果【检索内容】中没有相关信息，请明确告知用户“检索内容未包含该信息”，不要编造。
3. 保持回答的严谨性，使用专业学术术语。
4. 结构清晰，分点回答。

【检索内容】：
{context}

【用户问题】：
{question}

【学术回答】：
""")

    def generate(self, query, context_docs):
        """
        Generates response using retrieved docs.
        """
        # Format context
        context_text = "\n\n".join([f"Source [{i+1}]: {doc.page_content}" for i, doc in enumerate(context_docs)])
        
        # Build chain
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(query)
