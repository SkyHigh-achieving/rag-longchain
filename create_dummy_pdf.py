from fpdf import FPDF
import os

# Create dummy academic paper
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Academic Paper: Advanced RAG Systems', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Abstract
pdf.chapter_title('Abstract')
pdf.chapter_body(
    "This paper proposes a novel Hybrid Retrieval-Augmented Generation (RAG) architecture "
    "specifically designed for academic research. By integrating BM25 keyword search with "
    "dense vector retrieval (BGE-Small), and employing a Cross-Encoder Reranker (BGE-Reranker), "
    "our system achieves state-of-the-art performance in retrieving technical terminology. "
    "We also discuss the importance of local deployment for data privacy."
)

# Introduction
pdf.chapter_title('1. Introduction')
pdf.chapter_body(
    "Retrieval-Augmented Generation (RAG) has revolutionized NLP. However, standard vector-only "
    "approaches often fail to capture exact keyword matches, which are crucial in scientific domains. "
    "For instance, distinguishing between 'LoRA' and 'QLoRA' requires precise lexical matching. "
    "Our proposed method addresses this by using an EnsembleRetriever."
)

# Methodology
pdf.chapter_title('2. Methodology')
pdf.chapter_body(
    "2.1 Hybrid Search: We combine sparse retrieval (BM25) and dense retrieval (Faiss). "
    "The weights are set to 0.3 for BM25 and 0.7 for Vector Search.\n\n"
    "2.2 Reranking: A Cross-Encoder model re-scores the top-10 retrieved documents to filter out "
    "irrelevant noise, ensuring that the LLM receives only the most pertinent context."
)

# Conclusion
pdf.chapter_title('3. Conclusion')
pdf.chapter_body(
    "Experimental results show that our Hybrid RAG system outperforms baseline vector search "
    "by 15% in recall on the SciQ dataset. Future work includes integrating Graph RAG."
)

# Save
output_path = os.path.join("data", "demo_paper.pdf")
if not os.path.exists("data"):
    os.makedirs("data")
pdf.output(output_path)
print(f"Created demo PDF at {output_path}")
