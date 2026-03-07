import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path):
        """Loads a PDF and splits it into documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

    def process_directory(self, directory_path):
        """Processes all PDFs in a directory."""
        all_docs = []
        for file in os.listdir(directory_path):
            if file.endswith(".pdf"):
                all_docs.extend(self.load_pdf(os.path.join(directory_path, file)))
        return all_docs
