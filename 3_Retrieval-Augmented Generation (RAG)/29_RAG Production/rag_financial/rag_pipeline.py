from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="llama3.2:1b")

def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    return chunks

def create_vector_store(chunks, persist_directory="./chroma_db"):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    return vector_store

def query_rag(question, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    pdf_path = "data/sample_10k.pdf"
    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = create_vector_store(chunks)
    question = "what are the main risks associated with the company's international supply chain?"
    answer, sources = query_rag(question, vector_store)
    print("Answer:", answer)
    print("Sources:", [doc.page_content for doc in sources])