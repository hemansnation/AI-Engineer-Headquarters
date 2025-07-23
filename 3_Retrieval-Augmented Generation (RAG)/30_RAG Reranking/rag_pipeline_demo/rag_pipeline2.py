from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import CrossEncoder
import numpy as np

def load_documents_with_metadata():
    loader = TextLoader("dataset.txt")
    documents = loader.load()

    paragraphs = documents[0].page_content.split("\n\n")
    docs_with_metadata = []
    for i, paragraph in enumerate(paragraphs):
        source = "Tesla" if "Tesla" in paragraph else "Apple"
        year = "2023" if "2023" in paragraph else "2024"

        docs_with_metadata.append({
            "content": paragraph,
            "metadata": {
                "source": source,
                "year": year,
            }
        })
    return docs_with_metadata



text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = load_documents_with_metadata()
chunks = []
for doc in documents:
    split_docs = text_splitter.create_documents([doc['content']], [doc['metadata']])
    chunks.extend(split_docs)

print(f"Created {len(chunks)} chunks.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="advanced_collection")

llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, documents, top_k=3):
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in sorted_indices], scores[sorted_indices]


basic_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True
)

def reranked_qa_chain(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(query)
    reranked_docs, scores = rerank_documents(query, docs)

    context = "\n".join([doc.page_content for doc in reranked_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = llm(prompt)
    return {
        "result": result,
        "source_documents": reranked_docs,
        "scores": scores
    }

def structured_qa_chain(query, metadata_filter):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": metadata_filter})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain({"query": query})

def run_query(query, metadata_filter=None):
    print(f"\n Query: {query}")

    print("\nBasic QA Chain:")
    basic_result = basic_qa_chain({"query": query})
    print("Retrieved Chunks:")
    for doc in basic_result['source_documents']:
        print(f"- {doc.page_content} (Source: {doc.metadata['source']}")
    
    print("\nReranked RAG:")
    reranked_result = reranked_qa_chain(query)
    print("\nRetrieved Reranked Chunks:")
    for doc, score in zip(reranked_result['source_documents'], reranked_result['scores']):
        print(f"- {doc.page_content} (Source: {doc.metadata['source']}, Score: {score:.4f})")
    print(f"Answer: {reranked_result['result']}")

    if metadata_filter:
        print("\nStructured Retrieval RAG(Tesla Only):")
        structured_result = structured_qa_chain(query, metadata_filter)
        print("Retrieved Chunks:")
        for doc in structured_result['source_documents']:
            print(f"- {doc.page_content} (Source: {doc.metadata['source']})")
        print(f"Answer: {structured_result['result']}")
    

query = "What are the challenges for the company's autonomous vehicle technology?"

run_query(query, metadata_filter={"source": "Tesla"})