from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

loader = TextLoader("dataset.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="demo_collection")

llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

def run_query(query):
    result = qa_chain({"query": query})
    print(f"\nQuery: {query}")
    print(f"\nRetrieved Chunks: ")
    for doc in result['source_documents']:
        print(f"- {doc.page_content}")
    print(f"\nAnswer:", {result['result']})


run_query("Whats were Tesla's revenues in Q4 2023?")

run_query("What are the challenges for the company's autonomous vehicle technology?")