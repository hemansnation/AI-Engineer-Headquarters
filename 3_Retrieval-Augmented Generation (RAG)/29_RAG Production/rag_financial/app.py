from flask import Flask, request, jsonify
from rag_pipeline import load_and_chunk_pdf, create_vector_store, query_rag
from prometheus_flask_exporter import PrometheusMetrics
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)

pdf_path = "data/sample_10k.pdf"
chunks = load_and_chunk_pdf(pdf_path)
vector_store = create_vector_store(chunks)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json(force=True)  # Forces JSON parsing
    except Exception as e:
        return jsonify({"error": "Invalid JSON body"}), 400

    question = data.get('question') if data else None

    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        answer, sources = query_rag(question, vector_store)
        response = {
            "answer": answer,
            "sources": [doc.page_content for doc in sources]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route('/')
def index():
    return "Welcome to the RAG API! Use the /query endpoint to ask questions."

if __name__ == "__main__":
    app.run(port=5002, debug=True)