from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from resume_parser import ResumeParser
from utils import extract_text_from_pdf
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

parser = ResumeParser()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return render_template("error.html", message="No file uploaded.")

    file = request.files['resume']
    if file.filename == '':
        return render_template("error.html", message="No file selected.")

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        if text.startswith("Error:"):
            return render_template("error.html", message=text)

        entities = parser.extract_entities(text)
        score_data = parser.score_resume(entities)

        os.remove(file_path)

        return render_template('results.html', entities=entities, score_data=score_data)
    
    return render_template("error.html", message="Invalid file format. Please upload a PDF file.")


if __name__ == '__main__':
    app.run(debug=True)

