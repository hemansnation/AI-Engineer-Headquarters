# Multimodal AI System with Image Captioning

## Project Description
This project implements a Multimodal AI System using the Salesforce BLIP model to generate captions for images and process text inputs. It features a FastAPI backend for handling API requests and a simple HTML/JavaScript front-end for user interaction. The system runs on a CPU, using Python in an Anaconda environment, and is developed in VS Code. It’s modular, beginner-friendly, and extensible for AI engineers.

## Setup Instructions
### Prerequisites
- **Anaconda**: Download and install from [anaconda.com](https://www.anaconda.com/products/distribution).
- **VS Code**: Install from [code.visualstudio.com](https://code.visualstudio.com/).
- **Python 3.8+**: Ensure compatibility with BLIP and dependencies.
- **Git**: For cloning the repository.

### Environment Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/multimodal-ai-system.git
   cd multimodal-ai-system
   ```
2. **Create Anaconda Environment**:
   ```bash
   conda create -n multimodal-ai python=3.8
   conda activate multimodal-ai
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   ```
   torch==2.0.1
   transformers==4.31.0
   fastapi==0.104.1
   uvicorn==0.23.2
   pillow==10.0.0
   jinja2==3.1.2
   python-multipart==0.0.6
   ```

### Folder Structure
```
multimodal-ai-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI backend
│   ├── model.py             # BLIP model logic
│   └── templates/
│       └── index.html       # Front-end interface
├── static/
│   └── styles.css          # CSS for front-end
├── requirements.txt         # Dependencies
├── README.md               # This file
└── test_images/            # Sample images for testing
```

## How to Run
1. **Activate Environment**:
   ```bash
   conda activate multimodal-ai
   ```
2. **Run FastAPI Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. **Access the Application**:
   - Open a browser and navigate to `http://localhost:8000`.
   - Upload an image or enter text to get AI-generated captions or responses.

## How to Test
1. **Test Image Captioning**:
   - Use the web interface to upload an image from `test_images/`.
   - Example: Upload `cat.jpg` and expect a caption like “A fluffy cat sitting on a couch.”
2. **Test Text Processing**:
   - Enter a text prompt like “Describe a sunny day” in the text input field.
   - Verify the AI’s response in the output section.
3. **API Testing**:
   - Use `curl` or Postman to test endpoints:
     ```bash
     curl -X POST -F "file=@test_images/cat.jpg" http://localhost:8000/caption
     ```

## Screenshots
- **Main Interface**: [Placeholder for interface screenshot]
- **Image Captioning Output**: [Placeholder for caption output screenshot]
- **Text Response Output**: [Placeholder for text response screenshot]

## Notes
- The BLIP model runs on CPU, so ensure sufficient memory (at least 8GB RAM).
- Extend the system by adding new endpoints in `main.py` or enhancing the front-end in `index.html`.
- For deployment, refer to the optional `Dockerfile` below.

## Optional: Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t multimodal-ai .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 multimodal-ai
   ```