from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.model import generate_caption

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_form(
    request: Request,
    input_text: str = Form(None),
    image: UploadFile = File(None)
):
    image_caption = None
    if image:
        contents = await image.read()
        image_caption = generate_caption(contents)

    summary = f"Received image: {'yes' if image else 'no'}, text: {input_text or 'None'}"
    return JSONResponse(content={
        "image_caption": image_caption,
        "input_text": input_text,
        "summary": summary
    })
