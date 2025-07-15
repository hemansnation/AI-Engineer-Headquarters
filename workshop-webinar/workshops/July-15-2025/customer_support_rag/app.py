from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from retrieval import Retriever
from generation import Generator
import yaml

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

retriever = Retriever(config)
generator = Generator(config)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_rag(request: QueryRequest):
    contexts, metadata = retriever.hybrid_search(request.query)
    answer = generator.generate_answer(request.query, contexts)
    return {"query": request.query, "answer": answer, "contexts": contexts}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)