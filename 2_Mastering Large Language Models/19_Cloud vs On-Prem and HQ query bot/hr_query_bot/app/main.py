from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from .database import create_db_and_tables, SessionDep
from .auth import create_access_token, get_current_user, verify_password
from .llm import HRBot
from .models import User
from pydantic import BaseModel
from datetime import timedelta

app = FastAPI()
hr_bot = HRBot()

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    question: str

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), session: SessionDep = Depends()):
    user = session.exec(select(User).where(User.email == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password",
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"access_token":access_token, "token_type":"bearer"}

@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user), session: SessionDep = Depends()):
    response = hr_bot.query(request.question, session)
    return {"response": response}


