from sqlmodel import SQLModel, create_engine, Session
from typing import Annotated
from fastapi import Depends

sqllite_file_name = "hr_databse.db"
sqlite_url = f"sqlite:///{sqllite_file_name}"

connect_args = {"check_same_thread": False}

engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]