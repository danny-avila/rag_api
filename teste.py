import os
import sys
print(sys.path)

import uvicorn
from langchain.schema import Document
from contextlib import asynccontextmanager
from dotenv import find_dotenv, load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables.config import run_in_executor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import (
    FastAPI,
    File,
    Form,
    Query,
    UploadFile,
    HTTPException,
    status,
    Request,
)


load_dotenv(find_dotenv())



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port="8000", log_config=None)
