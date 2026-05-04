import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI

from app.api.routes import upload, qa, infer

app = FastAPI()

app.include_router(upload.router)
app.include_router(qa.router)
app.include_router(infer.router)



