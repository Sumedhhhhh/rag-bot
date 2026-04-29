import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI

from app.api.routes import upload, qa

app = FastAPI()

app.include_router(upload.router)
app.include_router(qa.router)


#from app.qa.pipeline import run_pipeline

# if __name__ == "__main__" :
#     pdf_path = "data/sample.pdf"
#     questions_path = "data/questions.json"

#     results = run_pipeline(pdf_path, questions_path)

#     print("\nFinal Answers:\n")
#     for q, a in results.items():
#         print(f"\nQ: {q}\nA: {a}\n")


