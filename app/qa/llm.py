import requests
from app.utils.logger import logger

def generate_answer_ollama(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
        You are a precise document question-answering system.

        Your task is to answer the question using ONLY the information provided in the context below.

        Instructions:
        - Use ONLY the given context
        - Do NOT use any prior knowledge
        - Do NOT make assumptions or guesses
        - If the answer is not clearly present, respond exactly with: Not found
        - Provide a clear, complete, and natural sentence as the answer
        - Do NOT mention the word "context" in your answer
        - Keep the answer concise and factual

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    
    logger.info(f"Sending question to Ollama (mistral): '{question}'")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]
    logger.info("Received response from Ollama")
    return "[MISTRAL]" + answer

