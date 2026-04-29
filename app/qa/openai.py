from openai import OpenAI
from app.config import OPENAI_API_KEY, MODEL_NAME

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def generate_answer_openai(question: str, docs) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])

    # system_prompt = (
    #     "You are a precise document question-answering system. "
    #     "Answer using ONLY the information in the provided context. "
    #     "Do NOT use prior knowledge or make assumptions. "
    #     "If the answer is not in the context, respond exactly with: Not found. "
    #     "Do NOT mention the word 'context' in your answer. "
    #     "Keep the answer concise and factual."
    # )

    system_prompt = (
    "You are a compliance-focused document question-answering system.\n\n"

    "Instructions:\n"
    "1. Answer ONLY using the provided document.\n"
    "2. Extract the answer if explicitly stated.\n"
    "3. If needed, infer the answer carefully from the document.\n"
    "4. If not found, respond exactly with: Not found.\n\n"

    "Answer Rules:\n"
    "- Keep answers concise (max 2-3 sentences).\n"
    "- Always include specific technical terms if present (e.g., 'TLS', 'HTTPS', 'AES-256', 'quarterly').\n"
    "- Prefer specific details (e.g., 'quarterly', 'AES-256', 'TLS', 'MFA').\n"
    "- Do NOT use vague phrases like 'industry-standard'.\n"
    "- Do NOT include unnecessary explanations.\n"
    "- Do NOT copy large paragraphs.\n"
    "- Do NOT use bullet points or numbered lists.\n"
    "- Do NOT mention 'document' or 'context'.\n"
    "- Do NOT hallucinate.\n"
)

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return "[OPENAI] " + response.choices[0].message.content.strip()
