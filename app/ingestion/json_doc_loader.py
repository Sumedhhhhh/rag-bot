import json
from langchain.schema import Document

def load_json_doc(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    documents = []

    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix} {k}".strip(), v)
        elif isinstance(obj, list):
            for item in obj:
                flatten(prefix, item)
        else:
            documents.append(Document(
                page_content=f"{prefix}: {str(obj)}"
            ))

    flatten("", data)

    return documents