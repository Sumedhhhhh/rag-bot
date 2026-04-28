def generate_answer(question, docs) :
    context = " ".join([doc.page_content for doc in docs])

    # simple heuristic answer extraction
    context = context.replace("\n", " ").strip()

    # truncate context
    context = context[:400]

    # make it look like a real answer
    return f"Based on the document, {context}"