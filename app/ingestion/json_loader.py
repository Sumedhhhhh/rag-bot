import json

def load_questions(path : str) :
    with open(path,"r" ) as f :
        data = json.load(f)
    return data["questions"]