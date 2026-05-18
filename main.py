from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_sm")


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    doc = nlp(request.text)

    tokens = []

    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dependency": token.dep_,
            "head": token.head.text
        })

    return {
        "tokens": tokens
    }