from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

MODEL_BY_LANGUAGE = {
    "en": "en_core_web_sm",
    "ru": "ru_core_news_sm",
}

loaded_models = {}


class AnalyzeRequest(BaseModel):
    text: str


def detect_language(text: str) -> str:
    for char in text:
        if "\u0430" <= char.lower() <= "\u044f" or char.lower() == "\u0451":
            return "ru"
    return "en"


def get_pipeline(language: str):
    if language in loaded_models:
        return loaded_models[language]

    model_name = MODEL_BY_LANGUAGE.get(language, MODEL_BY_LANGUAGE["en"])
    try:
        pipeline = spacy.load(model_name)
    except OSError:
        pipeline = spacy.blank(language)

    loaded_models[language] = pipeline
    return pipeline


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": MODEL_BY_LANGUAGE,
        "loaded": list(loaded_models.keys()),
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    language = detect_language(request.text)
    nlp = get_pipeline(language)
    doc = nlp(request.text)

    tokens = []

    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_ or token.text.lower(),
            "pos": token.pos_ or "X",
            "dependency": token.dep_,
            "head": token.head.text,
        })

    return {
        "language": language,
        "tokens": tokens,
    }
