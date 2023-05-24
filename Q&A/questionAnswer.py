from transformers import pipeline

# Load the question answering pipeline
nlp = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased")


def question_answering(question, paragraph):
    answer = nlp(question=question, context=paragraph)
    return answer["answer"]