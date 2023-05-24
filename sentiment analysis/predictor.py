from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


def predict(text): # Result: 1-5
    tokens = tokenizer.encode(f"{text}", return_tensors="pt")
    result = model(tokens)
    prediction = result.logits

    return int(torch.argmax(prediction)) + 1

def compare(rating, text):
    prediction = predict(text)
    if prediction >= rating - 1 and prediction <= rating + 1:
        return True
    return False