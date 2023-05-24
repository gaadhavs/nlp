from transformers import pipeline

classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)

def emotionAnalyzer(text):
    prediction = classifier(text)
    return prediction[0]

