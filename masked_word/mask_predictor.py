import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT model and tokenizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict(text):
    tokenized_text = tokenizer.tokenize(text)

    # Convert tokenized text to input tensor
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Find the masked token
    masked_index = tokenized_text.index('[MASK]')
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0][0, masked_index].topk(10)
        
    predicted_tokens = [tokenizer.convert_ids_to_tokens([i.item()])[0] for i in predictions.indices]
    probabilities = predictions.values.tolist()
    list1 = []
    list2 = []  
    for i in range(len(predicted_tokens)):

        list1.append(predicted_tokens[i])
        list2.append(probabilities[i])
        
    return list1
    
    