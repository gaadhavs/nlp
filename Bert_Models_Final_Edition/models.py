from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline, BertTokenizer, BertForMaskedLM, T5ForConditionalGeneration, T5Tokenizer
import torch, re
from sklearn.metrics.pairwise import cosine_similarity

class models:
    def __init__(self):
        self.tab1 = True
        self.tab2 = True
        self.tab3 = True
        self.tab4 = True
        self.tab5 = True
        self.tab6 = True

        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.model4 = None
        self.model5 = None
        self.model6 = None

        self.tokenizer2 = None
        self.tokenizer4 = None
        self.tokenizer5 = None
        self.tokenizer6 = None

    
    def emotionAnalyzer(self, text):

        if self.tab1:
            self.model1 = pipeline("text-classification", model = 'bhadresh-savani/bert-base-uncased-emotion', return_all_scores = True)
            self.tab1 = False

        prediction = self.model1(text)
        return prediction[0]
    

    def findSimilarity(self, s1, s2):

        if self.tab2:
            self.tokenizer2 = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
            self.model2 = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
            self.tab2 = False

        sentences = [s1, s2]
        # initialize dictionary to store tokenized sentences
        tokens = {'input_ids': [], 'attention_mask': []}

        for sentence in sentences:
            # encode each sentence and append to dictionary
            new_tokens = self.tokenizer2.encode_plus(sentence, max_length=128,
                                            truncation=True, padding='max_length',
                                            return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])
            
        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
        outputs = self.model2(**tokens)
        # outputs.keys()
        embeddings = outputs.last_hidden_state
        # embeddings.shape
        attention_mask = tokens['attention_mask']
        # attention_mask.shape
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        # mask.shape
        masked_embeddings = embeddings * mask
        # masked_embeddings.shape
        summed = torch.sum(masked_embeddings, 1)
        # summed.shape
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        # summed_mask.shape
        mean_pooled = summed / summed_mask
        mean_pooled = mean_pooled.detach().numpy()

        # calculate
        prediction = cosine_similarity(
            [mean_pooled[0]],
            mean_pooled[1:]
        )
        return prediction[0][0] * 100


    def checkPlagarism(self, s1, s2, threshold = 60):
        similarity = self.findSimilarity(s1, s2)
        if similarity > threshold:
            return True, similarity
        return False, similarity
        

    def question_answering(self, question, paragraph):
        if self.tab3:
            self.model3 = pipeline("question-answering", model = "distilbert-base-uncased-distilled-squad", tokenizer = "distilbert-base-uncased")
            self.tab3 = False
        answer = self.model3(question = question, context = paragraph)
        return answer["answer"]
    
    def predict(self, text): # Result: 1-5
        if self.tab6:
            self.tokenizer6 = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.model6 = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.tab6 = False

        tokens = self.tokenizer6.encode(f"{text}", return_tensors="pt")
        result = self.model6(tokens)
        prediction = result.logits

        return int(torch.argmax(prediction)) + 1

    def compare(self, rating, text):
        prediction = self.predict(text)
        if prediction >= rating - 1 and prediction <= rating + 1:
            return True
        return False
    
    def textSummerier(self, text):
        if self.tab5:
            self.model5 = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer5 = T5Tokenizer.from_pretrained("t5-base")
        
         # Encode the paragraph
        encoded_input = self.tokenizer5(text, return_tensors = "pt")

        # Generate the summary
        output = self.model5.generate(**encoded_input)

        # Decode the summary
        summary = self.tokenizer5.decode(output[0], skip_special_tokens = True)

        return summary
    
    def maskedWordPredictor(self, text):
        if self.tab4:
            self.model4 = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.tokenizer4 = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tab4 = False
    
        tokenized_text = self.tokenizer4.tokenize(text)

        # Convert tokenized text to input tensor
        indexed_tokens = self.tokenizer4.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Find the masked token
        masked_index = tokenized_text.index('[MASK]')

        # Predict the masked token using the pre-trained model
        self.model4.eval()
        with torch.no_grad():
            outputs = self.model4(tokens_tensor)
            predictions = outputs[0][0, masked_index].topk(10)
        
        # Print top 5 predicted tokens and their probabilities
        predicted_tokens = [self.tokenizer4.convert_ids_to_tokens([i.item()])[0] for i in predictions.indices]
        probabilities = predictions.values.tolist()

        return predicted_tokens, probabilities



model = models()

def getModel():
    return model