import torch
import joblib
from datetime import datetime
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = './model/distilbert-text-classifier'
LABEL_ENCODER_PATH = './model/label_encoder.joblib'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DisorderPredicter:

    def __init__(self, filepath):
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.filepath = filepath

    
    def chatprocessor(self):
        chat_dict = {'AI': [], 'Human': []}
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('Human'):
                chat_dict['Human'].append(line[7:])

            elif line.startswith('AI'):
                chat_dict['AI'].append(line[4:])

        print("Chat processed at: ", current_time)
        return chat_dict
    

    def chatpredictor(self):
        self.model.to(device)

        chat_dict = self.chatprocessor()
        human_sent = chat_dict['Human']
        print(human_sent)
        print(len(human_sent), '\n')

        encodings = self.tokenizer(
            human_sent, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**encodings)
        
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        labels = self.label_encoder.inverse_transform(preds)

        return labels
    
if __name__=="__main__":

    filepath = './chat_logs/chat_history_1.txt'

    chat = DisorderPredicter(filepath=filepath)
    result = chat.chatpredictor()

    print(result)
    print(len(result))


    

