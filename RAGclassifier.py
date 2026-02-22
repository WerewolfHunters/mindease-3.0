import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from collections import Counter

class RAGSimilarityClassifier:
    def __init__(self, dataset_path: str, embeddings_path: str, filepath: str,  model_name='all-MiniLM-L6-v2'):
        # Load cleaned dataset and embeddings
        self.df = pd.read_csv(dataset_path)
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.embeddings = np.load(embeddings_path)
        self.filepath = filepath

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Build FAISS index
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

    def chatprocessor(self):
        chat_dict = {'AI': [], 'Human': []}
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, 'r', encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('You'):
                chat_dict['Human'].append(line[5:])

            elif line.startswith('AI'):
                chat_dict['AI'].append(line[4:])

        print("Chat processed at: ", current_time)
        return chat_dict

    def predict_labels(self, top_k: int = 1):
        chat_dict = self.chatprocessor()
        human_sent = chat_dict['Human']
        print(len(human_sent))

        input_embeddings = self.model.encode(human_sent, convert_to_numpy=True, show_progress_bar=False)
        distances, indices = self.index.search(input_embeddings, top_k)
        
        predicted_labels = []
        for i, idx in enumerate(indices):
            label = self.labels[idx[0]]
            predicted_labels.append(label)

        # Count frequency of predicted labels
        label_counts = dict(Counter(predicted_labels))
        return predicted_labels, label_counts

if __name__=="__main__":
    embedding_path = './model/embeddings.npy'
    dataset_path = './model/balanced_cleaned_dataset.csv'
    filepath = './chat_logs/chat_history_anmol21.txt'

    classifier = RAGSimilarityClassifier(dataset_path, embedding_path, filepath)

    predicted_labels, label_counts = classifier.predict_labels()

    print("\nPredicted Labels:", predicted_labels)
    print("Label Counts:", label_counts)