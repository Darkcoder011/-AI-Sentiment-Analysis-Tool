import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
from sentiment_model import SentimentAnalyzer

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1).cpu()
                
                val_predictions.extend(predictions.numpy())
                val_labels.extend(labels.numpy())

        accuracy = accuracy_score(val_labels, val_predictions)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Average Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(classification_report(val_labels, val_predictions))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Load and preprocess your dataset here
    # This is an example using a CSV file with 'text' and 'label' columns
    df = pd.read_csv('sentiment_data.csv')
    
    # Convert labels to numeric values
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['label'].map(label_map)
    
    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, df['label'].values, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentimentAnalyzer()
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_model(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main() 