import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Union
import numpy as np

class SentimentAnalyzer(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            outputs = self(inputs["input_ids"], inputs["attention_mask"])
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            confidence = probabilities[0][prediction].item()
            
            return {
                "sentiment": sentiment_map[prediction],
                "confidence": confidence,
                "probabilities": {
                    sentiment_map[i]: prob.item()
                    for i, prob in enumerate(probabilities[0])
                }
            }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        return [self.predict(text) for text in texts] 