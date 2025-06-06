from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from sentiment_model import SentimentAnalyzer
import uvicorn

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using BERT model",
    version="1.0.0"
)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentAnalyzer()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    try:
        result = model.predict(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(input_data: BatchTextInput):
    try:
        results = model.batch_predict(input_data.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 