# Sentiment Analysis Tool for Marketing Applications

This project implements a high-accuracy sentiment analysis tool using PyTorch and BERT, designed specifically for marketing applications. The tool can classify text sentiment into positive, neutral, or negative categories with high accuracy, enabling targeted marketing campaigns and improved customer acquisition.

## Features

- BERT-based sentiment analysis with 3-class classification (positive, neutral, negative)
- High accuracy (>85%) through fine-tuned pre-trained models
- Batch processing capabilities for efficient handling of large datasets
- RESTful API for easy integration with marketing platforms
- Confidence scores and probability distributions for each prediction
- Scalable architecture suitable for production environments

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Prepare your dataset in CSV format with 'text' and 'label' columns
2. Run the training script:
```bash
python train.py
```

The model will be saved as 'best_model.pth' when it achieves the best validation accuracy.

### Running the API

Start the API server:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

1. Single Text Analysis:
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this product! It's amazing!"}'
```

2. Batch Analysis:
```bash
curl -X POST "http://localhost:8000/analyze-batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great product!", "It's okay", "Terrible service"]}'
```

3. Health Check:
```bash
curl "http://localhost:8000/health"
```

## Integration with Marketing Platforms

The API can be integrated with various marketing platforms:

1. **Email Marketing Platforms**:
   - Use sentiment analysis to segment customers based on their feedback
   - Send targeted campaigns based on sentiment scores
   - Automate follow-up actions based on sentiment

2. **Social Media Marketing**:
   - Analyze social media posts and comments
   - Identify brand advocates and detractors
   - Monitor campaign effectiveness

3. **Customer Support**:
   - Prioritize negative sentiment cases
   - Route customers to appropriate support channels
   - Track customer satisfaction trends

## Performance Optimization

For production deployment:

1. Use GPU acceleration when available
2. Implement request batching for better throughput
3. Use model quantization for reduced memory footprint
4. Implement caching for frequently analyzed texts
5. Use load balancing for horizontal scaling

## Example Use Cases

1. **Customer Feedback Analysis**:
```python
feedback = "The new features are exactly what I needed!"
result = model.predict(feedback)
# Result: {"sentiment": "positive", "confidence": 0.95, ...}
```

2. **Social Media Monitoring**:
```python
posts = ["Love the new design!", "It's okay", "Hate the new interface"]
results = model.batch_predict(posts)
# Process results for targeted marketing
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 