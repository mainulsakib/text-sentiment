import torch
from transformers import BertTokenizer, BertForSequenceClassification

class TextClassifier:
    def __init__(self, model_path='bert_imdb_classifier'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def classify(self, text, return_probability=False):
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
        # Get label and probability
        label = "Positive" if prediction.item() == 1 else "Negative"
        prob = probabilities[0][prediction].item()
        
        if return_probability:
            return label, prob
        return label

# Example usage
def main():
    # Initialize classifier
    classifier = TextClassifier()
    
    # Example reviews
    reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging throughout.",
        "I was very disappointed with this film. The story was boring and the characters were poorly developed.",
        "While it had some good moments, overall the movie was just okay.",
        "Type your review here to test..."  # Interactive input placeholder
    ]
    
    print("\nClassifying example reviews:")
    print("-" * 80)
    
    # Classify each review
    for review in reviews:
        label, probability = classifier.classify(review, return_probability=True)
        print(f"\nReview: {review}")
        print(f"Sentiment: {label}")
        print(f"Confidence: {probability:.2%}")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode - Enter your own text to classify (type 'quit' to exit)")
    print("=" * 80)
    
    while True:
        text = input("\nEnter text to classify: ")
        if text.lower() == 'quit':
            break
            
        label, probability = classifier.classify(text, return_probability=True)
        print(f"\nSentiment: {label}")
        print(f"Confidence: {probability:.2%}")

if __name__ == "__main__":
    main()