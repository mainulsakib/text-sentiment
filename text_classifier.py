import pandas as pd
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'label': torch.tensor(item['label'])
        }

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')
    
    for batch in progress_bar:
        # Move batch to device
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Save predictions and labels
        predictions = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
    
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    
    return epoch_loss, epoch_accuracy

def evaluate(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['label'].to(device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    test_loss = total_loss / len(test_loader)
    return all_predictions, all_labels, test_loss

def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load and prepare data
    print("Loading dataset...")
    dataset = load_dataset('imdb')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Create datasets and dataloaders
    train_dataset = IMDBDataset(tokenized_datasets['train'])
    test_dataset = IMDBDataset(tokenized_datasets['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    num_epochs = 3
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate
        test_predictions, test_labels, test_loss = evaluate(model, test_loader, device)
        test_accuracy = (np.array(test_predictions) == np.array(test_labels)).mean()
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # Final evaluation and metrics
    print("\nFinal Evaluation:")
    final_predictions, final_labels, _ = evaluate(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, final_predictions, target_names=['Negative', 'Positive']))
    
    # Plot confusion matrix
    plot_confusion_matrix(final_labels, final_predictions)
    
    # Save the model
    print("\nSaving model...")
    model.save_pretrained('bert_imdb_classifier')
    tokenizer.save_pretrained('bert_imdb_classifier')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()