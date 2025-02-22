from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import torch
from typing import Dict, List, Tuple
import logging
from utils import setup_logging, IntentDataset
from tqdm import tqdm

class IntentClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(__name__)
        self.device = config.device
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def prepare_data(self, intents: Dict[str, List[str]]) -> Tuple[DataLoader, DataLoader]:
        # First, fit the label encoder with all intent classes
        self.label_encoder.fit(list(intents.keys()))
        num_labels = len(self.label_encoder.classes_)
        
        # Now initialize the model with the correct number of labels
        self.model = BertForSequenceClassification.from_pretrained(
            self.config.bert_model_name,
            num_labels=num_labels
        ).to(self.device)
        
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for intent, examples in intents.items():
            label = self.label_encoder.transform([intent])[0]
            encodings = self.tokenizer(
                examples,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            all_input_ids.extend(encodings['input_ids'])
            all_attention_masks.extend(encodings['attention_mask'])
            all_labels.extend([label] * len(examples))

        # Convert to tensors
        all_input_ids = torch.stack(all_input_ids)
        all_attention_masks = torch.stack(all_attention_masks)
        all_labels = torch.tensor(all_labels)

        # Create full dataset
        dataset = IntentDataset(all_input_ids, all_attention_masks, all_labels)

        # Split into train and validation
        train_size = int((1 - self.config.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        # Generate random indices for splitting
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create train and validation datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data first.")
            
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            with tqdm(train_loader, desc=f"Training Epoch {epoch+1}") as train_pbar:
                for batch in train_pbar:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1}: "
                f"Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}"
            )
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            # Reload the best model state after training is complete
        self.logger.info("Reloading the best model state from best_model.pt")
        self.model.load_state_dict(torch.load('best_model.pt'))

    def predict(self, text: str) -> Tuple[str, float]:
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data first.")
            
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            predicted_intent = self.label_encoder.inverse_transform(
                [predicted_class.item()]
            )[0]
            
            return predicted_intent, confidence.item()