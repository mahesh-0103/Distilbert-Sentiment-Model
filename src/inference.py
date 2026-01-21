import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from .model import EnhancedDistilBERT
from typing import List, Dict, Union
import time

class SentimentPredictor:
    """Production-ready sentiment predictor"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: 'cpu' or 'cuda'
        """
        print(f"🔧 Loading model from {model_path}...")
        self.device = device
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = EnhancedDistilBERT(num_labels=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Store metadata
        self.accuracy = checkpoint.get('accuracy', 0.0)
        self.epoch = checkpoint.get('epoch', 0)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Accuracy: {self.accuracy*100:.2f}%")
        print(f"   Epoch: {self.epoch}")
    
    def predict(self, text: str, return_probs: bool = True) -> Dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
        
        inference_time = time.time() - start_time
        
        result = {
            'text': text,
            'sentiment': 'positive' if pred.item() == 1 else 'negative',
            'label': int(pred.item()),
            'confidence': float(probs[0][pred.item()]),
            'inference_time_ms': inference_time * 1000
        }
        
        if return_probs:
            result['probabilities'] = {
                'negative': float(probs[0][0]),
                'positive': float(probs[0][1])
            }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.predict(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'accuracy': f"{self.accuracy*100:.2f}%",
            'epoch': self.epoch,
            'total_parameters': f"{total_params:,}",
            'trainable_parameters': f"{trainable_params:,}",
            'efficiency': f"{100*trainable_params/total_params:.1f}%",
            'device': self.device
        }