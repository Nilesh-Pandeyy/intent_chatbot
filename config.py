from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import torch

@dataclass
class ChatbotConfig:
    # Semantic Matching Thresholds
    fuzzy_threshold: float = 0.85
    fuzzy_fallback_threshold: float = 0.75
     
    # BERT Model Configuration
    bert_model_name: str = 'bert-base-uncased'
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    bert_confidence_threshold: float = 0.80
        # Model Settings
    semantic_model_name: str = 'all-mpnet-base-v2'
    context_model_name: str = 'microsoft/deberta-v3-base'
    zero_shot_model_name: str = 'facebook/bart-large-mnli'
    
    # Enhanced Matching Settings
    use_zero_shot: bool = True
    use_contextual: bool = True
    context_history_size: int = 5
    semantic_weight: float = 0.6
    contextual_weight: float = 0.2
    zero_shot_weight: float = 0.2
    
    # Priority Match Settings
    use_priority_matching: bool = True
    priority_match_threshold: float = 0.95
    # Intent Verification Settings
    semantic_weight: float = 0.6
    bert_weight: float = 0.4
    high_confidence_threshold: float = 0.95
    minimum_keyword_match_score: float = 0.3
    
    # Response Generation
    max_response_length: int = 256
    response_temperature: float = 0.7
    top_p: float = 0.95
    
    # Enhancement Settings
    enable_context_enhancement: bool = True
    use_dynamic_responses: bool = True
    max_suggestions: int = 3
    
    # Performance Settings
    cache_embeddings: bool = True
    use_gpu: bool = torch.cuda.is_available()
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging Configuration
    debug_mode: bool = False
    log_confidence_scores: bool = True
    log_intent_verification: bool = True
    
    def __post_init__(self):
        """Validate configuration settings."""
        if self.semantic_weight + self.bert_weight != 1.0:
            raise ValueError("Semantic weight and BERT weight must sum to 1.0")
            
        if not 0 <= self.fuzzy_threshold <= 1:
            raise ValueError("Fuzzy threshold must be between 0 and 1")
            
        if not 0 <= self.bert_confidence_threshold <= 1:
            raise ValueError("BERT confidence threshold must be between 0 and 1")