import logging
import torch
from torch.utils.data import Dataset
from difflib import get_close_matches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger

COMMON_WORDS = [
    'property', 'video', 'buffering', 'loading', 'account', 
    'password', 'reeltor', 'login', 'image', 'search', 'filter',
    'app', 'crash', 'error', 'reset', 'upload', 'download',
    'profile', 'settings', 'notification', 'message', 'location',
    'detecting', 'correct', 'wrong', 'accuracy', 'gps','verification'
]
WORD_PRESERVATION = {
    'location': 'location',  # Ensure location never gets corrected to notification
    'gps': 'gps',
    'detecting': 'detecting',
    'accuracy': 'accuracy',
    'verification':'verification'
}
'''
def correct_text(text: str) -> str:
    """
    Text correction using fuzzy matching with difflib.
    """
    words = text.lower().split()
    corrected_words = []
    
    for word in words:
        # Only try to correct words longer than 3 characters
        if len(word) > 3:
            matches = get_close_matches(word, COMMON_WORDS, n=1, cutoff=0.7)
            corrected_words.append(matches[0] if matches else word)
        else:
            corrected_words.append(word)
            
    return ' '.join(corrected_words)
'''
def correct_text(text: str) -> str:
    """
    Text correction using fuzzy matching with difflib and word preservation.
    """
    words = text.lower().split()
    corrected_words = []
    
    for word in words:
        # First check if word should be preserved exactly
        if word in WORD_PRESERVATION:
            corrected_words.append(WORD_PRESERVATION[word])
        # Then try correction for words longer than 3 characters
        elif len(word) > 3:
            # More strict cutoff for sensitive terms
            cutoff = 0.8 if word in ['location', 'notification'] else 0.7
            matches = get_close_matches(word, COMMON_WORDS, n=1, cutoff=cutoff)
            corrected_words.append(matches[0] if matches else word)
        else:
            corrected_words.append(word)
            
    return ' '.join(corrected_words)
class IntentDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }