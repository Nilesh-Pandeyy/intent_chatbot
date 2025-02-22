from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import ChatbotConfig
from utils import setup_logging
from typing import Dict, List, Tuple, Optional
import re

class SemanticMatcher:
    def __init__(self, config: ChatbotConfig):
        """Initialize the semantic matcher with multiple models."""
        self.config = config
        self.logger = setup_logging(__name__)
        
        # Initialize main semantic model
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize additional models if GPU is available
        if torch.cuda.is_available():
            self.context_model = AutoModel.from_pretrained('microsoft/deberta-v3-base')
            self.context_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
            self.zero_shot = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli",
                                    device=0)  # Use GPU
        else:
            self.context_model = None
            self.zero_shot = None
        
        self.intent_embeddings = {}
        self.examples_map = {}
        self.context_history = []
        self.account_terms = {
            "action": ["login", "signin", "sign in", "access", "register", "signup", "sign up"],
            "issue": ["cant", "can't", "unable", "not working", "failed", "error", "problem", "stuck"],
            "auth": ["password", "otp", "verification", "verify", "authentication", "code"],
            "account": ["account", "profile", "email", "gmail", "apple id", "credentials"],
            "status": ["locked", "blocked", "pending", "suspended", "disabled"]
        }

        self.property_search_terms = {
            "action": ["search", "find", "look", "browse", "filter", "explore"],
            "property": ["property", "house", "apartment", "flat", "commercial", "office", "shop"],
            "issue": ["not working", "no results", "empty", "cant find", "error", "stuck"],
            "filters": ["budget", "location", "area", "type", "bhk", "price range"],
            "features": ["saved", "favorites", "recommendations", "suggestions", "similar"]
        }

        self.reels_terms = {
            "media": ["reel", "video", "clip", "content", "playback"],
            "action": ["play", "watch", "view", "load", "swipe"],
            "issue": ["not playing", "stuck", "frozen", "black screen", "buffering"],
            "quality": ["resolution", "quality", "blurry", "pixelated", "poor"],
            "audio": ["sound", "audio", "volume", "mute"]
        }

        self.chat_terms = {
            "action": ["message", "chat", "send", "receive", "communicate"],
            "media": ["image", "photo", "file", "attachment", "media"],
            "issue": ["not sending", "failed", "cant send", "not working", "error"],
            "features": ["history", "backup", "notification", "alert"],
            "status": ["delivered", "seen", "pending", "failed"]
        }

        self.community_terms = {
            "action": ["create", "join", "post", "share", "participate"],
            "entity": ["community", "group", "forum", "discussion"],
            "issue": ["cant create", "not working", "error", "failed", "stuck"],
            "features": ["member", "admin", "notification", "post", "content"],
            "access": ["permission", "access", "denied", "restricted"]
        }

        self.app_performance_terms = {
            "performance": ["slow", "lag", "freeze", "crash", "stuck", "hang"],
            "system": ["app", "application", "system", "device", "phone"],
            "issue": ["not working", "error", "problem", "malfunction"],
            "resources": ["memory", "storage", "cache", "data"],
            "status": ["unresponsive", "responding", "loading", "processing"]
        }

        self.location_terms = {
        "primary": ["location", "gps", "area", "map", "place", "address"],
        "action": ["access", "enable", "find", "detect", "search", "track"],
        "issue": ["not working", "error", "incorrect", "wrong", "inaccurate", "poor", "failed"],
        "permission": ["allow", "denied", "permission", "access", "enable"],
        "accuracy": ["precise", "accurate", "exact", "correct", "accuracy", "poor"]

        }

        self.notification_terms = {
            "type": ["notification", "alert", "update", "message", "reminder"],
            "action": ["receive", "get", "send", "push", "show"],
            "issue": ["not getting", "missing", "delayed", "not working", "error"],
            "settings": ["preferences", "settings", "configure", "setup"],
            "features": ["sound", "vibration", "priority", "silent"]
        }
        
        # Enhanced priority matches from your existing code
        self.priority_matches = {

                "issue in uploading property": "Property_Upload",
    "problem uploading property": "Property_Upload",
    "facing issue in upload": "Property_Upload",
    "upload not working": "Property_Upload",
    "unable to upload": "Property_Upload",
    "can't post property": "Property_Upload",
    "difficulty in uploading": "Property_Upload",
    "property upload failed": "Property_Upload",
    
    # Specific Upload Issues
    "property details not saving": "Property_Upload",
    "images not uploading": "Property_Upload",
    "error while uploading photos": "Property_Upload",
    "cant save property information": "Property_Upload",
    "listing stuck": "Property_Upload",
    "property post not working": "Property_Upload",

    "location access not working": "Location_Issues",
        "location not working": "Location_Issues",
        "cant access location": "Location_Issues",
        "location permission denied": "Location_Issues",
        "gps not working": "Location_Issues",
        "unable to access location": "Location_Issues",
        "location services error": "Location_Issues",
        "location access error": "Location_Issues",
        "location access denied": "Location_Issues",
        "wrong area showing": "Location_Issues",
        "location not accurate": "Location_Issues",
        "incorrect location": "Location_Issues",
        "area detection wrong": "Location_Issues",
        "location accuracy poor": "Location_Issues",
        "poor location accuracy": "Location_Issues",
        "gps accuracy issues": "Location_Issues",
        "inaccurate location": "Location_Issues",
        "location precision poor": "Location_Issues",
    
    # Upload Process Questions
    "how to upload property": "Property_Upload",
    "steps to post property": "Property_Upload",
    "guide for property upload": "Property_Upload",
    "property posting process": "Property_Upload",
    "help with property upload": "Property_Upload",
    
    # Upload Status Issues
    "upload stuck in processing": "Property_Upload",
    "property upload pending": "Property_Upload",
    "post under review": "Property_Upload",
    "upload status not updating": "Property_Upload",
    # Account Login Issues
    "can't log into my account": "Account_Login",
    "login not working": "Account_Login",
    "unable to sign in": "Account_Login",
    "login page error": "Account_Login",
    "account access problem": "Account_Login",
    "google sign in failed": "Account_Login",
    "gmail login not working": "Account_Login",
    "apple id login failed": "Account_Login",
    "cant login with apple": "Account_Login",
    "password reset not working": "Account_Login",
    "forgot password link": "Account_Login",
    "otp not received": "Account_Login",
    "verification code missing": "Account_Login",
    "account is locked": "Account_Login",
    "email verification pending": "Account_Login",

    # Property Search Issues
    "cant find properties": "Property_Search",
    "search not working": "Property_Search",
    "no search results": "Property_Search",
    "search error": "Property_Search",
    "cant find commercial": "Property_Search",
    "no commercial spaces": "Property_Search",
    "shop search not working": "Property_Search",
    "no recommendations": "Property_Search",
    "explore page empty": "Property_Search",
    "suggestions not showing": "Property_Search",
    "filter not working": "Property_Search",
    "cant find 3bhk": "Property_Search",
    "budget range not updating": "Property_Search",
    "location filter stuck": "Property_Search",
    "saved search missing": "Property_Search",
    "favorites not showing": "Property_Search",

    # Property Upload Issues
    "cant upload property": "Property_Upload",
    "listing creation failed": "Property_Upload",
    "upload error": "Property_Upload",
    "image upload failed": "Property_Upload",
    "photos not uploading": "Property_Upload",
    "photo size error": "Property_Upload",
    "cant edit listing": "Property_Upload",
    "editing not working": "Property_Upload",
    "cant update property": "Property_Upload",
    "post stuck processing": "Property_Upload",
    "listing not visible": "Property_Upload",
    "post rejected": "Property_Upload",
    "details not saving": "Property_Upload",
    "cant add amenities": "Property_Upload",
    "location wrong in post": "Property_Upload",

    # Reels Features Issues
    "reels not playing": "Reels_Features",
    "video stuck loading": "Reels_Features",
    "black screen in reels": "Reels_Features",
    "videos frozen": "Reels_Features",
    "no sound in video": "Reels_Features",
    "cant swipe reels": "Reels_Features",
    "swipe not working": "Reels_Features",
    "cant see owner details": "Reels_Features",
    "reel details missing": "Reels_Features",
    "video quality poor": "Reels_Features",
    "videos not smooth": "Reels_Features",
    "reels stuck loading": "Reels_Features",
    "infinite loading": "Reels_Features",
    "app freezes in reels": "Reels_Features",
    "reel section crash": "Reels_Features",

    # Chat Features Issues
    "cant send messages": "Chat_Features",
    "message not delivering": "Chat_Features",
    "chat not working": "Chat_Features",
    "sending failed": "Chat_Features",
    "chat history gone": "Chat_Features",
    "old messages gone": "Chat_Features",
    "past conversations": "Chat_Features",
    "chat notifications": "Chat_Features",
    "cant send images": "Chat_Features",
    "media not sending": "Chat_Features",
    "photo sharing failed": "Chat_Features",
    "chat keeps crashing": "Chat_Features",
    "messages not loading": "Chat_Features",
    "chat not responding": "Chat_Features",

    # Community Features Issues
    "cant create community": "Community_Features",
    "community creation failed": "Community_Features",
    "community setup error": "Community_Features",
    "cant post in community": "Community_Features",
    "post not publishing": "Community_Features",
    "community post failed": "Community_Features",
    "community posts not loading": "Community_Features",
    "cant see community": "Community_Features",
    "feed not updating": "Community_Features",
    "cant join community": "Community_Features",
    "membership error": "Community_Features",
    "access denied": "Community_Features",
    "community alerts": "Community_Features",
    "community search": "Community_Features",

    # App Performance Issues
    "app very slow": "App_Performance",
    "slow response": "App_Performance",
    "app lagging": "App_Performance",
    "performance poor": "App_Performance",
    "app keeps crashing": "App_Performance",
    "app closes suddenly": "App_Performance",
    "unexpected shutdowns": "App_Performance",
    "app freezes": "App_Performance",
    "screen frozen": "App_Performance",
    "app not responding": "App_Performance",
    "controls not working": "App_Performance",
    "content not loading": "App_Performance",
    "loading screen stuck": "App_Performance",
    "app stuck": "App_Performance",
    "app malfunctioning": "App_Performance",

    # Location Issues
    "location access not working": "Location_Issues",
    "cant access location": "Location_Issues",
    "permission denied": "Location_Issues",
    "gps not working": "Location_Issues",
    "location services": "Location_Issues",
    "wrong area showing": "Location_Issues",
    "location not accurate": "Location_Issues",
    "incorrect location": "Location_Issues",
    "area detection wrong": "Location_Issues",
    "cant search by location": "Location_Issues",
    "area search not working": "Location_Issues",
    "location filter issues": "Location_Issues",
    "cant find nearby": "Location_Issues",
    "location preferences": "Location_Issues",
    "location settings": "Location_Issues",

    # Notification Issues
    "not getting alerts": "Notifications",
    "notifications missing": "Notifications",
    "alerts not working": "Notifications",
    "price change notifications": "Notifications",
    "notification settings": "Notifications",
    "cant change notification": "Notifications",
    "alert settings": "Notifications",
    "notifications not showing": "Notifications",
    "no alert sounds": "Notifications",
    "delayed notifications": "Notifications",
    "missing alerts": "Notifications",
    "notification system": "Notifications",
    "chat notifications": "Notifications",
    "community alerts": "Notifications",

    # General Help
    "how to use": "General_Help",
    "app usage guide": "General_Help",
    "basic features": "General_Help",
    "getting started": "General_Help",
    "new user guide": "General_Help",
    "app features": "General_Help",
    "available functions": "General_Help",
    "need help": "General_Help",
    "support required": "General_Help",
    "property terms": "General_Help",
    "real estate guidelines": "General_Help",
    "usage instructions": "General_Help",
    "best practices": "General_Help",
    "helpful hints": "General_Help"
}
        

    def is_account_login_query(self, query: str) -> bool:
        """Check if query is related to account/login issues."""
        query_lower = query.lower()
        
        has_action = any(term in query_lower for term in self.account_terms["action"])
        has_issue = any(term in query_lower for term in self.account_terms["issue"])
        has_auth = any(term in query_lower for term in self.account_terms["auth"])
        has_account = any(term in query_lower for term in self.account_terms["account"])
        has_status = any(term in query_lower for term in self.account_terms["status"])
        
        return (has_account or has_auth) and (has_issue or has_action or has_status)

    def is_property_search_query(self, query: str) -> bool:
        """Check if query is related to property search."""
        query_lower = query.lower()
        
        has_action = any(term in query_lower for term in self.property_search_terms["action"])
        has_property = any(term in query_lower for term in self.property_search_terms["property"])
        has_issue = any(term in query_lower for term in self.property_search_terms["issue"])
        has_filters = any(term in query_lower for term in self.property_search_terms["filters"])
        has_features = any(term in query_lower for term in self.property_search_terms["features"])
        
        return (has_property or has_filters) and (has_action or has_issue or has_features)

    def is_reels_features_query(self, query: str) -> bool:
        """Check if query is related to reels/video features."""
        query_lower = query.lower()
        
        has_media = any(term in query_lower for term in self.reels_terms["media"])
        has_action = any(term in query_lower for term in self.reels_terms["action"])
        has_issue = any(term in query_lower for term in self.reels_terms["issue"])
        has_quality = any(term in query_lower for term in self.reels_terms["quality"])
        has_audio = any(term in query_lower for term in self.reels_terms["audio"])
        
        return has_media and (has_action or has_issue or has_quality or has_audio)

    def is_chat_features_query(self, query: str) -> bool:
        """Check if query is related to chat features."""
        query_lower = query.lower()
        
        has_action = any(term in query_lower for term in self.chat_terms["action"])
        has_media = any(term in query_lower for term in self.chat_terms["media"])
        has_issue = any(term in query_lower for term in self.chat_terms["issue"])
        has_features = any(term in query_lower for term in self.chat_terms["features"])
        has_status = any(term in query_lower for term in self.chat_terms["status"])
        
        return (has_action or has_media) and (has_issue or has_features or has_status)

    def is_community_features_query(self, query: str) -> bool:
        """Check if query is related to community features."""
        query_lower = query.lower()
        
        has_action = any(term in query_lower for term in self.community_terms["action"])
        has_entity = any(term in query_lower for term in self.community_terms["entity"])
        has_issue = any(term in query_lower for term in self.community_terms["issue"])
        has_features = any(term in query_lower for term in self.community_terms["features"])
        has_access = any(term in query_lower for term in self.community_terms["access"])
        
        return has_entity and (has_action or has_issue or has_features or has_access)

    def is_app_performance_query(self, query: str) -> bool:
        """Check if query is related to app performance."""
        query_lower = query.lower()
        
        has_performance = any(term in query_lower for term in self.app_performance_terms["performance"])
        has_system = any(term in query_lower for term in self.app_performance_terms["system"])
        has_issue = any(term in query_lower for term in self.app_performance_terms["issue"])
        has_resources = any(term in query_lower for term in self.app_performance_terms["resources"])
        has_status = any(term in query_lower for term in self.app_performance_terms["status"])
        
        return (has_system or has_performance) and (has_issue or has_resources or has_status)

    def is_location_query(self, query: str) -> bool:
        """Enhanced check for location-related queries with explicit pattern matching."""
        query_lower = query.lower()
        
        # Direct location phrases that should always match
        explicit_patterns = [
            "location is not",
            "location not",
            "wrong location",
            "incorrect location",
            "location incorrect",
            "location detection",
            "detecting location",
            "location accuracy",
            "gps not"
        ]
        
        if any(pattern in query_lower for pattern in explicit_patterns):
            return True
            
        # Component-based matching
        has_location = "location" in query_lower or "gps" in query_lower
        has_detection = "detect" in query_lower or "detecting" in query_lower
        has_issue = any(term in query_lower for term in ["not", "wrong", "incorrect", "poor", "bad"])
        
        return has_location and (has_detection or has_issue)

    def is_notification_query(self, query: str) -> bool:
        """Check if query is related to notifications."""
        query_lower = query.lower()
        
        has_type = any(term in query_lower for term in self.notification_terms["type"])
        has_action = any(term in query_lower for term in self.notification_terms["action"])
        has_issue = any(term in query_lower for term in self.notification_terms["issue"])
        has_settings = any(term in query_lower for term in self.notification_terms["settings"])
        has_features = any(term in query_lower for term in self.notification_terms["features"])
        
        return has_type and (has_action or has_issue or has_settings or has_features)

    def _find_match_for_query(self, query: str) -> Tuple[Optional[str], Optional[str], float]:
        """Enhanced match finding with specific feature detection."""
        query_lower = query.lower()
        
        # Check for specific feature patterns
        if self.is_image_upload_query(query_lower):
            return "Property_Upload", self._get_best_example(query, "Property_Upload"), 0.95
        elif self.is_account_login_query(query_lower):
            return "Account_Login", self._get_best_example(query, "Account_Login"), 0.95
        elif self.is_property_search_query(query_lower):
            return "Property_Search", self._get_best_example(query, "Property_Search"), 0.95
        elif self.is_reels_features_query(query_lower):
            return "Reels_Features", self._get_best_example(query, "Reels_Features"), 0.95
        elif self.is_chat_features_query(query_lower):
            return "Chat_Features", self._get_best_example(query, "Chat_Features"), 0.95
        elif self.is_community_features_query(query_lower):
            return "Community_Features", self._get_best_example(query, "Community_Features"), 0.95
        elif self.is_app_performance_query(query_lower):
            return "App_Performance", self._get_best_example(query, "App_Performance"), 0.95
        elif self.is_location_query(query_lower):
            return "Location_Issues", self._get_best_example(query, "Location_Issues"), 0.95
        elif self.is_notification_query(query_lower):
            return "Notifications", self._get_best_example(query, "Notifications"), 0.95
        
        # Check priority matches
        for phrase, intent in self.priority_matches.items():
            if phrase in query_lower:
                return intent, self._get_best_example(query, intent), 0.95
            

    def prepare_embeddings(self, intents: Dict[str, List[str]]):
        """Prepare embeddings and additional context patterns."""
        for intent, examples in intents.items():
            self.examples_map[intent] = examples
            
            # Generate semantic embeddings
            embeddings = self.semantic_model.encode(examples, convert_to_tensor=True)
            self.intent_embeddings[intent] = embeddings
            
            # Generate contextual patterns if available
            if self.context_model is not None:
                self._prepare_contextual_patterns(intent, examples)
            
            self.logger.info(f"Prepared embeddings for intent: {intent}")

    def _prepare_contextual_patterns(self, intent: str, examples: List[str]):
        """Prepare contextual patterns using DeBERTa."""
        patterns = []
        for example in examples:
            inputs = self.context_tokenizer(example, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.context_model(**inputs)
                patterns.append(outputs.last_hidden_state.mean(dim=1))
        
        if intent not in self.intent_embeddings:
            self.intent_embeddings[intent] = {}
        self.intent_embeddings[intent]['contextual'] = torch.cat(patterns)


    def _prepare_query_variations(self, query: str) -> List[str]:
        """Generate variations of the query to improve matching."""
        variations = [query]
        query_lower = query.lower()
        
        # Common substitutions
        substitutions = {
            "cant": "can't",
            "doesnt": "doesn't",
            "isnt": "isn't",
            "wont": "won't",
            "wheres": "where's",
            "howto": "how to",
            "signin": "sign in",
            "login": "log in",
            "cant find": "cannot find",
            "unable to": "can't",
            "not able to": "cannot",
            "no": "not",
            "image": "photo",
            "picture": "photo",
            "error": "problem",
            "issue": "problem"
        }
        
        # Generate variations with common substitutions
        for old, new in substitutions.items():
            if old in query_lower:
                variations.append(query_lower.replace(old, new))
                variations.append(query_lower.replace(new, old))
        
        # Add variations without punctuation
        variations.append(re.sub(r'[^\w\s]', '', query_lower))
        
        # Add variations with/without common prefixes
        prefixes = ["how to", "how do i", "why", "what", "where"]
        query_words = query_lower.split()
        if query_words[0] in prefixes:
            variations.append(' '.join(query_words[1:]))
        
        # Add negation variations
        negations = ["can't", "cannot", "not", "no", "unable to"]
        for neg in negations:
            if neg in query_lower:
                simple_form = query_lower.replace(neg, "").strip()
                variations.append(simple_form)
        
        return list(set(variations)) 
    def _find_match_for_query(self, query: str) -> Tuple[Optional[str], Optional[str], float]:
        """Find the best match for a single query variation."""
        query_lower = query.lower()
        
        # Check priority matches first
        for phrase, intent in self.priority_matches.items():
            if phrase in query_lower:
                best_example = self._get_best_example(query, intent)
                return intent, best_example, 0.95
        
        # Get semantic similarity scores
        query_embedding = self.semantic_model.encode([query])[0]
        semantic_scores = {}
        
        for intent, embeddings in self.intent_embeddings.items():
            if isinstance(embeddings, dict):
                semantic_embeddings = embeddings.get('semantic', torch.tensor([]))
            else:
                semantic_embeddings = embeddings
            
            similarities = cosine_similarity([query_embedding], 
                                        semantic_embeddings.cpu().numpy())[0]
            semantic_scores[intent] = np.max(similarities)
        
        # Get zero-shot classification if available
        if self.zero_shot is not None:
            zero_shot_results = self.zero_shot(
                query,
                list(self.examples_map.keys()),
                multi_label=True
            )
            for intent, score in zip(zero_shot_results['labels'], 
                                zero_shot_results['scores']):
                if intent in semantic_scores:
                    semantic_scores[intent] = (semantic_scores[intent] + score) / 2
        
        # Consider context history
        if self.context_history:
            recent_intent = self.context_history[-1]
            if recent_intent in semantic_scores:
                semantic_scores[recent_intent] *= 1.1  # Slight boost for context
        
        # Find best match
        best_score = -1
        best_intent = None
        best_example = None
        
        for intent, score in semantic_scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent
                best_example = self._get_best_example(query, intent)
        
        return best_intent, best_example, best_score
    def find_best_match(self, query: str) -> Tuple[Optional[str], Optional[str], float]:
        """Enhanced matching using query variations and multiple models."""
        # Generate query variations
        query_variations = self._prepare_query_variations(query)
        best_score = -1
        best_intent = None
        best_example = None
        
        # Check each variation
        for variation in query_variations:
            intent, example, score = self._find_match_for_query(variation)
            if score > best_score:
                best_score = score
                best_intent = intent
                best_example = example
        
        # Update context history
        if best_intent:
            self.context_history.append(best_intent)
            if len(self.context_history) > 5:
                self.context_history.pop(0)
        
        return best_intent, best_example, best_score
    def _get_best_example(self, query: str, intent: str) -> str:
        """Get the best matching example for an intent."""
        if intent not in self.examples_map:
            return ""
            
        query_embedding = self.semantic_model.encode([query])[0]
        examples = self.examples_map[intent]
        example_embeddings = self.semantic_model.encode(examples)
        
        similarities = cosine_similarity([query_embedding], example_embeddings)[0]
        best_idx = np.argmax(similarities)
        
        return examples[best_idx]


    def get_intent_similarity(self, query: str, intent: str) -> float:
        """Get enhanced similarity score between query and intent."""
        if intent not in self.intent_embeddings:
            return 0.0
            
        query_embedding = self.semantic_model.encode([query])[0]
        embeddings = self.intent_embeddings[intent]
        
        if isinstance(embeddings, dict):
            semantic_embeddings = embeddings.get('semantic', torch.tensor([]))
            contextual_embeddings = embeddings.get('contextual', torch.tensor([]))
            
            semantic_sim = cosine_similarity([query_embedding], 
                                           semantic_embeddings.cpu().numpy())[0]
            
            if contextual_embeddings.nelement() > 0:
                contextual_sim = cosine_similarity([query_embedding], 
                                                 contextual_embeddings.cpu().numpy())[0]
                return max(np.max(semantic_sim), np.max(contextual_sim))
                
            return np.max(semantic_sim)
        
        similarities = cosine_similarity([query_embedding], 
                                       embeddings.cpu().numpy())[0]
        return np.max(similarities)