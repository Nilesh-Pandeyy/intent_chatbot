from typing import Dict, List, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import setup_logging
from config import ChatbotConfig
import random


class ResponseGenerator:
    def __init__(self, config: ChatbotConfig, responses: Dict[str, Dict[str, List[str]]]):
        self.config = config
        self.logger = setup_logging(__name__)
        self.responses = responses
        '''
        self.performance_patterns = {
        "speed_issues": {
            "keywords": ["slow", "lag", "sluggish", "delay", "taking time", "not fast", 
                        "loading slow", "performance poor", "very slow"],
            "context": "speed"
        },
        "crash_issues": {
            "keywords": ["crash", "close", "shut down", "stopping", "closes suddenly", 
                        "keeps closing", "force stop", "unexpected close"],
            "context": "crashes"
        },
        "freeze_issues": {
            "keywords": ["freeze", "frozen", "stuck", "not responding", "hangs", 
                        "screen frozen", "unresponsive", "controls stuck"],
            "context": "freezing"
        },
        "loading_issues": {
            "keywords": ["loading", "stuck loading", "infinite loading", "not loading", 
                        "content not loading", "loading screen"],
            "context": "loading"
        },
        "technical_issues": {
            "keywords": ["features not working", "malfunctioning", "technical problem", 
                        "not functioning", "system error", "app error"],
            "context": "technical"
        }
    }
        '''
        # Direct phrase mapping for exact matches
        self.exact_matches = {

             "app very slow": ("App_Performance", "speed"),
        "slow response time": ("App_Performance", "speed"),
        "app lagging": ("App_Performance", "speed"),
        "performance very poor": ("App_Performance", "speed"),
        "everything loading slow": ("App_Performance", "speed"),
        
        # Crashes
        "app keeps crashing": ("App_Performance", "crashes"),
        "application closes suddenly": ("App_Performance", "crashes"),
        "unexpected shutdowns": ("App_Performance", "crashes"),
        "app not stable": ("App_Performance", "crashes"),
        "frequent crashes": ("App_Performance", "crashes"),
        
        # Freezing
        "app freezes frequently": ("App_Performance", "freezing"),
        "screen frozen": ("App_Performance", "freezing"),
        "app not responding": ("App_Performance", "freezing"),
        "controls not working": ("App_Performance", "freezing"),
        "app hangs": ("App_Performance", "freezing"),
        
        # Loading
        "content not loading": ("App_Performance", "loading"),
        "infinite loading": ("App_Performance", "loading"),
        "loading screen stuck": ("App_Performance", "loading"),
        "app stuck on loading": ("App_Performance", "loading"),
        "data not loading": ("App_Performance", "loading"),
        
        # General Issues
        "app not working properly": ("App_Performance", "technical"),
        "features not responding": ("App_Performance", "technical"),
        "app malfunctioning": ("App_Performance", "technical"),
        "technical problems": ("App_Performance", "technical"),
        "system errors": ("App_Performance", "technical"),
            # Reels/Video Issues
            "reel video got freeze": ("Reels_Features", "loading"),
            "video got stuck": ("Reels_Features", "loading"),
            "reel not playing": ("Reels_Features", "playback"),
            "video freeze while watching": ("Reels_Features", "loading"),
            "reel stuck on loading": ("Reels_Features", "loading"),
            "reel video not working": ("Reels_Features", "playback"),
            # OTP/Verification
            "verification code not coming": ("Account_Login", "otp"),
            "verification code missing": ("Account_Login", "otp"),
            "apple id verification error": ("Account_Login", "apple"),
            
            # Location
            "location permission denied": ("Location_Issues", "access"),
            "location accuracy very poor": ("Location_Issues", "accuracy"),
            "can't enable location services": ("Location_Issues", "access"),
            "location filter stuck": ("Location_Issues", "filter"),
            "unable to detect current location": ("Location_Issues", "access"),
            
            # Property/Business
            "unable to search business properties": ("Property_Search", "commercial"),
            "no suggestions in feed": ("Property_Search", "recommendations"),
            "top locality section blank": ("Property_Search", "recommendations"),
            "featured properties missing": ("Property_Search", "recommendations"),
            "similar properties not showing": ("Property_Search", "recommendations"),
            "property suggestions not loading": ("Property_Search", "recommendations"),
            "recommended listings disappeared": ("Property_Search", "recommendations")
        }

        # Complete intent patterns
        self.intent_patterns = {

            "speed_issues": {
            "keywords": ["slow", "lag", "sluggish", "delay", "taking time", "not fast", 
                        "loading slow", "performance poor", "very slow"],
            "context": "speed"
        },
        "crash_issues": {
            "keywords": ["crash", "close", "shut down", "stopping", "closes suddenly", 
                        "keeps closing", "force stop", "unexpected close"],
            "context": "crashes"
        },
        "freeze_issues": {
            "keywords": ["freeze", "frozen", "stuck", "not responding", "hangs", 
                        "screen frozen", "unresponsive", "controls stuck"],
            "context": "freezing"
        },
        "loading_issues": {
            "keywords": ["loading", "stuck loading", "infinite loading", "not loading", 
                        "content not loading", "loading screen"],
            "context": "loading"
        },
        "technical_issues": {
            "keywords": ["features not working", "malfunctioning", "technical problem", 
                        "not functioning", "system error", "app error"],
            "context": "technical"
        },

            # Account Login patterns
            "account_patterns": {
                "intent": "Account_Login",
                "context": "account",
                "keywords": ["account", "login", "sign in", "access", "profile", "can't login"]
            },
            "gmail_patterns": {
                "intent": "Account_Login",
                "context": "gmail",
                "keywords": ["google", "gmail", "google sign", "google account", "gmail login"]
            },
            "apple_patterns": {
                "intent": "Account_Login",
                "context": "apple",
                "keywords": ["apple id", "apple verification", "apple login", "ios verification", "apple sign"]
            },
            "otp_patterns": {
                "intent": "Account_Login",
                "context": "otp",
                "keywords": ["verification code", "otp", "verification", "code missing", "not coming", "sms code"]
            },
            "password_patterns": {
                "intent": "Account_Login",
                "context": "password",
                "keywords": ["password", "reset", "forgot", "change password", "password not working"]
            },

            # Location patterns
            "location_access": {
                "intent": "Location_Issues",
                "context": "access",
                "keywords": ["location permission", "enable location", "location services", "detect location", "location access", "gps not working"]
            },
            "location_accuracy": {
                "intent": "Location_Issues",
                "context": "accuracy",
                "keywords": ["location accuracy", "accuracy poor", "wrong location", "incorrect location", "wrong area"]
            },
            "location_search": {
                "intent": "Location_Issues",
                "context": "search",
                "keywords": ["location search", "area search", "find location", "search by location", "location based search"]
            },
            "location_filter": {
                "intent": "Location_Issues",
                "context": "filter",
                "keywords": ["location filter", "area filter", "filter by location", "filter stuck"]
            },

            # Property Search patterns
            "property_guide": {
                "intent": "Property_Search",
                "context": "guide",
                "keywords": ["how to search", "search guide", "find properties", "search help", "property guide"]
            },
            "property_filter": {
                "intent": "Property_Search",
                "context": "filter",
                "keywords": ["filter", "price range", "budget filter", "type filter", "searchbar", "filter not working"]
            },
            "property_results": {
                "intent": "Property_Search",
                "context": "results",
                "keywords": ["no results", "empty results", "nothing found", "no properties", "search results"]
            },
            "property_location": {
                "intent": "Property_Search",
                "context": "location",
                "keywords": ["in my area", "nearby", "around me", "this locality", "local search"]
            },
            "property_saved": {
                "intent": "Property_Search",
                "context": "saved",
                "keywords": ["saved", "favorites", "bookmarks", "saved searches", "saved properties"]
            },
            "property_commercial": {
                "intent": "Property_Search",
                "context": "commercial",
                "keywords": ["commercial", "showroom", "office space", "retail space", "shop", "business property"]
            },
            "property_recommendations": {
                "intent": "Property_Search",
                "context": "recommendations",
                "keywords": ["recommendations", "explore", "suggestions", "top locality", "trending", "featured", "similar properties"]
            },

            # Property Upload patterns
            "upload_guide": {
                "intent": "Property_Upload",
                "context": "guide",
                "keywords": ["how to post", "upload guide", "post property", "listing guide", "property posting"]
            },
            "upload_image": {
                "intent": "Property_Upload",
                "context": "image",
                "keywords": ["upload photos", "images", "pictures", "photo upload", "property photos"]
            },
            "upload_listing": {
                "intent": "Property_Upload",
                "context": "listing",
                "keywords": ["create listing", "post property", "add listing", "new listing", "list property"]
            },
            "upload_visibility": {
                "intent": "Property_Upload",
                "context": "visibility",
                "keywords": ["not visible", "can't see listing", "listing hidden", "not showing", "visibility"]
            },
            "upload_details": {
                "intent": "Property_Upload",
                "context": "details",
                "keywords": ["property details", "specifications", "details not saving", "add details"]
            },
            "upload_status": {
                "intent": "Property_Upload",
                "context": "status",
                "keywords": ["post status", "under review", "processing", "approval status"]
            },
                         # Reels Features patterns
            "reels_playback": {
                "intent": "Reels_Features",
                "context": "playback",
                "keywords": ["reel not playing", "video stuck", "video freeze", "reel freeze", "video frozen", "reel video", "video not playing"]
            },
            "reels_loading": {
                "intent": "Reels_Features",
                "context": "loading",
                "keywords": ["reel loading", "video loading", "video buffering", "reel stuck loading", "infinite loading"]
            },
            "reels_navigation": {
                "intent": "Reels_Features",
                "context": "navigation",
                "keywords": ["can't swipe", "swipe not working", "navigation stuck", "can't navigate reels"]
            },
            "reels_performance": {
                "intent": "Reels_Features",
                "context": "performance",
                "keywords": ["reel lag", "video lag", "performance issues", "app freeze in reels", "reel performance"]
            },
            "reels_quality": {
                "intent": "Reels_Features",
                "context": "reel_info",
                "keywords": ["video quality", "poor quality", "blurry video", "low resolution", "quality settings"]
            }

        }
    '''
    def _detect_intent_and_context(self, user_input: str) -> Tuple[str, str]:
        """Enhanced intent and context detection."""
        user_input_lower = user_input.lower()
        
        # Check exact matches first
        for phrase, (intent, context) in self.exact_matches.items():
            if phrase in user_input_lower:
                return intent, context
        
        # Score tracking for pattern matching
        best_match = {
            "score": 0,
            "intent": None,
            "context": None
        }

        # Check each pattern group
        for pattern_group in self.intent_patterns.values():
            score = 0
            
            # Check for keyword matches with weighted scoring
            for keyword in pattern_group["keywords"]:
                if keyword in user_input_lower:
                    # Higher weight for multi-word matches and exact matches
                    weight = 3 if keyword == user_input_lower else 2 if " " in keyword else 1
                    score += weight
            
            # Update best match if better score found
            if score > best_match["score"]:
                best_match["score"] = score
                best_match["intent"] = pattern_group["intent"]
                best_match["context"] = pattern_group["context"]

        # If good match found, return it
        if best_match["score"] > 0:
            return best_match["intent"], best_match["context"]

        # Specific fallback rules
        if "verification" in user_input_lower or "otp" in user_input_lower:
            return "Account_Login", "otp"
        if "location" in user_input_lower:
            if "accuracy" in user_input_lower or "wrong" in user_input_lower:
                return "Location_Issues", "accuracy"
            return "Location_Issues", "access"
        if "recommendations" in user_input_lower or "suggestions" in user_input_lower:
            return "Property_Search", "recommendations"
        if "commercial" in user_input_lower or "business" in user_input_lower:
            return "Property_Search", "commercial"

        return "General_Help", "support"

    '''
    def _detect_intent_and_context(self, user_input: str) -> Tuple[str, str]:
        """Enhanced intent and context detection."""
        user_input_lower = user_input.lower()


        location_patterns = {
            "accuracy": ["not detecting correct", "wrong location", "incorrect location", "inaccurate", "poor accuracy"],
            "access": ["can't access", "not working", "permission denied", "unable to access"],
            "detection": ["not detecting", "can't detect", "detection failed", "not finding location"]
        }
        for context, patterns in location_patterns.items():
            if any(pattern in user_input_lower for pattern in patterns):
                return "Location_Issues", context
            
        
        if any(combo in user_input_lower for combo in [
            "verification code not coming",
            "verification code missing",
            "verification not received",
            "otp not coming",
            "otp not received",
            "not getting otp",
            "no otp",
            "otp missing",
            "code not coming",
            "verification failed",
            "otp failed"
        ]):
            return "Account_Login", "otp"
        
        # Broader OTP/verification check
        has_otp = any(term in user_input_lower for term in ["otp", "one time password"])
        has_verification = any(term in user_input_lower for term in ["verification", "verify", "code"])
        has_issue = any(term in user_input_lower for term in ["not", "missing", "failed", "no", "isn't"])
        
        if (has_otp or has_verification) and has_issue:
            return "Account_Login", "otp"
        # Check for high-priority location patterns first
        if "accuracy" in user_input_lower and any(term in user_input_lower for term in ["poor", "bad", "wrong", "incorrect"]):
            return "Location_Issues", "accuracy"
            
        # Check exact matches
        for phrase, (intent, context) in self.exact_matches.items():
            if phrase in user_input_lower:
                return intent, context
        
        # Score tracking for pattern matching
        best_match = {
            "score": 0,
            "intent": None,
            "context": None
        }

        # Check each pattern group
        for pattern_group in self.intent_patterns.values():
            score = 0
            
            # Add higher base score for location patterns
            base_score = 3 if pattern_group.get("intent") == "Location_Issues" else 1
            
            # Check for keyword matches with weighted scoring
            for keyword in pattern_group["keywords"]:
                if keyword in user_input_lower:
                    # Higher weight for multi-word matches and exact matches
                    weight = base_score * (3 if keyword == user_input_lower else 2 if " " in keyword else 1)
                    score += weight
            
            # Update best match if better score found
            if score > best_match["score"]:
                best_match["score"] = score
                best_match["intent"] = pattern_group["intent"]
                best_match["context"] = pattern_group["context"]

        # If good match found, return it
        if best_match["score"] > 0:
            return best_match["intent"], best_match["context"]

        # Enhanced specific fallback rules
        if "verification" in user_input_lower or "otp" in user_input_lower:
            return "Account_Login", "otp"
            
        if "location" in user_input_lower:
            # Enhanced location checks
            if any(term in user_input_lower for term in ["accuracy", "precise", "wrong", "incorrect", "poor"]):
                return "Location_Issues", "accuracy"
            if any(term in user_input_lower for term in ["access", "permission", "enable", "detect"]):
                return "Location_Issues", "access"
            if any(term in user_input_lower for term in ["search", "find", "nearby"]):
                return "Location_Issues", "search"
            return "Location_Issues", "access"  # Default location context
            
        if "recommendations" in user_input_lower or "suggestions" in user_input_lower:
            return "Property_Search", "recommendations"
            
        if "commercial" in user_input_lower or "business" in user_input_lower:
            return "Property_Search", "commercial"

        return "General_Help", "support"
    
    def generate_response(self, user_input: str, intent: Optional[str] = None) -> str:
        """Generate appropriate response based on intent and user input."""
        self.logger.info(f"Detected intent: {intent}, input: {user_input}")
        
        if not intent:
            return self._get_fallback_response()

        # Get response categories for the intent
        intent_responses = self.responses.get(intent, {})
        if not intent_responses:
            return self._get_fallback_response()

        # Convert input to lower case for matching
        user_input_lower = user_input.lower()

               # Special handling for Account_Login intent
        if intent == "Account_Login":
            # Mobile number update
            if any(word in user_input_lower for word in ["stuck", "loading", "frozen"]):
                loading_responses = intent_responses.get("loading_stuck", [])
                if loading_responses:
                    return random.choice(loading_responses)
            if ("mobile" in user_input_lower or "number" in user_input_lower) and \
               ("change" in user_input_lower or "update" in user_input_lower):
                mobile_responses = intent_responses.get("mobile_update", [])
                if mobile_responses:
                    return random.choice(mobile_responses)
            # Email verification
            if "email" in user_input_lower and \
               ("verify" in user_input_lower or "verification" in user_input_lower or "pending" in user_input_lower):
                email_responses = intent_responses.get("email_verification", [])
                if email_responses:
                    return random.choice(email_responses)
            # Profile update
            if "profile" in user_input_lower and \
               ("update" in user_input_lower or "change" in user_input_lower or "modify" in user_input_lower):
                profile_responses = intent_responses.get("profile_update", [])
                if profile_responses:
                    return random.choice(profile_responses)

        # Special handling for Property_Search intent
        elif intent == "Property_Search":
            # Search not working
            if "search" in user_input_lower and \
               ("not working" in user_input_lower or "error" in user_input_lower):
                error_responses = intent_responses.get("search_error", [])
                if error_responses:
                    return random.choice(error_responses)
            # No properties found
            if ("no properties" in user_input_lower or "empty" in user_input_lower or \
                "not found" in user_input_lower) and "results" in user_input_lower:
                results_responses = intent_responses.get("results", [])
                if results_responses:
                    return random.choice(results_responses)
            # Rent vs Buy option
            if "rent" in user_input_lower and "buy" in user_input_lower:
                option_responses = intent_responses.get("rent_buy", [])
                if option_responses:
                    return random.choice(option_responses)

        # Special handling for Property_Upload intent
        elif intent == "Property_Upload":
            # Image upload issues
            if ("image" in user_input_lower or "photo" in user_input_lower) and \
               ("upload" in user_input_lower or "size" in user_input_lower or "limit" in user_input_lower):
                image_responses = intent_responses.get("image", [])
                if image_responses:
                    return random.choice(image_responses)
            # Post rejection
            if "rejected" in user_input_lower or "not approved" in user_input_lower:
                rejection_responses = intent_responses.get("status", [])
                if rejection_responses:
                    return random.choice(rejection_responses)
            # Location pin error
            if "location" in user_input_lower and \
               ("wrong" in user_input_lower or "error" in user_input_lower or "pin" in user_input_lower):
                location_responses = intent_responses.get("location_pin", [])
                if location_responses:
                    return random.choice(location_responses)
            if any(phrase in user_input_lower for phrase in ["quality", "resolution", "blurry"]):
                quality_responses = intent_responses.get("image_quality", [])
                if quality_responses:
                    return random.choice(quality_responses)
                    
            if "floor plan" in user_input_lower or "layout" in user_input_lower:
                plan_responses = intent_responses.get("floor_plan", [])
                if plan_responses:
                    return random.choice(plan_responses)
        elif intent == "Property_Management":
            if "sold" in user_input_lower or "status" in user_input_lower:
                status_responses = intent_responses.get("status_update", [])
                if status_responses:
                    return random.choice(status_responses)
                    
            if any(word in user_input_lower for word in ["dimension", "measurement", "size"]):
                dimension_responses = intent_responses.get("dimensions", [])
                if dimension_responses:
                    return random.choice(dimension_responses)
        # Special handling for Reels_Features intent
        elif intent == "Reels_Features":
            navigation_keywords = ["navigate", "swipe", "navigation", "move between", "switch"]
            if any(keyword in user_input_lower for keyword in navigation_keywords):
                navigation_responses = intent_responses.get("navigation", [])
                if navigation_responses:
                    return random.choice(navigation_responses)
            # Playback issues
            if ("not playing" in user_input_lower or "black screen" in user_input_lower or \
                "frozen" in user_input_lower):
                playback_responses = intent_responses.get("playback", [])
                if playback_responses:
                    return random.choice(playback_responses)
            # Sound issues
            if "sound" in user_input_lower and \
               ("no" in user_input_lower or "not working" in user_input_lower):
                audio_responses = intent_responses.get("audio", [])
                if audio_responses:
                    return random.choice(audio_responses)
            # Details not visible
            if ("details" in user_input_lower or "info" in user_input_lower) and \
               ("blank" in user_input_lower or "not showing" in user_input_lower):
                info_responses = intent_responses.get("reel_info", [])
                if info_responses:
                    return random.choice(info_responses)

        # Special handling for Chat_Features intent
        elif intent == "Chat_Features":
            if "notification" in user_input_lower or "alert" in user_input_lower:
                notification_responses = intent_responses.get("notifications", [])
                if notification_responses:
                    return random.choice(notification_responses)
                    
            if "history" in user_input_lower or "old" in user_input_lower:
                history_responses = intent_responses.get("history", [])
                if history_responses:
                    return random.choice(history_responses)
            # Message sending issues
            if ("message" in user_input_lower or "chat" in user_input_lower) and \
               ("not sending" in user_input_lower or "failed" in user_input_lower):
                sending_responses = intent_responses.get("sending", [])
                if sending_responses:
                    return random.choice(sending_responses)
            # Chat history issues
            if "history" in user_input_lower or \
               ("old" in user_input_lower and "messages" in user_input_lower):
                history_responses = intent_responses.get("history", [])
                if history_responses:
                    return random.choice(history_responses)
            # Notifications not working
            if "notification" in user_input_lower and \
               ("not getting" in user_input_lower or "not working" in user_input_lower):
                notification_responses = intent_responses.get("notifications", [])
                if notification_responses:
                    return random.choice(notification_responses)

        # Special handling for Community_Features intent
        elif intent == "Community_Features":
            if any(phrase in user_input_lower for phrase in ["member", "approval", "request"]):
                member_responses = intent_responses.get("member_requests", [])
                if member_responses:
                    return random.choice(member_responses)
                    
            if "analytics" in user_input_lower or "stats" in user_input_lower:
                analytics_responses = intent_responses.get("analytics", [])
                if analytics_responses:
                    return random.choice(analytics_responses)
            # Creation issues
            if ("create" in user_input_lower or "new" in user_input_lower) and \
               "community" in user_input_lower:
                creation_responses = intent_responses.get("creation", [])
                if creation_responses:
                    return random.choice(creation_responses)
            # Posting issues
            if ("post" in user_input_lower or "share" in user_input_lower) and \
               ("not working" in user_input_lower or "failed" in user_input_lower):
                posting_responses = intent_responses.get("posting", [])
                if posting_responses:
                    return random.choice(posting_responses)
            # Access denied
            if ("join" in user_input_lower or "access" in user_input_lower) and \
               "denied" in user_input_lower:
                access_responses = intent_responses.get("membership", [])
                if access_responses:
                    return random.choice(access_responses)
        elif intent == "Location_Issues":

            access_keywords = ["access not working", "cant access", "unable to access", 
                            "permission denied", "location permission", "enable location"]
            if any(keyword in user_input_lower for keyword in access_keywords):
                if "access" in intent_responses:
                    return random.choice(intent_responses["access"])
                    
            # Location update issues
            update_keywords = ["cant update", "update not working", "unable to update", 
                            "location update", "update location"]
            if any(keyword in user_input_lower for keyword in update_keywords):
                if "filter" in intent_responses:
                    return random.choice(intent_responses["filter"])
                    
            # Location accuracy issues
            accuracy_keywords = ["accuracy poor", "not accurate", "wrong location", 
                                "incorrect location", "location wrong", "poor accuracy"]
            if any(keyword in user_input_lower for keyword in accuracy_keywords):
                if "accuracy" in intent_responses:
                    return random.choice(intent_responses["accuracy"])
                    
            # GPS specific issues
            gps_keywords = ["gps not working", "gps error", "gps problem", 
                        "location services", "location tracking"]
            if any(keyword in user_input_lower for keyword in gps_keywords):
                if "access" in intent_responses:
                    return random.choice(intent_responses["access"])
                    
            # Location detection issues
            detection_keywords = ["detect location", "find location", "location detection", 
                                "unable to detect", "cant detect", "detection failed"]
            if any(keyword in user_input_lower for keyword in detection_keywords):
                if "access" in intent_responses:
                    return random.choice(intent_responses["access"])
                    
            # Location search issues
            search_keywords = ["search location", "find nearby", "location search", 
                            "area search", "search by location"]
            if any(keyword in user_input_lower for keyword in search_keywords):
                if "search" in intent_responses:
                    return random.choice(intent_responses["search"])
                    
            # Default to access response for unmatched location issues
            if "access" in intent_responses:
                return random.choice(intent_responses["access"])
                

        # Special handling for App_Performance intent
        elif intent == "App_Performance":
            # Speed/Performance issues
            if any(word in user_input_lower for word in ["slow", "lag", "sluggish", "performance poor", "very slow"]):
                speed_responses = intent_responses.get("speed", [])
                if speed_responses:
                    return random.choice(speed_responses)
                    
            # Crash issues
            if any(word in user_input_lower for word in ["crash", "closes", "shutdown", "closing suddenly", "keeps crashing"]):
                crash_responses = intent_responses.get("crashes", [])
                if crash_responses:
                    return random.choice(crash_responses)
                    
            # Freezing issues
            if any(word in user_input_lower for word in ["freeze", "frozen", "stuck", "not responding", "hangs"]):
                freeze_responses = intent_responses.get("freezing", [])
                if freeze_responses:
                    return random.choice(freeze_responses)
                    
            # Loading issues
            if any(word in user_input_lower for word in ["loading", "stuck loading", "infinite loading", "not loading"]):
                loading_responses = intent_responses.get("loading", [])
                if loading_responses:
                    return random.choice(loading_responses)
                    
            # Technical/General issues
            if any(word in user_input_lower for word in ["technical", "error", "malfunction", "problem", "not working"]):
                technical_responses = intent_responses.get("technical", [])
                if technical_responses:
                    return random.choice(technical_responses)
                    
            # Response/Control issues
            if any(word in user_input_lower for word in ["controls", "buttons", "features", "not responding", "unresponsive"]):
                response_responses = intent_responses.get("response", [])
                if response_responses:
                    return random.choice(response_responses)
            
            # Check for image-specific issues
            if any(keyword in user_input_lower for keyword in self.performance_patterns["image_loading"]["keywords"]):
                responses = self.responses["App_Performance"].get("image_performance", [])
                if responses:
                    return random.choice(responses)
            if "slow" in user_input_lower or "lag" in user_input_lower:
                speed_responses = intent_responses.get("speed", [])
                if speed_responses:
                    return random.choice(speed_responses)
            # Crashing issues
            if "crash" in user_input_lower or "closes" in user_input_lower:
                crash_responses = intent_responses.get("crashes", [])
                if crash_responses:
                    return random.choice(crash_responses)
            # Freezing issues
            if "frozen" in user_input_lower or "not responding" in user_input_lower:
                freeze_responses = intent_responses.get("freezing", [])
                if freeze_responses:
                    return random.choice(freeze_responses)

        # Match specific categories based on keywords
        for category, responses in intent_responses.items():
            if self._matches_category(user_input_lower, category):
                return random.choice(responses)

        if intent == "Account_Login":
            account_responses = intent_responses.get("account", [])
            if account_responses:
                return random.choice(account_responses)
                
        elif intent == "Property_Search":
            guide_responses = intent_responses.get("guide", [])
            if guide_responses:
                return random.choice(guide_responses)
                
        elif intent == "Property_Upload":
            upload_responses = intent_responses.get("guide", [])
            if upload_responses:
                return random.choice(upload_responses)
                
        elif intent == "Reels_Features":
            playback_responses = intent_responses.get("playback", [])
            if playback_responses:
                return random.choice(playback_responses)
                
        elif intent == "Chat_Features":
            sending_responses = intent_responses.get("sending", [])
            if sending_responses:
                return random.choice(sending_responses)
                
        elif intent == "Community_Features":
            notification_keywords = [
                "notification", "alert", "not receiving", "not getting",
                "no updates", "missing alerts"
            ]
            if any(keyword in user_input_lower for keyword in notification_keywords):
                notification_responses = intent_responses.get("notifications", [])
                if notification_responses:
                    return random.choice(notification_responses)
            creation_responses = intent_responses.get("creation", [])
            if creation_responses:
                return random.choice(creation_responses)
                
        elif intent == "App_Performance":
            performance_responses = intent_responses.get("speed", [])
            if performance_responses:
                return random.choice(performance_responses)
                
        elif intent == "Location_Issues":
            access_responses = intent_responses.get("access", [])
            if access_responses:
                return random.choice(access_responses)
                
        elif intent == "Notifications":
            settings_responses = intent_responses.get("settings", [])
            if settings_responses:
                return random.choice(settings_responses)
                
        elif intent == "General_Help":
            support_responses = intent_responses.get("support", [])
            if support_responses:
                return random.choice(support_responses)
                
        elif intent == "Browse_Properties":
            navigation_responses = intent_responses.get("navigation", [])
            if navigation_responses:
                return random.choice(navigation_responses)
                
        elif intent == "Profile_Management":
            settings_responses = intent_responses.get("settings", [])
            if settings_responses:
                return random.choice(settings_responses)

        # Match specific categories based on keywords
        for category, responses in intent_responses.items():
            if self._matches_category(user_input_lower, category):
                return random.choice(responses)

        # If no specific category matched, use general response
        general_responses = next(iter(intent_responses.values()), [])
        return random.choice(general_responses) if general_responses else self._get_fallback_response()

    def _matches_category(self, user_input: str, category: str) -> bool:
        """Check if user input matches a response category."""
        category_keywords = {
            # Account Login Categories
            "mobile_update": {"mobile", "number", "phone", "contact", "change number", "update number", "registered mobile"},
            "email_verification": {"email", "verify", "verification", "pending", "verify email"},
            "profile_update": {"profile", "information", "details", "can't update", "modify profile"},
            "account": {"profile", "account", "settings", "manage"},
            "password": {"password", "reset", "forgot"},
            "otp": {"otp", "verification", "code", "verify"},
            "apple": {"apple", "ios"},
            "gmail": {"gmail", "google"},


            # Property Search Categories
            "search_error": {"search not working", "can't find", "search error", "no results"},
            "recommendations": {"recommendation", "recommended", "suggestions", "not showing"},
            "rent_buy": {"rent vs buy", "rental", "buying", "set rent"},
            "location": {"nearby", "location", "area search", "filter stuck"},
            "results": {"no properties", "list empty", "timeout", "not updating"},
            "saved": {"favorites", "saved properties", "saved search"},

            # Property Upload Categories
            "guide": {"posting guide", "how to post", "upload guide"},
            "image": {"image upload", "photo size", "format", "maximum images", "6 images"},
            "listing": {"listing creation", "upload error", "failed"},
            "details": {"information update", "add amenities", "details"},
            "status": {"post rejected", "under review", "not visible"},
            "location_pin": {"location wrong", "address", "pin error"},

            # Reels Categories
            "playback": {"not playing", "stuck loading", "black screen", "frozen"},
            "loading": {"loading", "buffering", "stuck on loading"},
            "navigation": {"swipe", "navigation", "stuck", "blank"},
            "reel_info": {"owner details", "property info", "details missing", "not showing"},

            # Chat Categories
            "sending": {"message", "can't send", "not delivering", "failed"},
            "history": {"chat history", "previous chats", "old messages", "backup"},
            "stability": {"crashing", "crash", "blank", "loading"},
            "notifications": {"chat notifications", "alerts", "sound", "not getting"},

            # Community Categories
            "creation": {"create community", "start community", "setup"},
            "posting": {"post", "publish", "share", "content"},
            "content": {"posts not loading", "can't see", "feed", "blank"},
            "membership": {"join", "access denied", "membership"},

            # App Performance Categories
            "speed": {"slow", "lagging", "performance", "loading slow"},
            "crashes": {"crashing", "closes", "shutdowns", "not stable"},
            "freezing": {"frozen", "freezes", "not responding", "hangs"},
            "loading": {"stuck", "loading screen", "not loading"}
        }

        keywords = category_keywords.get(category, {category})
        return any(keyword in user_input for keyword in keywords)
    
    def _get_fallback_response(self) -> str:
        """Return a fallback response when no intent is matched."""
        fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "Could you please provide more details about your question?",
            "I apologize, but I need more information to help you properly."
        ]
        return random.choice(fallback_responses)

    def _format_response(self, response: str, intent: str) -> str:
        """Format response with consistent styling."""
        formatted = []
        formatted.append("👋 Hello! I understand your concern.")
        
        if intent:
            formatted.append(f"\n📋 Issue Category: {intent.replace('_', ' ')}")
        
        formatted.append("\n🔍 Solution:")
        formatted.append(response)
        
        formatted.append("\n💡 Additional Help:")
        formatted.append("• Visit our help center: help.reeltor.com")
        formatted.append("• Contact support: support@reeltor.com")
        formatted.append("• Call us: 1-800-REELTOR")
        
        return "\n".join(formatted)
    
