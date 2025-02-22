from config import ChatbotConfig
from response_generator import ResponseGenerator
from intent_classifier import IntentClassifier
from semantic_matcher import SemanticMatcher
from utils import setup_logging
from typing import Dict, List, Tuple, Optional, Any
import logging
import re

from utils import correct_text

class ReeltorChatbot:
    def __init__(
        self,
        config: ChatbotConfig,
        intents: Dict[str, List[str]],
        responses: Dict[str, Dict[str, List[str]]]
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.semantic_matcher = SemanticMatcher(config)
        self.intent_classifier = IntentClassifier(config)
        self.response_generator = ResponseGenerator(config, responses)
        
        # Prepare data
        self.semantic_matcher.prepare_embeddings(intents)
        train_loader, val_loader = self.intent_classifier.prepare_data(intents)
        
        # Train BERT model
        self.logger.info("Training BERT model...")
        self.intent_classifier.train(train_loader, val_loader)
        self.logger.info("BERT model training completed")


    def _detect_multiple_intents(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Detect multiple intents from a complex user query.
        Returns a list of dictionaries containing intent and relevant parts of the query.
        """
        # Split on common conjunctions
        parts = re.split(r'\s+(?:and|but|also|plus|moreover|furthermore)\s+', user_input.lower())
        
        detected_intents = []
        for part in parts:
            # Get semantic match for each part
            intent, example, score = self.semantic_matcher.find_best_match(part)
            if score >= self.config.fuzzy_threshold:
                detected_intents.append({
                    'query_part': part,
                    'intent': intent,
                    'confidence': score
                })
        
        return detected_intents
    def _is_complex_query(self, user_input: str) -> bool:
        """
        Determine if a query contains multiple distinct issues/intents.
        """
        # Common conjunction patterns
        conjunctions = [
            "and", "&", "+", "also", "plus", 
            "as well as", "along with","furthermore",
            "additionally", "moreover"
        ]
        
        # Check for explicit conjunctions
        has_conjunction = any(conj in user_input.lower() for conj in conjunctions)
        
        # Check for multiple distinct issue markers
        issue_markers = [
            ("can't", "isn't"),
            ("unable to", "not working"),
            ("problem with", "issue with"),
            ("error in", "failed to"),
        ]
        
        has_multiple_issues = any(
            all(marker in user_input.lower() for marker in marker_pair)
            for marker_pair in issue_markers
        )
        
        return has_conjunction or has_multiple_issues
    '''
        def process_input(self, user_input: str) -> str:
        try:
            # Clean input text
            user_input = correct_text(user_input)
            
            # First try semantic matching
            intent, example, score = self.semantic_matcher.find_best_match(user_input)
            self.logger.info(f"Semantic matching score: {score} for intent: {intent}")
            
            if score >= self.config.fuzzy_threshold:
                self.logger.info(f"Using high confidence semantic match: {intent} ({score:.2f})")
                return self.response_generator.generate_response(user_input, intent)
            
            # If semantic matching isn't confident enough, try BERT classifier
            bert_intent, bert_confidence = self.intent_classifier.predict(user_input)
            self.logger.info(f"BERT confidence: {bert_confidence} for intent: {bert_intent}")
            
            # Implement intent verification
            verified_intent = self._verify_intent(user_input, intent, bert_intent, score, bert_confidence)
            
            if verified_intent:
                return self.response_generator.generate_response(user_input, verified_intent)
            
            # Fallback response if no confident intent is found
            return self.response_generator.generate_response(user_input)

        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return (
                "I apologize, but I encountered an error processing your request. "
                "Please try again or contact Reeltor support if the issue persists."
            )
    '''
    def process_input(self, user_input: str) -> str:
        try:
            # Clean input text
            user_input = correct_text(user_input)
            if self._is_complex_query(user_input):
                return (
                    "I notice you have multiple issues to discuss. To help you better, "
                    "please provide each query separately. This will allow me to address "
                    "each concern thoroughly.\n\n"
                    "For example, instead of:\n"
                    "'I can't upload photos and search isn't working'\n\n"
                    "Please send as:\n"
                    "1. I can't upload photos\n"
                    "2. Search isn't working"
                )
            intent, example, score = self.semantic_matcher.find_best_match(user_input)
            self.logger.info(f"Semantic matching score: {score} for intent: {intent}")
            
            if score >= self.config.fuzzy_threshold:
                self.logger.info(f"Using high confidence semantic match: {intent} ({score:.2f})")
                return self.response_generator.generate_response(user_input, intent)
            
            bert_intent, bert_confidence = self.intent_classifier.predict(user_input)
            self.logger.info(f"BERT confidence: {bert_confidence} for intent: {bert_intent}")
            
            verified_intent = self._verify_intent(user_input, intent, bert_intent, score, bert_confidence)
            
            if verified_intent:
                return self.response_generator.generate_response(user_input, verified_intent)
            
            return self.response_generator.generate_response(user_input)

        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return (
                "I apologize, but I encountered an error processing your request. "
                "Please try again or contact Reeltor support if the issue persists."
            )


    def _verify_intent(
        self, 
        user_input: str, 
        semantic_intent: str, 
        bert_intent: str, 
        semantic_score: float, 
        bert_confidence: float
    ) -> Optional[str]:
        """Enhanced intent verification with comprehensive keyword patterns."""
        
        # Define weights for different methods
        SEMANTIC_WEIGHT = 0.6
        BERT_WEIGHT = 0.4
        
        # Calculate weighted scores
        semantic_weighted = semantic_score * SEMANTIC_WEIGHT
        bert_weighted = bert_confidence * BERT_WEIGHT
        
        # If both methods agree, use that intent
        if semantic_intent == bert_intent:
            return semantic_intent
            
        # If semantic score is very high, trust it
        if semantic_score > 0.95:
            return semantic_intent
            
        # If BERT confidence is very high, trust it
        if bert_confidence > 0.95:
            return bert_intent
            
        # Enhanced keyword patterns for accurate intent matching
        user_input_lower = user_input.lower()
        
        # Account Login and Profile Management Keywords
        account_keywords = {
            "change mobile", "change number", "update number", "new number", 
            "registered mobile", "phone number", "contact number",
            "email verification", "verify email", "email pending",
            "update profile", "change profile", "modify profile",
            "profile information", "account settings", "contact details",
            "mobile update", "phone update", "number change",
            "account locked", "can't login", "login failed",
            "password reset", "forgot password", "reset link",
            "account blocked", "verification pending", "verify account"
        }
        if any(keyword in user_input_lower for keyword in account_keywords):
            return "Account_Login"
            
        # Property Search Keywords
        search_keywords = {
            "search not working", "can't find", "no results", "search error",
            "recommendations", "recommended", "suggestions", "explore",
            "rent vs buy", "rental option", "buying option", "property type",
            "nearby", "area search", "location filter", "locality",
            "no properties", "empty results", "timeout", "search failed",
            "favorites", "saved properties", "saved searches", "bookmarks",
            "commercial property", "property filter", "filter options",
            "search criteria", "property list", "search results",
            "find properties", "property search", "search not responding",
            "location search", "area filter", "location settings",
            "search timeout", "results empty", "no listings"
        }
        if any(keyword in user_input_lower for keyword in search_keywords):
            return "Property_Search"
            
        # Property Upload Keywords
        upload_keywords = {
            "upload", "posting", "listing", "create post",
            "image upload", "photo size", "picture format", "image limit",
            "maximum images", "image count", "photo limit", "6 images",
            "post rejected", "information update", "add amenities",
            "wrong location", "address save", "pin error", "location wrong",
            "property details", "listing details", "property information",
            "photo upload", "image error", "upload failed", "can't upload",
            "listing creation", "post property", "add property",
            "property specs", "amenities list", "location pin",
            "address wrong", "map location", "property location"
        }
        if any(keyword in user_input_lower for keyword in upload_keywords):
            return "Property_Upload"
            
        # Reels Features Keywords
        reels_keywords = {
            "reels not playing", "black screen", "no loading", "no sound",
            "swipe", "stuck", "owner details", "property info",
            "reel details", "video info", "content visible", "video blank",
            "smooth playing", "buffering", "loading", "video frozen",
            "app freeze", "app hang", "not responding", "video stuck",
            "reel navigation", "video quality", "playback issue",
            "sound problem", "audio missing", "video error",
            "reel content", "property details", "video details",
            "swipe right", "video play", "reel stopped",
            "loading error", "playback stopped", "video buffer"
        }
        if any(keyword in user_input_lower for keyword in reels_keywords):
            return "Reels_Features"
            
        # Chat Features Keywords
        chat_keywords = {
            "message", "chat", "send", "messaging", "can't send",
            "chat history", "previous chats", "old messages", "past chats",
            "chat notification", "message alert", "chat sound", "alerts",
            "chat crash", "chat respond", "chat blank", "chat window",
            "loading messages", "connection error", "message failed",
            "chat backup", "message history", "conversation history",
            "property owner", "send message", "message delivery",
            "chat working", "message system", "chat feature",
            "notification sound", "message notifications", "chat alerts",
            "chat section", "messaging system", "communication"
        }
        if any(keyword in user_input_lower for keyword in chat_keywords):
            return "Chat_Features"
            
        # Community Features Keywords
        community_keywords = {
            "create community", "community setup", "start community",
            "post community", "share community", "community content",
            "join community", "membership", "access denied", "can't join",
            "community notification", "community alert", "community updates",
            "find communities", "discover communities", "browse communities",
            "community post", "community page", "feed update",
            "community setup", "create group", "start group",
            "community access", "member access", "join request",
            "post approval", "community rules", "group settings",
            "community search", "find groups", "discover groups",
            "post failed", "content posting", "community feed"
        }
        if any(keyword in user_input_lower for keyword in community_keywords):
            return "Community_Features"
            
        # App Performance Keywords
        performance_keywords = {
            "slow", "lag", "performance", "loading slow", "app slow",
            "crash", "shutdown", "unstable", "not stable", "keeps crashing",
            "freeze", "frozen", "not respond", "hang", "app freeze",
            "loading", "stuck", "not working", "app hang",
            "app performance", "response time", "app speed",
            "application crash", "system error", "app error",
            "screen frozen", "controls stuck", "unresponsive",
            "app status", "loading screen", "startup issue",
            "app closing", "force close", "app restart"
        }
        if any(keyword in user_input_lower for keyword in performance_keywords):
            return "App_Performance"
            
        # Location Issues Keywords
        location_keywords = {
            "location access", "gps", "location settings", "location error",
            "wrong location", "incorrect location", "location accuracy",
            "area detection", "location permission", "location denied",
            "location service", "map error", "location pin",
            "current location", "my location", "location update",
            "area search", "nearby search", "location filter",
            "distance search", "radius search", "location based"
        }
        if any(keyword in user_input_lower for keyword in location_keywords):
            return "Location_Issues"
            
        # If confidence levels are too low for both, check semantic fallback
        if semantic_score > self.config.fuzzy_fallback_threshold:
            return semantic_intent
            
        # If all checks fail, return None to trigger fallback response
        return None   
    def _get_intent_specific_context(self, intent: str, user_input: str) -> Dict[str, Any]:
        """Get additional context information for specific intents."""
        context = {}
        user_input_lower = user_input.lower()
        
        if intent == "Property_Search":
            context["location_based"] = any(word in user_input_lower for word in ["near", "nearby", "area", "location"])
            context["filter_based"] = any(word in user_input_lower for word in ["filter", "budget", "bhk", "type"])
            context["results_based"] = any(word in user_input_lower for word in ["no results", "empty", "not found"])
            
        elif intent == "Property_Upload":
            context["image_related"] = any(word in user_input_lower for word in ["image", "photo", "picture", "upload"])
            context["visibility_related"] = any(word in user_input_lower for word in ["not visible", "can't see", "hidden"])
            context["posting_related"] = any(word in user_input_lower for word in ["post", "create", "add", "new"])
            
        elif intent == "Reels_Features":
            context["playback_issues"] = any(word in user_input_lower for word in ["not playing", "stuck", "frozen"])
            context["performance_issues"] = any(word in user_input_lower for word in ["freeze", "crash", "slow"])
            context["content_issues"] = any(word in user_input_lower for word in ["details", "info", "missing"])
            
        elif intent == "Chat_Features":
            context["history_related"] = any(word in user_input_lower for word in ["history", "old", "previous"])
            context["sending_related"] = any(word in user_input_lower for word in ["send", "deliver", "message"])
            context["notification_related"] = any(word in user_input_lower for word in ["notification", "alert", "update"])
            
        return context

    def _enhance_response(self, response: str, intent: str, context: Dict[str, Any]) -> str:
        """Enhance response based on specific context."""
        if not context:
            return response
            
        enhanced_response = response
        
        if intent == "Property_Search" and context.get("location_based"):
            enhanced_response += "\n\nAdditional location-based tips:\n• Enable GPS for better results\n• Try expanding search radius\n• Use landmarks for precise location"
            
        elif intent == "Property_Upload" and context.get("image_related"):
            enhanced_response += "\n\nPhoto upload tips:\n• Use landscape orientation\n• Ensure good lighting\n• Keep file size under 5MB"
            
        elif intent == "Reels_Features" and context.get("performance_issues"):
            enhanced_response += "\n\nPerformance optimization tips:\n• Close background apps\n• Clear app cache\n• Check internet speed"
            
        elif intent == "Chat_Features" and context.get("history_related"):
            enhanced_response += "\n\nMessage recovery tips:\n• Check archived chats\n• Sync account data\n• Contact support for backup retrieval"
            
        return enhanced_response
'''
def main():
    config = ChatbotConfig()
    
    # Define intents and responses
    NEW_INTENTS = {
        "Greeting": [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "hi there",
            "hello there",
            "greetings",
            "howdy",
            "welcome",
            "hey there",
            "hi reeltor",
            "hello reeltor"
        ],

        "Account_Login": [
            # Basic Login Issues
            "Can't log into my account",
            "Login not working",
            "Unable to sign in",
            "Login page error",
            "Account access problem",
            
            # Gmail/Google Issues
            "Gmail login not working",
            "Google sign in failed",
            "Cannot connect Google account",
            "Google authentication error",
            "Google login stuck loading",
            "Unable to sign in with Gmail",
            "Google account not connecting",
            
            # Apple ID Issues
            "Apple ID sign in failed",
            "Can't login with Apple",
            "Apple login not working after iOS update",
            "Apple ID verification failed",
            "Apple login error",
            "iOS login problems",
            
            # Password Related
            "How do I reset my password?",
            "Password reset not working",
            "Forgot password link not received",
            "Can't change password",
            "Password not accepted",
            "Reset link expired",
            "How to create new password",
            
            # OTP/Verification
            "Not getting OTP for login",
            "OTP not received",
            "Verification code missing",
            "SMS code not working",
            "OTP expired too quickly",
            "Wrong OTP error",
            "Verification failed",
            
            # Account Issues
            "My account is locked",
            "How to change registered mobile number",
            "Email verification pending",
            "Account verification failed",
            "Can't update profile information",
            "Unable to delete account",
            "Account blocked message",

            "Login page stuck on loading",
            "Login screen not responding",
            "Sign in page frozen",
            "Authentication page stuck",
            "Login process not completing",
            "Sign in loading forever",
            "Cannot proceed past login screen",
            "Login verification stuck",
        ],
        "Property_Management": [
            "Unable to mark property as sold",
            "Can't update property status to sold",
            "Property status not changing to sold",
            "Sold marking not working",
            "Cannot update listing status",
            "Property dimensions showing wrong",
            "Incorrect property measurements",
            "Wrong dimensions in listing",
            "Floor plan not visible",
            "Cannot see property layout",
            "Floor plan missing from listing",
            "Property blueprint not showing"
],

        "Property_Search": [
            # Basic Search
            "How to search properties",
            "Search not working",
            "Can't find properties",
            "No search results",
            "Search error",

            "Can't find commercial properties",
            "Commercial property search not working",
            "Unable to find showroom",
            "No commercial spaces showing",
            "Office space search error",
            "Shop/showroom search not working",
            "Commercial property filter issues",
            "Business property not found",
            "Commercial listings not visible",
            "Retail space search problems",

                # Recommendations/Explore
            "No recommendations showing",
            "Explore page empty",
            "Recommended properties not showing",
            "Explore section not working",
            "No suggestions in explore",
            "Property recommendations missing",
            "Can't see similar properties",
            "Top locality section empty",
            "Featured properties not visible",
            "Trending properties not showing"
            
            # Filters
            "Filter not working",
            "How to filter properties by budget?",
            "Location filter not showing my area",
            "Can't find 3BHK properties",
            "Where to set rent vs buy option?",
            "Filter for PG not working",
            "How to search for villa only?",
            "Budget range not updating",
            "Property type filter error",
            
            # Location Based
            "Location search error",
            "No properties showing in nearby section",
            "Top locality list is empty",
            "Can't see properties around me",
            "How to change my location?",
            "Area search not working",
            "Location filter stuck",
            
            # Results
            "Search results not loading",
            "No properties found",
            "Results not updating",
            "Can't see search results",
            "Property list empty",
            "Search timeout error",
            
            # Saved Searches
            "Where are my saved properties?",
            "Saved search missing",
            "Can't save search",
            "Favorites not showing",
            "How to save property search",
            "Saved properties disappeared"


        ],


        "Property_Upload": [
            # Upload Process
            "How to post my property?",
            "Can't upload property",
            "Property posting guide",
            "Listing creation failed",
            "Upload error",
            
            # Images
            "Image upload failed while posting",
            "Can't upload property photos",
            "Property images not uploading",
            "Photo size error",
            "Image format not supported",
            "Maximum images limit",
            
            # Listing Management
            "Can't edit my property listing",
            "How to modify property details",
            "Can't update property info",
            "Editing not working",
            "Unable to change listing",
            
            # Status Issues
            "Property post stuck in processing",
            "Post under review for long time",
            "Listing not visible",
            "Property not appearing",
            "Post rejected",
            
            # Property Details
            "Can't add property details",
            "Details not saving",
            "Information update failed",
            "Property specs error",
            "Unable to add amenities",
            
            # Location
            "Location wrong in my post",
            "Can't set property location",
            "Address not saving",
            "Location pin error",
            "Map not working",

            "Image quality reduced after posting",
            "Photos losing quality when uploaded",
            "Property pictures blurry after upload",
            "Images not uploading in full resolution",
            "Picture quality degraded after posting",
            "Photo resolution reduced",
            "High-res images getting compressed",
            "Property photos pixelated after upload",
        ],

        "Reels_Features": [
            # Playback Issues
            "Reels not playing",
            "Video stuck loading",
            "Black screen in reels",
            "Reels not loading in my area",
            "Videos frozen",
            "No sound in property videos"
            
            # Navigation
            "When I swipe right, property details are blank",
            "Can't swipe reels",
            "Swipe not working",
            "Unable to navigate reels",
            "Reels stuck",
            
            # Content Display
            "Can't see owner details in reel",
            "Property info not showing",
            "Reel details missing",
            "Video information blank",
            "Content not visible",
            
            # Quality Issues
            "Video quality is poor in reels",
            "Videos not playing smoothly",
            "Low resolution",
            "Blurry videos",
            "Quality settings not working",
            
            # Loading Problems
            "Reels stuck on loading",
            "Videos not loading",
            "Infinite loading screen",
            "Reel buffering",
            "Content loading failed",
            
            # App Performance
            "App freezes when opening reels",
            "Reel section crash",
            "App hangs in reels",
            "Performance issues",
            "Reels not responding"
        ],

        "Chat_Features": [
            # Messaging
            "Unable to message property owner",
            "Can't send messages",
            "Message not delivering",
            "Chat not working",
            "Sending failed",
            
            # History/Storage
            "Chat history disappeared",
            "Where to find previous chats?",
            "Old messages gone",
            "Can't see past conversations",
            "Chat backup missing",
            
            # Notifications
            "Not getting chat notifications",
            "Message alerts not working",
            "Chat sound not working",
            "No notification for new messages",
            "Missing chat updates",
            
            # Media Sharing
            "Can't send images in chat",
            "Unable to share files",
            "Media not sending",
            "Photo sharing failed",
            "Document upload error",
            
            # Technical Issues
            "Chat section keeps crashing",
            "Chat window blank",
            "Messages not loading",
            "Chat feature not responding",
            "Connection error in chat"
        ],

        "Community_Features": [
            # Creation
            "How to create a new community?",
            "Community creation failed",
            "Can't make new community",
            "Community setup error",
            "Unable to start community",
            
            # Posting
            "Can't post in community section",
            "Post not publishing",
            "Unable to share in community",
            "Community post failed",
            "Content not posting",
            
            # Viewing/Loading
            "Community posts not loading",
            "Can't see community content",
            "Feed not updating",
            "Community page blank",
            "Posts disappearing",
            
            # Membership
            "Unable to join local community",
            "Can't join community",
            "Membership error",
            "Access denied to community",
            "Join request failed",
            
            # Notifications
            "Not getting community notifications",
            "Community alerts not working",
            "Missing post notifications",
            "Update alerts not showing",
            "Notification settings issue",
            
            # Search/Discovery
            "Community search not working",
            "Can't find communities",
            "Search results empty",
            "Discovery not working",
            "Browse communities error",

            "Member approval requests not showing",
            "Can't see pending members",
            "Community join requests missing",
            "Member requests not visible",
            "Community analytics not updating",
            "Community stats not refreshing",
            "Group metrics not showing",
            "Analytics dashboard frozen"
        ],

        "App_Performance": [
            # Speed Issues
            "App very slow",
            "Slow response time",
            "App lagging",
            "Performance very poor",
            "Everything loading slow",
            
            # Crashes
            "App keeps crashing",
            "Application closes suddenly",
            "Unexpected shutdowns",
            "App not stable",
            "Frequent crashes",
            
            # Freezing
            "App freezes frequently",
            "Screen frozen",
            "App not responding",
            "Controls not working",
            "App hangs",
            
            # Loading
            "Content not loading",
            "Infinite loading",
            "Loading screen stuck",
            "App stuck on loading",
            "Data not loading",
            
            # General Issues
            "App not working properly",
            "Features not responding",
            "App malfunctioning",
            "Technical problems",
            "System errors"
        ],

        "Location_Issues": [
            # Access
            "Location access not working",
            "Can't access location",
            "Location permission denied",
            "GPS not working",
            "Location services error",
            
            # Accuracy
            "Wrong area showing",
            "Location not accurate",
            "Incorrect location detected",
            "Area detection wrong",
            "GPS accuracy issues",
            
            # Search Related
            "Can't search by location",
            "Area search not working",
            "Location filter issues",
            "Can't find nearby properties",
            "Location based search failed",
            
            # Settings
            "How to change location settings",
            "Location preferences not saving",
            "Can't update location",
            "Location setup failed",
            "Settings not working"
        ],

        "Notifications": [
            # Property Alerts
            "Not getting property alerts",
            "Property notifications missing",
            "New listing alerts not working",
            "Price change notifications",
            "Property update alerts",
            
            # Settings
            "Where to manage notification settings?",
            "Can't change notification preferences",
            "Alert settings not saving",
            "Notification setup",
            "Configure alerts",
            
            # Technical Issues
            "Notifications not showing",
            "No alert sounds",
            "Delayed notifications",
            "Missing important alerts",
            "Notification system error",
            
            # Specific Features
            "Chat notification problems",
            "Community alerts not working",
            "System notifications",
            "Custom alert settings",
            "Priority notifications"
        ],

        "General_Help": [
            # Basic Usage
            "How to use this app?",
            "App usage guide",
            "Basic features help",
            "Getting started",
            "New user guide",
            
            # Features
            "What are the app features?",
            "Available functions",
            "App capabilities",
            "Feature overview",
            "Tool guide",
            
            # Support
            "Need help with app",
            "Support required",
            "How to get assistance",
            "Help options",
            "Contact support",
            
            # Information
            "Property terms explanation",
            "Real estate guidelines",
            "App information",
            "Usage instructions",
            "Feature documentation",
            
            # Tips
            "Tips for using app",
            "Best practices",
            "User guidelines",
            "Helpful hints",
            "Usage recommendations"
        ]
    }

    CONTEXTUAL_RESPONSES = {
        "Greeting": {
            "welcome": [
                "👋 Welcome to Reeltor!\n\nI'm here to help you with:\n1. Property search & listings\n2. Account management\n3. App features & settings\n4. Technical support\n5. General inquiries",
                "Hello! 👋 Welcome to Reeltor Support!\n\nHow can I assist you today?\n1. Find properties\n2. Manage your account\n3. Use app features\n4. Get technical help\n5. Ask general questions"
            ],
            "return": [
                "Welcome back to Reeltor! 👋\n\nNeed help with:\n1. Recent searches\n2. Saved properties\n3. Account settings\n4. Property updates\n5. Community features",
                "Hello again! 👋\n\nHow can I assist you today?\n1. Continue property search\n2. Check saved listings\n3. Update preferences\n4. Get support\n5. Explore features"
            ]
        },

"Property_Management": {
    "status_update": [
        "Can't update property status?\n\n1. Status change steps:\n   • Go to listing management\n   • Select property\n   • Click 'Change Status'\n   • Choose 'Sold'\n   • Save changes\n2. If not working:\n   • Clear cache\n   • Try web portal\n   • Contact support",
        
        "Property details incorrect?\n\n1. Update process:\n   • Edit listing\n   • Verify measurements\n   • Update floor plan\n   • Check all fields\n2. Common issues:\n   • Cache problems\n   • Sync errors\n   • Permission issues"
    ],
    "floor_plan": [
        "Floor plan visibility issues?\n\n1. Upload requirements:\n   • File format: PDF/JPG\n   • Size limit: 5MB\n   • Resolution: min 1000x1000\n2. Display fixes:\n   • Clear cache\n   • Reupload plan\n   • Check permissions\n3. Still missing:\n   • Try web upload\n   • Contact support",
        
        "Can't see floor plan?\n\n1. Basic checks:\n   • Refresh listing\n   • Clear cache\n   • Update app\n2. Upload again:\n   • Check format\n   • Verify size\n   • Monitor upload"
    ]
},

"Account_Login": {
            "apple": [
                "For Apple ID login issues:\n\n1. Check Apple ID status:\n   • Verify account is active\n   • Check iOS version\n   • Sign out/in of Apple ID\n2. App troubleshooting:\n   • Clear app cache\n   • Update Reeltor app\n   • Reinstall if needed\n3. Verification fixes:\n   • Check two-factor settings\n   • Verify email/phone\n   • Update security",
                
                "Apple ID sign-in failed?\n\n1. Update iOS device\n2. Check Apple ID settings\n3. Verify verification methods\n4. Reset Apple ID password\n5. Contact Apple support if needed"
            ],
            "gmail": [
                "For gmail  login issues:\n\n1. Check gmail status:\n   • Verify account is active\n   • Check iOS version\n   • Sign out/in of gmail \n2. App troubleshooting:\n   • Clear app cache\n   • Update Reeltor app\n   • Reinstall if needed\n3. Verification fixes:\n   • Check two-factor settings\n   • Verify email/phone\n   • Update security",
                
                "Gmail sign-in failed?\n\n1. Update iOS device\n2. Check Gmail settings\n3. Verify verification methods\n4. Reset Gmail password\n5. Contact Google support if needed"
            ],
            "password": [
                "Password reset problems?\n\n1. Reset process:\n   • Click 'Forgot Password'\n   • Check all email folders\n   • Use link within 24 hours\n2. New password tips:\n   • Use strong combination\n   • Avoid previous passwords\n   • Follow requirements\n3. If link expired:\n   • Request new link\n   • Contact support",
                
                "Password issues?\n\n1. Request new reset link\n2. Check spam folder\n3. Create secure password\n4. Clear browser cache\n5. Contact support if needed"
            ],
            "account": [
        "Account management issues?\n\n1. Change mobile number:\n   • Go to Profile Settings\n   • Select 'Update Mobile'\n   • Enter new number\n   • Verify with OTP\n2. Profile updates:\n   • Change contact details\n   • Update email\n   • Verify changes\n3. Account actions:\n   • Complete verification\n   • Save changes\n   • Contact support",
        
        "Profile/Account issues?\n\n1. Open settings\n2. Update details\n3. Verify changes\n4. Complete OTP\n5. Save changes"
    ],
            "mobile_update": [
        "Change registered mobile number:\n\n1. Access settings:\n   • Open Profile page\n   • Go to Contact Details\n   • Select Change Number\n2. Update process:\n   • Enter new number\n   • Get OTP verification\n   • Confirm changes\n3. Troubleshooting:\n   • Check format (with country code)\n   • Verify OTP delivery\n   • Contact support if needed",
        
        "Update mobile number:\n\n1. Go to Profile\n2. Select Change Number\n3. Enter new number\n4. Verify with OTP\n5. Save changes"
    ],
    "loading_stuck": [
        "Login page stuck?\n\n1. Immediate fixes:\n   • Force close app\n   • Clear cache\n   • Check internet\n2. Authentication steps:\n   • Try different login method\n   • Reset app data\n   • Update app\n3. If persisting:\n   • Reinstall app\n   • Contact support\n   • Try web login",
        
        "Login screen frozen?\n\n1. Basic fixes:\n   • Restart app\n   • Check connection\n   • Clear cache\n2. Advanced fixes:\n   • Reset app data\n   • Try different device\n   • Use web version"
    ],
            "otp": [
                "OTP/Verification issues?\n\n1. Check basics:\n   • Correct phone number\n   • SMS permissions\n   • Network connection\n2. Troubleshooting:\n   • Request new code\n   • Check blocked messages\n   • Try alternate number\n3. Other options:\n   • Email verification\n   • Contact support",
                
                "Verification problems?\n\n1. Verify phone number\n2. Check SMS settings\n3. Try alternate method\n4. Clear app cache\n5. Contact support"
            ],
            "account": [
                "Account management issues?\n\n1. Access problems:\n   • Verify credentials\n   • Check account status\n   • Clear app data\n2. Profile updates:\n   • Change mobile number\n   • Update email\n   • Verify changes\n3. Account actions:\n   • Unblock account\n   • Delete account\n   • Contact support",
                
                "Profile/Account issues?\n\n1. Check login status\n2. Update profile info\n3. Verify changes saved\n4. Clear app data\n5. Contact support"
            ],
            "email_verification": [
        "Email verification pending?\n\n1. Check status:\n   • Look for verification email\n   • Check spam folder\n   • Verify email address\n2. Verification steps:\n   • Click verification link\n   • Follow instructions\n   • Refresh status\n3. If not received:\n   • Request new email\n   • Contact support\n   • Check correct email",
        
        "Email not verified?\n\n1. Check inbox/spam\n2. Click verify link\n3. Request new email\n4. Contact support\n5. Update email"
    ],
     "profile_update": [
        "Profile update issues?\n\n1. Update steps:\n   • Go to Profile Settings\n   • Choose field to update\n   • Enter new information\n   • Save changes\n2. Common problems:\n   • Fields not saving\n   • Validation errors\n   • Upload issues\n3. Solutions:\n   • Try again later\n   • Clear cache\n   • Contact support",
        
        "Can't update profile?\n\n1. Check connection\n2. Clear cache\n3. Try web version\n4. Save changes\n5. Contact support"
    ]
        
        },
        "Property_Search": {
            "guide": [
                "How to search properties effectively:\n\n1. Set location (city/locality)\n2. Choose property type:\n   • Apartment/Villa/PG\n   • BHK configuration\n3. Set budget range\n4. Apply filters:\n   • Furnishing\n   • Availability\n   • Amenities\n5. Save search for updates",
                
                "Property search tips:\n\n1. Use specific filters\n2. Save preferred searches\n3. Enable notifications\n4. Check nearby areas\n5. Compare similar properties"
            ],

            "search_error": [
        "Search error help:\n\n1. Common issues:\n   • Connection problems\n   • Filter conflicts\n   • Location errors\n2. Quick fixes:\n   • Clear all filters\n   • Refresh page\n   • Check internet\n3. Advanced solutions:\n   • Clear app cache\n   • Update app\n   • Try web version",
        
        "Search not working?\n\n1. Check connection\n2. Clear filters\n3. Refresh page\n4. Update app\n5. Try again"
    ],
    "no_results": [
        "No properties found?\n\n1. Adjust filters:\n   • Broaden price range\n   • Expand location\n   • Remove restrictions\n2. Search tips:\n   • Use fewer filters\n   • Check spelling\n   • Try nearby areas\n3. Other options:\n   • Save search\n   • Enable alerts\n   • Contact support",
        
        "Empty results?\n\n1. Broaden search\n2. Check filters\n3. Try nearby areas\n4. Save search\n5. Get alerts"
    ],
    "recommendations": [
        "Missing recommendations?\n\n1. Check settings:\n   • Update preferences\n   • Enable recommendations\n   • Location services\n2. Content issues:\n   • Refresh page\n   • Clear cache\n   • Update filters\n3. Still missing:\n   • Contact support\n   • Try web version",
        
        "No recommendations?\n\n1. Set preferences\n2. Enable location\n3. Refresh page\n4. Clear cache\n5. Update app"
    ],
    "rent_buy": [
        "Set rent/buy options:\n\n1. Main filters:\n   • Top filter bar\n   • Select purpose\n   • Choose rent/buy\n2. Advanced options:\n   • Price range\n   • Property type\n   • Available dates\n3. Save preferences:\n   • Update filters\n   • Save search\n   • Enable alerts",
        
        "Rent vs Buy filter:\n\n1. Open filters\n2. Select purpose\n3. Set preferences\n4. Apply filters\n5. Save search"
    ],
    "location_issues": [
        "Location search problems?\n\n1. Settings check:\n   • Enable location\n   • Grant permissions\n   • Check accuracy\n2. Search options:\n   • Enter manually\n   • Use landmarks\n   • Try map view\n3. Troubleshooting:\n   • Clear cache\n   • Update app\n   • Reset location",
        
        "Location not working?\n\n1. Enable GPS\n2. Update location\n3. Try manual entry\n4. Clear cache\n5. Contact support"
    ],

    "filter": [
                "Using search filters:\n\n1. Property type filters:\n   • Select BHK (1/2/3/4+)\n   • Choose Villa/PG/Apartment\n   • Set budget range\n2. Location filters:\n   • Select area/locality\n   • Use map view\n   • Set radius\n3. Price filters:\n   • Min-max budget\n   • Sort by price\n4. Reset if needed",
                
                "Filter not working?\n\n1. Clear current filters\n2. Try basic search first\n3. Update preferences\n4. Check app version\n5. Contact support"
            ],
            "results": [
                "Search results issues?\n\n1. No results found:\n   • Broaden criteria\n   • Check filters\n   • Expand location\n2. Loading problems:\n   • Check internet\n   • Clear cache\n   • Update app\n3. Try alternatives:\n   • Different location\n   • Adjust filters",
                
                "Results not showing?\n\n1. Check all filters\n2. Expand search area\n3. Update application\n4. Clear app data\n5. Try web version"
            ],
            "location": [
                "Location search problems?\n\n1. Location access:\n   • Enable GPS\n   • Grant permissions\n   • Check accuracy\n2. Search options:\n   • Use landmarks\n   • Try area name\n   • Use pincode\n3. No results?\n   • Expand radius\n   • Check spelling",
                
                "Area search issues?\n\n1. Verify GPS settings\n2. Enable precise location\n3. Try manual entry\n4. Check nearby areas\n5. Update app"
            ],
            "saved": [
                "Managing saved searches:\n\n1. Access saved items:\n   • Check Favorites tab\n   • View saved searches\n   • Recent history\n2. Missing saves:\n   • Verify login\n   • Sync account\n   • Clear cache\n3. Restore searches:\n   • Reload app\n   • Contact support",
                
                "Can't find saves?\n\n1. Verify account login\n2. Check saved section\n3. Sync data\n4. Clear cache\n5. Update app"
            ],
         "commercial": [
        "Commercial property search issues?\n\n1. Search options:\n   • Use property type filter\n   • Select commercial category\n   • Choose specific type:\n     - Showroom\n     - Office\n     - Retail space\n2. Filter settings:\n   • Area requirements\n   • Budget range\n   • Location preferences\n3. If no results:\n   • Expand search radius\n   • Modify requirements\n   • Try different localities",
        
        "No commercial properties?\n\n1. Check filters:\n   • Property category\n   • Commercial type\n   • Size requirements\n2. Search tips:\n   • Use landmark search\n   • Contact local agents\n   • Try nearby areas\n3. Advanced search:\n   • Use map view\n   • Save search\n   • Set alerts"
    ],
    "recommendations": [
        "Recommendation issues?\n\n1. Explore page:\n   • Check internet connection\n   • Pull to refresh\n   • Update preferences\n2. Visibility settings:\n   • Location services\n   • Search history\n   • Preferences\n3. Top localities:\n   • Update location\n   • Clear app cache\n   • Reset preferences",
        
        "No suggestions showing?\n\n1. Basic fixes:\n   • Refresh page\n   • Clear cache\n   • Update app\n2. Preferences:\n   • Set search criteria\n   • Update location\n   • Choose interests\n3. Alternative options:\n   • Browse categories\n   • Use search filters\n   • Check saved searches"
    ]
        },



        "Property_Upload": {
                "guide": [
                    "Property posting guide:\n\n1. Basic Information:\n   • Property type\n   • Location details\n   • Price and area\n2. Detailed Specifications:\n   • Room configuration\n   • Amenities available\n   • Furnishing status\n3. Photo Guidelines:\n   • Clear, well-lit images\n   • All major areas\n   • Max size: 5MB each\n4. Additional Details:\n   • Possession status\n   • Available from date\n   • Legal documents",
                    
                    "Posting steps:\n\n1. Start new listing\n2. Fill all sections\n3. Add quality photos\n4. Review details\n5. Submit for approval"
                ],
                    "image_quality": [
        "Image quality issues?\n\n1. Upload guidelines:\n   • Max size: 10MB per image\n   • Recommended resolution: 1920x1080\n   • Format: JPG/PNG\n2. Quality tips:\n   • Use original photos\n   • Avoid re-compression\n   • Check upload settings\n3. If persisting:\n   • Try web upload\n   • Contact support",
        
        "Photo resolution reduced?\n\n1. Photo requirements:\n   • Use high-res images\n   • Follow size limits\n   • Check format\n2. Upload process:\n   • One image at a time\n   • Monitor compression\n   • Verify preview"
    ],


                    "upload_error": [
        "Upload problems?\n\n1. Basic checks:\n   • Internet connection\n   • File sizes\n   • Image formats\n2. Upload process:\n   • Try one by one\n   • Save as draft\n   • Check requirements\n3. If failing:\n   • Clear cache\n   • Update app\n   • Try web version",
        
        "Can't upload?\n\n1. Check connection\n2. Verify formats\n3. Try individually\n4. Clear cache\n5. Use web version"
    ],
    "image_limits": [
        "Image upload guidelines:\n\n1. Requirements:\n   • Maximum 6 images\n   • Size: 5MB per image\n   • Format: JPG/PNG\n2. Best practices:\n   • Good lighting\n   • Clear photos\n   • Main features\n3. Common issues:\n   • Reduce size\n   • Proper format\n   • Try individually",
        
        "Photo limits:\n\n1. Max 6 images\n2. 5MB each\n3. JPG/PNG only\n4. Clear quality\n5. Try one by one"
    ],
    "listing_failed": [
        "Listing creation failed?\n\n1. Required fields:\n   • Property details\n   • Location info\n   • Contact details\n2. Common errors:\n   • Missing data\n   • Invalid format\n   • Upload issues\n3. Solutions:\n   • Save draft\n   • Complete all fields\n   • Try again",
        
        "Creation error?\n\n1. Check all fields\n2. Valid location\n3. Save progress\n4. Try again\n5. Contact support"
    ],
    "post_rejection": [
        "Post rejected?\n\n1. Common reasons:\n   • Incomplete info\n   • Invalid photos\n   • Policy violation\n2. Fix issues:\n   • Review feedback\n   • Update content\n   • Check guidelines\n3. Resubmission:\n   • Verify all details\n   • Quality photos\n   • Contact support",
        
        "Listing rejected?\n\n1. Read feedback\n2. Fix issues\n3. Update content\n4. Check rules\n5. Try again"
    ],
    "amenities_error": [
        "Can't add amenities?\n\n1. Selection issues:\n   • Check categories\n   • Proper selection\n   • Save changes\n2. Common problems:\n   • Loading error\n   • Not saving\n   • Missing options\n3. Solutions:\n   • Refresh page\n   • Clear cache\n   • Try web version",
        
        "Amenities problem?\n\n1. Select properly\n2. Save changes\n3. Refresh page\n4. Clear cache\n5. Try again"
    ],
    "location_error": [
        "Location pin issues?\n\n1. Setting location:\n   • Enable GPS\n   • Search address\n   • Drop pin\n2. Common problems:\n   • Wrong coordinates\n   • Pin not saving\n   • Map errors\n3. Solutions:\n   • Manual entry\n   • Use landmarks\n   • Contact support",
        
        "Address problems?\n\n1. Enable GPS\n2. Manual entry\n3. Use landmarks\n4. Save location\n5. Try again"
    ],

                "image": [
                    "Image upload issues?\n\n1. Check image specifications:\n   • Max size: 5MB per image\n   • Format: JPG/PNG only\n   • Resolution: min 800x600\n2. Upload solutions:\n   • Reduce image size\n   • Try one by one\n   • Clear cache first\n3. If still failing:\n   • Use different images\n   • Try web upload",
                    
                    "Photo upload tips:\n\n1. Optimize image size\n2. Check format support\n3. Upload individually\n4. Verify requirements\n5. Clear cache first"
                ],
                "listing": [
                    "Property listing problems?\n\n1. Creation steps:\n   • Fill mandatory fields\n   • Add property details\n   • Upload photos\n   • Set location\n2. Error handling:\n   • Save as draft\n   • Check validation\n   • Verify details\n3. If stuck:\n   • Clear cache\n   • Try web version",
                    
                    "Posting issues?\n\n1. Complete all fields\n2. Add required photos\n3. Verify location\n4. Save progress\n5. Check guidelines"
                ],
                "editing": [
                    "Editing property listing:\n\n1. Access options:\n   • Go to My Listings\n   • Select property\n   • Choose Edit\n2. Update process:\n   • Modify details\n   • Update photos\n   • Change specs\n3. Save changes:\n   • Review updates\n   • Confirm changes",
                    
                    "Can't edit listing?\n\n1. Check permissions\n2. Clear app cache\n3. Try section by section\n4. Save frequently\n5. Use web version"
                ],
                "details": [
                    "Property details issues?\n\n1. Required info:\n   • Basic details\n   • Property specs\n   • Amenities list\n   • Location data\n2. Saving tips:\n   • Regular saves\n   • Section by section\n   • Verify changes\n3. Troubleshooting:\n   • Clear cache\n   • Check format",
                    
                    "Details not saving?\n\n1. Save in sections\n2. Check all fields\n3. Verify format\n4. Update app\n5. Try web version"
                ],
                "visibility": [
                    "Property not visible?\n\n1. Check status:\n   • Under review\n   • Published\n   • Rejected\n2. Requirements:\n   • Complete info\n   • Valid photos\n   • Correct location\n3. Next steps:\n   • Contact support\n   • Check guidelines",
                    
                    "Listing hidden?\n\n1. Verify status\n2. Complete all fields\n3. Check requirements\n4. Update details\n5. Contact support"
                ],
                "status": [
                    "Posting status issues?\n\n1. Review process:\n   • Check queue\n   • Verify requirements\n   • Wait time info\n2. If rejected:\n   • Read feedback\n   • Fix issues\n   • Resubmit\n3. Stuck processing:\n   • Contact support\n   • Check guidelines",
                    
                    "Status problems?\n\n1. Check current status\n2. Review requirements\n3. Fix any issues\n4. Resubmit if needed\n5. Contact support"
                ]
            },

    "Reels_Features": {
                "playback": [
                    "Reels not playing?\n\n1. Connection requirements:\n   • 4G/WiFi connection\n   • Stable internet\n   • Good signal strength\n2. App solutions:\n   • Clear cache\n   • Update app\n   • Restart app\n3. Device fixes:\n   • Free up storage\n   • Close other apps",
                    
                    "Video playback issues?\n\n1. Check internet speed\n2. Clear app cache\n3. Update application\n4. Free up memory\n5. Try different network"
                ],
                    "playback_issues": [
        "Reels not playing?\n\n1. Connection checks:\n   • Internet stability\n   • Signal strength\n   • Network type\n2. App solutions:\n   • Clear cache\n   • Update app\n   • Restart app\n3. Device fixes:\n   • Free storage\n   • Close apps\n   • System update",
        
        "Video problems?\n\n1. Check internet\n2. Clear cache\n3. Update app\n4. Restart app\n5. Free storage"
    ],
    "display_problems": [
        "Black screen/blank details?\n\n1. Viewing issues:\n   • Refresh feed\n   • Clear cache\n   • Check connection\n2. Content loading:\n   • Wait briefly\n   • Reload app\n   • Update app\n3. If persisting:\n   • Reinstall app\n   • Contact support",
        
        "Display issues?\n\n1. Refresh feed\n2. Clear cache\n3. Check internet\n4. Update app\n5. Try again"
    ],
    "audio_issues": [
        "No sound in reels?\n\n1. Basic checks:\n   • Volume settings\n   • Mute button\n   • Media volume\n2. App solutions:\n   • Restart app\n   • Clear cache\n   • Update app\n3. Device fixes:\n   • Sound settings\n   • System restart\n   • App permissions",
        
        "Sound problems?\n\n1. Check volume\n2. Unmute video\n3. Clear cache\n4. Restart app\n5. Update app"
    ],
                "loading": [
                    "Reels loading problems?\n\n1. Network checks:\n   • Internet speed\n   • Connection stability\n   • Network switch\n2. App fixes:\n   • Clear cache\n   • Update app\n   • Force stop\n3. Memory issues:\n   • Free storage\n   • Close apps",
                    
                    "Content not loading?\n\n1. Verify connection\n2. Clear app data\n3. Check storage\n4. Update app\n5. Restart device"
                ],
                "navigation": [
                    "Reel Navigation issues?\n\n1. Swipe problems:\n   • Check screen\n   • Clear cache\n   • Update app\n2. Content issues:\n   • Refresh feed\n   • Clear data\n   • Restart app\n3. Details missing:\n   • Check connection\n   • Reload content",
                    
                    "Can't navigate reels?\n\n1. Test touch screen\n2. Clear cached data\n3. Update app\n4. Free up memory\n5. Try reinstalling"
                ],
                "performance": [
                    "Reel performance issues?\n\n1. App problems:\n   • Clear cache\n   • Update app\n   • Check storage\n2. System issues:\n   • Close apps\n   • Restart device\n   • Check memory\n3. If freezing:\n   • Force stop\n   • Reinstall",
                    
                    "App freezing in reels?\n\n1. Free up memory\n2. Update app\n3. Clear cache\n4. Check system\n5. Reinstall if needed"
                ],
                "reel_info": [
                    "Reel details missing?\n\n1. Content issues:\n   • Refresh reel\n   • Clear cache\n   • Update app\n2. Display problems:\n   • Check connection\n   • Reload content\n   • Force stop\n3. Still missing:\n   • Report issue\n   • Contact support",
                    
                    "Information not showing?\n\n1. Refresh content\n2. Clear app data\n3. Check connection\n4. Update app\n5. Report problem"
                ]
            },

    "Chat_Features": {
                "sending": [
                    "Message sending issues?\n\n1. Connection checks:\n   • Internet stability\n   • Network type\n   • Signal strength\n2. App solutions:\n   • Clear chat cache\n   • Update app\n   • Force stop\n3. If persisting:\n   • Restart app\n   • Check blocks",
                    
                    "Can't send messages?\n\n1. Verify connection\n2. Clear chat cache\n3. Update app\n4. Check permissions\n5. Try web version"
                ],
                    "messaging_error": [
        "Message sending failed?\n\n1. Connection issues:\n   • Check internet\n   • Signal strength\n   • Network type\n2. App solutions:\n   • Clear cache\n   • Force stop\n   • Update app\n3. Other fixes:\n   • Restart app\n   • Try web version\n   • Contact support",
        
        "Can't send messages?\n\n1. Check internet\n2. Clear cache\n3. Update app\n4. Restart app\n5. Try again"
    ],
    "history_issues": [
        "Chat history missing?\n\n1. Find messages:\n   • Check archives\n   • Search function\n   • Date filters\n2. Recovery options:\n   • Sync account\n   • Clear cache\n   • Restore backup\n3. If lost:\n   • Contact support\n   • Check settings\n   • Try web version",
        
        "Missing chats?\n\n1. Check archives\n2. Search history\n3. Sync account\n4. Clear cache\n5. Contact support"
    ],
    "notification_problems": [
        "Chat notifications issue?\n\n1. Settings check:\n   • App permissions\n   • System settings\n   • Do Not Disturb\n2. App fixes:\n   • Clear cache\n   • Update app\n   • Reset settings\n3. Still missing:\n   • Reinstall app\n   • System update\n   • Contact support",
        
        "No chat alerts?\n\n1. Check settings\n2. Enable alerts\n3. Clear cache\n4. Update app\n5. Try again"
    ],
                "media": [
                    "Media sharing problems?\n\n1. File requirements:\n   • Size limit: 10MB\n   • Supported formats\n   • Good connection\n2. Sending tips:\n   • Compress files\n   • Send individually\n   • Clear cache\n3. Alternatives:\n   • Use web version\n   • Different format",
                    
                    "Can't share files?\n\n1. Check file size\n2. Verify format\n3. Test connection\n4. Send one by one\n5. Try alternatives"
                ],
                "stability": [
                    "Chat stability issues?\n\n1. Performance fixes:\n   • Clear cache\n   • Update app\n   • Free memory\n2. Connection issues:\n   • Check internet\n   • Test network\n   • Restart app\n3. Crashing:\n   • Force stop\n   • Reinstall",
                    
                    "Chat crashing?\n\n1. Clear app data\n2. Close other apps\n3. Update app\n4. Check system\n5. Reinstall if needed"
                ],
                "notifications": [
                    "Chat notification issues?\n\n1. Settings check:\n   • App permissions\n   • System settings\n   • DND mode\n2. Solutions:\n   • Reset settings\n   • Clear cache\n   • Update app\n3. Still missing:\n   • Check background\n   • Contact support",
                    
                    "No alerts?\n\n1. Check permissions\n2. Enable notifications\n3. Update settings\n4. Clear cache\n5. Restart app"
                ],
                "connection": [
                    "Chat connection issues?\n\n1. Network problems:\n   • Check internet\n   • Switch networks\n   • Reset connection\n2. App fixes:\n   • Clear cache\n   • Force stop\n   • Update app\n3. If persisting:\n   • Reinstall\n   • Contact support",
                    
                    "Connection errors?\n\n1. Test internet\n2. Switch network\n3. Update app\n4. Clear data\n5. Try web version"
                ]
            },

    "Community_Features": {
                "creation": [
                    "Community creation issues?\n\n1. Account requirements:\n   • Verified profile\n   • Complete details\n   • Active status\n2. Creation process:\n   • Set community type\n   • Add description\n   • Set privacy\n3. If failing:\n   • Check guidelines\n   • Contact support",
                    
                    "Can't create community?\n\n1. Verify account status\n2. Check permissions\n3. Follow guidelines\n4. Clear cache\n5. Try web version"
                ],
                "posting": [
                    "Community posting problems?\n\n1. Post requirements:\n   • Member status\n   • Permission check\n   • Content rules\n2. Content issues:\n   • File sizes\n   • Format check\n   • Guidelines\n3. If stuck:\n   • Save draft\n   • Try text only",
                    
                    "Can't post content?\n\n1. Check membership\n2. Verify permissions\n3. Review rules\n4. Clear cache\n5. Try simpler post"
                ],
                "content": [
                    "Community content issues?\n\n1. Viewing problems:\n   • Check membership\n   • Privacy settings\n   • Access rights\n2. Feed issues:\n   • Refresh feed\n   • Clear cache\n   • Update app\n3. Nothing showing:\n   • Check filters\n   • Contact admin",
                    
                    "Feed not updating?\n\n1. Pull to refresh\n2. Clear app data\n3. Check membership\n4. Update app\n5. Report issue"
                ],
                "membership": [
                    "Join/Access problems?\n\n1. Membership check:\n   • Eligibility\n   • Requirements\n   • Restrictions\n2. Access denied:\n   • Contact admin\n   • Verify status\n   • Check rules\n3. Other issues:\n   • Clear cache\n   • Update app",
                    
                    "Can't join community?\n\n1. Review requirements\n2. Check restrictions\n3. Wait for approval\n4. Contact admin\n5. Try web version"
                ],
                "notifications": [
                    "Community alerts issues?\n\n1. Check settings:\n   • App notifications\n   • Community alerts\n   • System settings\n2. Fixes:\n   • Update preferences\n   • Clear cache\n   • Reset settings\n3. Still missing:\n   • Contact support",
                    
                    "Missing notifications?\n\n1. Enable alerts\n2. Check settings\n3. Update app\n4. Clear cache\n5. Contact admin"
                ],
                "feed": [
                    "Feed problems?\n\n1. Content loading:\n   • Check connection\n   • Refresh feed\n   • Clear cache\n2. Display issues:\n   • Update app\n   • Reset preferences\n   • Free memory\n3. No content:\n   • Verify access\n   • Contact admin",
                    
                    "Content not showing?\n\n1. Check membership\n2. Refresh feed\n3. Clear cache\n4. Update app\n5. Report issue"
                ],
                    "member_requests": [
        "Member requests not visible?\n\n1. Access checks:\n   • Admin permissions\n   • Notification settings\n   • Request section\n2. Common fixes:\n   • Refresh page\n   • Clear cache\n   • Update app\n3. Still missing:\n   • Check settings\n   • Contact support",
        
        "Can't see approvals?\n\n1. Check permissions\n2. Refresh requests\n3. Clear cache\n4. Update app\n5. Contact admin"
    ],
    "analytics": [
        "Analytics not updating?\n\n1. Data refresh:\n   • Manual sync\n   • Clear cache\n   • Check timeframe\n2. Display issues:\n   • Refresh page\n   • Update app\n   • Reset filters\n3. Still stuck:\n   • Export data\n   • Contact support",
        
        "Stats not showing?\n\n1. Verify permissions\n2. Refresh data\n3. Clear cache\n4. Check filters\n5. Contact admin"
    ],
                 "creation_issues": [
        "Community creation problems?\n\n1. Account checks:\n   • Verified profile\n   • Active status\n   • Permissions\n2. Creation steps:\n   • Set details\n   • Add description\n   • Configure rules\n3. If failing:\n   • Check eligibility\n   • Review guidelines\n   • Contact support",
        
        "Can't create community?\n\n1. Verify account\n2. Check permissions\n3. Set details\n4. Follow rules\n5. Try again"
    ],
    "posting_problems": [
        "Posting to community failed?\n\n1. Content checks:\n   • Follow guidelines\n   • Proper format\n   • Size limits\n2. Posting issues:\n   • Save draft\n   • Try again\n   • Check permissions\n3. Still failing:\n   • Clear cache\n   • Update app\n   • Contact admin",
        
        "Can't post content?\n\n1. Check rules\n2. Verify format\n3. Save draft\n4. Try again\n5. Contact admin"
    ],
    "access_denied": [
        "Access/membership issues?\n\n1. Join requirements:\n   • Complete profile\n   • Verify account\n   • Meet criteria\n2. Access problems:\n   • Check status\n   • Review rules\n   • Wait approval\n3. If denied:\n   • Contact admin\n   • Review guidelines\n   • Try again later",
        
        "Can't join community?\n\n1. Check requirements\n2. Complete profile\n3. Wait approval\n4. Contact admin\n5. Try again"
    ],
    "feed_issues": [
        "Community feed problems?\n\n1. Loading issues:\n   • Check connection\n   • Refresh feed\n   • Clear cache\n2. Content missing:\n   • Update app\n   • Check access\n   • Reset feed\n3. Not working:\n   • Force stop\n   • Reinstall\n   • Contact support",
        
        "Feed not loading?\n\n1. Check internet\n2. Refresh page\n3. Clear cache\n4. Update app\n5. Try again"
    ]
            },

    "App_Performance": {
                "crashes": [
                    "App crashing frequently?\n\n1. Immediate fixes:\n   • Force stop app\n   • Clear cache\n   • Restart device\n2. Memory issues:\n   • Close other apps\n   • Free up storage\n   • Check RAM\n3. Advanced fixes:\n   • Update app\n   • Clean install\n   • System update",
                    
                    "Stability problems?\n\n1. Clear app data\n2. Close background apps\n3. Update system\n4. Free up memory\n5. Reinstall app"
                ],
                            "image_performance": [
                "Image loading issues?\n\n1. Immediate fixes:\n   • Clear app cache\n   • Force stop app\n   • Restart application\n2. Connection checks:\n   • Verify internet connection\n   • Try different network\n   • Check signal strength\n3. Advanced solutions:\n   • Update app to latest version\n   • Free up device storage\n   • Check image permissions",
                
                "Images not showing?\n\n1. Basic troubleshooting:\n   • Refresh the page\n   • Clear app cache\n   • Check internet connection\n2. Advanced fixes:\n   • Update application\n   • Reset app data\n   • Reinstall if needed"
            ],
            "combined_performance": [
                "Image and performance issues?\n\n1. Quick fixes:\n   • Clear app cache and data\n   • Close background apps\n   • Free up device storage\n2. Connection solutions:\n   • Check internet speed\n   • Switch to Wi-Fi/mobile data\n   • Reset network settings\n3. Advanced troubleshooting:\n   • Update app version\n   • Check device memory\n   • Reinstall application",
                
                "App slow with image problems?\n\n1. Performance boost:\n   • Clear cache/data\n   • Close other apps\n   • Free up storage\n2. Image fixes:\n   • Check permissions\n   • Verify connection\n   • Update app\n3. Further help:\n   • Reset app settings\n   • Contact support"
            ],

                "speed": [
                    "Performance issues?\n\n1. Basic optimization:\n   • Clear cache\n   • Close apps\n   • Free storage\n2. App solutions:\n   • Update version\n   • Check memory\n   • Reset app\n3. System fixes:\n   • Update OS\n   • Check resources",
                    
                    "App running slow?\n\n1. Clear cached data\n2. Free up storage\n3. Close other apps\n4. Update app\n5. Try reinstalling"
                ],
                "response": [
                    "App not responding?\n\n1. Quick fixes:\n   • Force stop\n   • Clear memory\n   • Restart app\n2. Device solutions:\n   • Restart device\n   • Check storage\n   • Update system\n3. If frozen:\n   • Reinstall app\n   • Contact support",
                    
                    "Controls not working?\n\n1. Force stop app\n2. Clear all data\n3. Update version\n4. Check system\n5. Reinstall app"
                ],
                "technical": [
                    "Technical problems?\n\n1. App issues:\n   • Clear data\n   • Update app\n   • Check storage\n2. System checks:\n   • OS version\n   • Memory usage\n   • Background apps\n3. Advanced fixes:\n   • Clean install\n   • System reset",
                    
                    "App malfunctioning?\n\n1. Check requirements\n2. Clear all data\n3. Update system\n4. Free up space\n5. Reinstall app"
                ],
                "loading": [
                    "Content loading issues?\n\n1. Connection check:\n   • Internet speed\n   • Network type\n   • Signal strength\n2. App fixes:\n   • Clear cache\n   • Update app\n   • Force stop\n3. Storage issues:\n   • Free space\n   • Check memory",
                    
                    "Loading problems?\n\n1. Verify connection\n2. Clear app data\n3. Update app\n4. Free up storage\n5. Try reinstalling"
                ],
                    "slow_performance": [
        "App running slow?\n\n1. Quick fixes:\n   • Close other apps\n   • Clear cache\n   • Free storage\n2. App solutions:\n   • Force stop\n   • Update app\n   • Reset settings\n3. Device fixes:\n   • Restart device\n   • Update system\n   • Check memory",
        
        "Performance issues?\n\n1. Clear cache\n2. Close apps\n3. Free storage\n4. Update app\n5. Restart device"
    ],
    "app_crashes": [
        "App crashing frequently?\n\n1. Immediate actions:\n   • Force stop\n   • Clear cache\n   • Free memory\n2. App fixes:\n   • Update app\n   • Reinstall\n   • Reset settings\n3. Advanced solutions:\n   • System update\n   • Check storage\n   • Factory reset",
        
        "Crash problems?\n\n1. Force stop\n2. Clear data\n3. Update app\n4. Reinstall\n5. Contact support"
    ],
    "frozen_app": [
        "App frozen/unresponsive?\n\n1. Basic fixes:\n   • Wait briefly\n   • Force close\n   • Clear memory\n2. App solutions:\n   • Clear cache\n   • Update app\n   • Reinstall\n3. Device fixes:\n   • Restart device\n   • Free storage\n   • System update",
        
        "App not responding?\n\n1. Force close\n2. Clear cache\n3. Free memory\n4. Update app\n5. Restart device"
    ],
    "loading_problems": [
        "Loading screen stuck?\n\n1. Connection check:\n   • Internet speed\n   • Network type\n   • Switch network\n2. App fixes:\n   • Force stop\n   • Clear cache\n   • Update app\n3. Last resort:\n   • Reinstall app\n   • System update\n   • Contact support",
        
        "Loading issues?\n\n1. Check internet\n2. Clear cache\n3. Force stop\n4. Update app\n5. Try again"
    ]
            },

    "Location_Issues": {
                "access": [
                    "Location access problems?\n\n1. Permission setup:\n   • Enable GPS\n   • Allow app access\n   • System settings\n2. App solutions:\n   • Clear cache\n   • Update app\n   • Reset permissions\n3. If persisting:\n   • Check GPS signal\n   • Restart device",
                    
                    "Can't access location?\n\n1. Enable location services\n2. Grant permissions\n3. Check GPS\n4. Update app\n5. Restart device"
                ],
                "accuracy": [
                    "Location accuracy issues?\n\n1. Improve accuracy:\n   • High accuracy mode\n   • Clear location cache\n   • Update Google Maps\n2. Device fixes:\n   • Reset GPS\n   • Check signal\n   • Update system\n3. Manual options:\n   • Enter address\n   • Use landmarks",
                    
                    "Wrong location?\n\n1. Enable high accuracy\n2. Update GPS settings\n3. Clear cache\n4. Reset location\n5. Try manual entry"
                ],
                "search": [
                    "Location search problems?\n\n1. Search options:\n   • Check spelling\n   • Use landmarks\n   • Try nearby areas\n2. App fixes:\n   • Clear history\n   • Update app\n   • Reset location\n3. No results:\n   • Expand radius\n   • Different keywords",
                    
                    "Area search issues?\n\n1. Verify search terms\n2. Check coverage\n3. Use alternatives\n4. Clear history\n5. Contact support"
                ],
                "filter": [
                    "Location filter issues?\n\n1. Filter setup:\n   • Reset filters\n   • Check radius\n   • Update preferences\n2. Search tips:\n   • Use landmarks\n   • Try variations\n   • Expand area\n3. If not working:\n   • Clear cache\n   • Update app",
                    
                    "Filter not working?\n\n1. Clear all filters\n2. Set new radius\n3. Try different terms\n4. Update app\n5. Report issue"
                ]
            },
    "Notifications": {
                "property": [
                    "Property notification issues?\n\n1. General settings:\n   • Enable app notifications\n   • Check system settings\n   • Allow background refresh\n2. Property alerts:\n   • Price updates\n   • New listings\n   • Saved searches\n3. If missing:\n   • Clear cache\n   • Reset preferences",
                    
                    "Not getting property alerts?\n\n1. Check app settings\n2. Enable system alerts\n3. Verify preferences\n4. Update app\n5. Contact support"
                ],
                "settings": [
                    "Notification settings help:\n\n1. App settings:\n   • Open Settings\n   • Notifications section\n   • Choose categories\n2. System settings:\n   • Device notifications\n   • Background app refresh\n   • Priority settings\n3. Customize:\n   • Alert types\n   • Sound/vibration\n   • Frequency",
                    
                    "Configure notifications:\n\n1. Access settings\n2. Select categories\n3. Set preferences\n4. Test alerts\n5. Save changes"
                ],
                "technical": [
                    "Notification technical issues?\n\n1. Basic troubleshooting:\n   • Clear cache\n   • Force stop app\n   • Restart device\n2. Advanced fixes:\n   • Reset settings\n   • Update app\n   • Reinstall app\n3. Still issues:\n   • Check system\n   • Contact support",
                    
                    "Alerts not working?\n\n1. Verify permissions\n2. Clear app data\n3. Update app\n4. Reset settings\n5. Try reinstalling"
                ],
                "specific": [
                    "Category-specific alerts:\n\n1. Property alerts:\n   • Price changes\n   • New listings\n   • Saved searches\n2. Chat notifications:\n   • Messages\n   • Updates\n   • Replies\n3. Community alerts:\n   • Posts\n   • Activities\n   • Updates",
                    
                    "Custom notifications:\n\n1. Choose categories\n2. Set priorities\n3. Customize timing\n4. Select alert type\n5. Save preferences"
                ],
                "general": [
                    "General notification help:\n\n1. Check basics:\n   • Internet connection\n   • App permissions\n   • Battery settings\n2. System checks:\n   • Do Not Disturb\n   • Silent mode\n   • Background apps\n3. Optimization:\n   • Battery saver\n   • Data restrictions",
                    
                    "Alert system help:\n\n1. Enable all permissions\n2. Check restrictions\n3. Update settings\n4. Verify system\n5. Test notifications"
                ],
                "sync": [
                    "Notification sync issues?\n\n1. Connection check:\n   • Internet stability\n   • Background data\n   • Server status\n2. App solutions:\n   • Force sync\n   • Clear cache\n   • Update app\n3. If failing:\n   • Reset settings\n   • Reinstall app",
                    
                    "Sync problems?\n\n1. Check connection\n2. Enable background data\n3. Force sync\n4. Clear cache\n5. Update app"
                ]
            },

    "General_Help": {
                "support": [
                    "Need assistance?\n\n1. Support options:\n   • Help Center (help.reeltor.com)\n   • Live Chat Support\n   • Email: support@reeltor.com\n2. Quick help:\n   • FAQ section\n   • Video tutorials\n   • Community forum\n3. Direct contact:\n   • Call: 1-800-REELTOR\n   • Submit ticket",
                    
                    "How to get help?\n\n1. Visit help center\n2. Contact support\n3. Check FAQs\n4. Use live chat\n5. Call helpline"
                ],
                "information": [
                    "App information guide:\n\n1. Property terms:\n   • Common terminology\n   • Legal definitions\n   • Property types\n2. Guidelines:\n   • Usage policies\n   • Posting rules\n   • Safety tips\n3. Real estate info:\n   • Market basics\n   • Transaction guides",
                    
                    "Need information?\n\n1. Check help section\n2. Read guidelines\n3. View tutorials\n4. Browse FAQs\n5. Contact support"
                ],
                "terms": [
                    "Property terminology guide:\n\n1. Basic terms:\n   • Property types\n   • Measurements\n   • Legal terms\n2. Documentation:\n   • Required papers\n   • Verification process\n   • Legal checks\n3. Guidelines:\n   • Posting rules\n   • Photo policies",
                    
                    "Understanding terms?\n\n1. Check glossary\n2. Read guides\n3. View examples\n4. Ask support\n5. Visit help center"
                ],
                "guidelines": [
                    "Real estate guidelines:\n\n1. Property listing:\n   • Documentation needed\n   • Photo requirements\n   • Description rules\n2. Legal aspects:\n   • Verification process\n   • Required permits\n   • Compliance rules\n3. Best practices:\n   • Posting tips\n   • Safety measures",
                    
                    "Need guidelines?\n\n1. Read regulations\n2. Check requirements\n3. View examples\n4. Contact support\n5. Visit help center"
                ]
            },

            "Browse_Properties": {
                "navigation": [
                    "To browse properties:\n\n1. Use the search bar\n2. Apply relevant filters\n3. View property cards\n4. Save interesting listings\n5. Contact property owners",
                    "Property browsing help:\n\n1. Select property type\n2. Set price range\n3. Choose location\n4. View details\n5. Save favorites"
                ],
                "filters": [
                    "Using search filters:\n\n1. Select property type\n2. Set budget range\n3. Choose location\n4. Specify amenities\n5. Apply additional filters",
                    "Filter options help:\n\n1. Use price filters\n2. Select property features\n3. Choose locality\n4. Set property size\n5. Specify requirements"
                ],
                "comparison": [
                    "To compare properties:\n\n1. Save properties of interest\n2. View detailed specifications\n3. Check amenities list\n4. Compare prices\n5. Review locations",
                    "Property comparison help:\n\n1. Select properties to compare\n2. Review features\n3. Check differences\n4. Compare prices\n5. Evaluate locations"
                ],
                "saved": [
                    "Managing saved properties:\n\n1. View saved listings\n2. Organize favorites\n3. Set alerts\n4. Remove outdated saves\n5. Share saved properties",
                    "Saved property features:\n\n1. Access favorites section\n2. Update saved searches\n3. Get property updates\n4. Manage alerts\n5. Contact sellers"
                ]
            },

            "Profile_Management": {
                "settings": [
                    "Profile settings help:\n\n1. Update personal info\n2. Manage contact details\n3. Set preferences\n4. Update password\n5. Configure privacy",
                    "Account management:\n\n1. Edit profile details\n2. Update contact info\n3. Change settings\n4. Manage notifications\n5. Set privacy options"
                ],
                "preferences": [
                    "Setting preferences:\n\n1. Choose property types\n2. Set location preference\n3. Configure alerts\n4. Update search settings\n5. Save changes",
                    "Preference management:\n\n1. Update search filters\n2. Set notification preferences\n3. Choose alert types\n4. Configure display options\n5. Save settings"
                ],
                "security": [
                    "Account security:\n\n1. Update password\n2. Enable two-factor auth\n3. Manage devices\n4. Review activity\n5. Set security questions",
                    "Security settings:\n\n1. Change password\n2. Verify contact details\n3. Check login activity\n4. Manage permissions\n5. Update security"
                ],
                "verification": [
                    "Account verification:\n\n1. Verify email address\n2. Confirm phone number\n3. Upload required documents\n4. Complete verification\n5. Wait for approval",
                    "Verification process:\n\n1. Submit documents\n2. Verify contacts\n3. Complete profile\n4. Check status\n5. Contact support"
                ]
            }
        }


    chatbot = ReeltorChatbot(config, NEW_INTENTS, CONTEXTUAL_RESPONSES)

    print("""
    🏠 Welcome to Reeltor Support Assistant! 
    
    I'm here to help you with:
    • Account issues
    • App functionality
    • Content loading
    • General questions
    
    Type 'exit' to end our conversation.
    """)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("""
                🙏 Thank you for using Reeltor Support!
                
                We hope we could help you today.
                Remember, we're available 24/7 for any real estate needs.
                
                ╔════════════════════════════╗
                ║      Have a great day!     ║
                ╚════════════════════════════╝
                """)
                break

            response = chatbot.process_input(user_input)
            print(f"\n🤖 Assistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye! Thanks for using Reeltor Support.")
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred: {str(e)}")
            print("Please try again or contact Reeltor support.")

if __name__ == "__main__":
    main()
'''