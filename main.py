'''
from config import ChatbotConfig
from response_generator import ResponseGenerator
from intent_classifier import IntentClassifier
from semantic_matcher import SemanticMatcher
from utils import setup_logging
from typing import Dict, List
import logging
from utils import correct_text
class ReeltorChatbot:
    def __init__(
        self,
        config: ChatbotConfig,
        intents: Dict[str, List[str]],
        responses: Dict[str, List[str]]
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

    def process_input(self, user_input: str) -> str:
        try:
            user_input = correct_text(user_input)

            # First try semantic matching
            intent, example, score = self.semantic_matcher.find_best_match(user_input)
            self.logger.info(f"Semantic matching score: {score} for intent: {intent}")
            
            if score >= self.config.fuzzy_threshold:
                self.logger.info(
                    f"Using high confidence semantic match: {intent} ({score:.2f})"
                )
                return self.response_generator.generate_response(user_input, intent)
            
            # If semantic matching isn't confident enough, try BERT classifier
            bert_intent, bert_confidence = self.intent_classifier.predict(user_input)
            self.logger.info(
                f"BERT confidence: {bert_confidence} for intent: {bert_intent}"
            )

            # Use BERT's prediction if it's confident
            if bert_confidence > self.config.bert_confidence_threshold:
                self.logger.info(
                    f"Using BERT prediction: {bert_intent} ({bert_confidence:.2f})"
                )
                return self.response_generator.generate_response(
                    user_input,
                    bert_intent
                )
            
            # If both methods have low confidence but semantic matching is better
            if score > bert_confidence and score > self.config.fuzzy_fallback_threshold:
                self.logger.info(
                    f"Using moderate confidence semantic match: {intent} ({score:.2f})"
                )
                return self.response_generator.generate_response(user_input, intent)
            
            # Only fallback if all confidence scores are very low
            self.logger.info("All confidence scores too low, using fallback")
            return self.response_generator.generate_response(user_input)

        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return (
                "I apologize, but I encountered an error processing your request. "
                "Please try again or contact Reeltor support if the issue persists."
            )

def main():
    config = ChatbotConfig()
    
    # Define intents and responses
    intents = {

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
        "howdy"
    ],

        "Handle Account Problem": [
            "I can't log into my account",
            "How do I reset my password?",
            "My account is locked after too many attempts",
            "I'm not receiving the OTP on my phone",
            "How do I change my registered mobile number?",
            "I forgot my username",
            "My email verification link isn't working",
            "How do I update my profile information?",
            "I'm unable to delete my account",
            "Can't change my password even after resetting",
            "Getting 'invalid credentials' error despite correct password",
            "How do I link my email to my account?",
            "My phone number is already registered with another account",
            "Why am I logged out automatically?",
            "Can't login with Google account",
            "Apple ID login not working",
            "How do I unlink my Google account?",
            "Login with Apple shows error",
            "Google sign-in keeps loading",

        ],
        "Handle App Functionality Issue": [
            "The app keeps crashing when I open it",
            "Search filter is not working properly",
            "Can't save properties to favorites",
            "Unable to view property prices",
            "The filter options are not responding",
            "Can't upload property media file",
            "The app is not responding",
            "Notification settings aren't saving",
            "Why is the app so slow?",
            "The share button isn't working",
            "Can't switch between buy and rent options",
            "The app is crashing",
            "App freezes when applying multiple filters",
            "I can't upload my property listing",
            "Why is my property post stuck in processing?",
            "How do I edit my property listing?",
            "My property images won't upload",
            "Can't add property details to my listing",
            "Error when trying to post my property",
            "Why did my property post get rejected?",
            "How do I remove my property listing?",
            "Can't update property price after posting",
            "Property location not showing correctly in my post",
            "Unable to add property amenities",
            "My property post isn't visible to others",
            "Can't add multiple photos to my listing",
            "Can't send messages to property owners",
            "Chat messages not delivering",
            "Not receiving chat notifications",
            "Unable to view chat history",
            "Messages disappearing from chat",
            "Can't initiate chat with seller",
            "Chat feature freezing",
            "Unable to share media in chat",
            "Chat notifications not working",
            "Can't block spam messages",
            "Previous chats not loading",
            "Chat screen blank or unresponsive",
            "Unable to delete chat messages",
            "Chat sound notifications not working",
            "Unable to create new community",
            "Can't post in community section",
            "Community posts not showing",
            "Unable to join communities",
            "Can't see community members",
            "Community notifications not working",
            "Unable to leave a community",
            "Can't share posts in community",
            "Community search not working",
            "Unable to moderate community",
            "Can't edit community settings",
            "Community posts not updating",
            "Unable to report community content",
            "Can't invite members to community",
            "Community media not loading",
            "Community posts not loading",
            "Community search not showing results",
            "Community posts not refreshing",
            "Community notifications not showing",
            "Community members list not updating",
            
            
        ],
        "Handle Content Loading Problem": [
            "Property images are not loading",
            "Can't see any properties in my area",
            "Reel videos aren't playing",
            "Property details page is blank",
            "The app is taking too long to load",
            "Property videos are stuck on buffering",
            "Property descriptions are missing",
            "Videos won't play",
            "Reels not working",
            "Blank property page",
            "Empty property details",
            "Content not showing up",
            "Media not loading",
            "Property video not playing in reels",
            "Video quality is too low",
            "Videos stop playing halfway",
            "Can't load video previews",
            "Video audio not working",
            "Reel videos keep freezing",
            "Video playback is choppy",
            "Video title not loading",
            "Video keeps pausing automatically",
            "Video playback error messages",
            "Property images are blurry",
            "Can't zoom into property photos",
            "Images load partially then freeze",
            "High-resolution photos not loading",
            "Can't swipe through photos",
            "Amenities list not loading",
            "Property specifications missing",
            "Owner details not loading",
            "Similar properties not showing",
            "Property status not updating",
            "Features list incomplete",
            "Location details missing",
            "video view information not visible",
            "Neighborhood info not loading",
            "looking for pg but doesn't show",
            "looking for flat but doesn't show",
            "looking for villa but doesn't show",
            "looking for plot but doesn't show",
            "looking for commercial but doesn't show",
            "looking for rental but doesn't show",
            "when I open app for the first time it shows blank reels and images",
            "I am not able see pg in reels videos",
            "I am not able see flat in reels videos",
            "I am not able see villa in reels videos",
            "I am not able see plot in reels videos",
            "I am not able see commercial in reels videos",
            "I am not able see rental in reels videos",
        ],
        "Provide General Help": [
            "How do I use this app?",
            "Where can I find saved properties?",
            "How do I contact property owners?",
            "What are the features of this app?",
            "How do I schedule a site visit?",
            "How to filter properties by budget?",
            "How do I share property details?",
            "What's the difference between carpet area and built-up area?",
            "How can I get notifications for properties?"
        ]
    }

    responses = {

            "Greeting": [
        """Welcome to Reeltor! 👋

I'm your friendly real estate assistant, here to help make your property journey smoother! 🏠

How can I assist you today? I can help with:
• Finding your dream property 🔍
• Troubleshooting app issues ⚙️
• Account support 👤
• Property posting guidance 📝
• Community features 👥"""
    ],




        "Handle Account Problem": [
            "It seems like you're having trouble logging in. You can reset your password by clicking the 'Forgot Password'.",
            "If your account is locked, please contact support to unlock it.",
            "If you're not receiving the OTP on your phone, try checking your spam folder or request a new OTP.",
            "To update your registered mobile number, go to your profile settings and follow the instructions.",
            "If you've forgotten your username, you can retrieve it via email or contact support.",
            "If the email verification link isn't working, try requesting a new one or check your spam folder.",
            "For issues with updating profile information, ensure you're logged in properly and try again.",
            "If you're unable to delete your account, please contact support for assistance.",
            "Try resetting your password again if you're still unable to change it.",
            "If you're still getting an 'invalid credentials' error despite entering the correct password, try resetting your password.",
            "To link your email to your account, go to your profile settings and follow the instructions.",
            "If your phone number is already registered with another account, please contact support to resolve the issue.",
            "If you're being logged out automatically, try clearing your browser cache or app data.",
            "For Google login issues, try clearing your browser cache and ensuring you're using the correct Google account.",
            "Apple ID login problems often resolve by signing out of Apple ID on your device and signing back in.",
            "Apple login errors often resolve by updating iOS or checking Apple ID settings.",
            "Endless Google sign-in loading usually fixes by clearing Google app cache."

        ],
        "Handle App Functionality Issue": [
            "The app crashing issue might be resolved by clearing the cache and restarting the app.",
            "If the search filter isn't working, try updating the app to the latest version.",
            "For issues with saving properties to favorites, ensure you're logged in properly.",
            "If you're unable to view property prices, try refreshing the page or restarting the app.",
            "If the filter options are not responding, try closing and reopening the app.",
            "For issues with uploading property media files, ensure you have a stable internet connection.",
            "If the app is not responding, try force-closing the app and reopening it.",
            "If notification settings aren't saving, ensure you have granted the necessary permissions.",
            "If the app is slow, try closing other apps running in the background or restarting your device.",
            "If the share button isn't working, try using a different browser or update the application.",
            "If you can't switch between buy and rent options, try restarting the app or reinstalling it.",
            "The app crashing issue might be resolved by clearing the cache and restarting the app.",
            "If the app freezes when applying multiple filters, try reducing the number of filters or restarting the app.",
            "For property upload issues, ensure all required fields are filled and images meet size requirements",
            "If your post is stuck processing, try clearing the app cache and resubmitting. Check your internet connection.",
            "Edit your property listing through Profile > My Listings > Edit. Make sure to save changes.",
            "For image upload issues, try reducing image size or uploading one at a time. Check file format (JPG/PNG only).",
            "Unable to add details? Try again filling detail, then complete the listing section by section.",
            "Posting errors often resolve by checking all mandatory fields and having stable internet.",
            "Property rejections usually occur if guidelines aren't met. Check the image size,location and internet speed",
            "Remove listings not available as there is no option to hide them. Contact support for further assistance.",
            "Price updates not available in the app. You cannot edit the price in the listing details.",
            "Verify location by choosing option manually if auto-location isn't accurate.",
            "Try updating amenities one at a time to prevent submission errors.",
            "Visibility issues? Check if your listing is posted and active. Contact support if it's still not visible.",
            "For multiple photos, upload sequentially and wait for each to complete.",
            "For message sending issues, check your internet connection and try restarting the chat.",
            "If messages aren't delivering, verify both users have active internet connections.",
            "Enable chat notifications in Profile > Settings > Notifications > Chat Alerts.",
            "Chat history issues often resolve after clearing cache or updating the app.",
            "Message disappearance might be due to cache clearing. Try refreshing the chat.",
            "To start a chat, use the 'Message' button on property listings or user profiles.",
            "For freezing issues, try force-stopping the app and restarting.",
            "Check both app and device notification settings for chat alerts.",
            "Try logging out and back in to restore previous chats.",
            "Blank chat screens usually resolve after app restart or cache clearing.",
            "Delete messages by long-pressing the message and selecting 'Delete'.",
            "Enable sound notifications in Profile > Settings > Sound & Notifications.",
            "Create communities via Communities tab > '+' button. Fill required details.",
            "Posting issues? Verify you're a member and have posting again after restarting application.",
            "If posts aren't visible, try refreshing or check your internet connection.",
            "Join communities through the 'Join' button on community pages.",
            "Member list visible in Community > Members (if permitted by community settings).",
            "Enable community notifications in Profile > Settings > Community Alerts.",
            "Leave communities through Community Settings > Leave Community.",
            "Share posts using the share icon on individual posts.",
            "For search issues, try using different keywords or check spelling.",
            "Moderate through Community Settings > Moderation Tools (admin only).",
            "Edit community settings via Community > Settings (admin only).",
            "Post updates may be delayed. Try pulling down to refresh.",
            "Report inappropriate content using the '...' menu on posts.",
            "Invite members through Community > Invite Members > Share Link.",
            "Media loading issues often resolve with cache clearing or app restart.",
            "Posts not loading usually resolve after app update or cache clearing.",
            "community Search issues often resolve after app update or cache clearing.",
            "community Posts not refreshing usually resolve after app update or cache clearing.",
            "Community notifications not showing usually resolve after app update or cache clearing.",
            "Community members list not updating usually resolve after app update or cache clearing.",
            "If not able to post community, try updating the app or cache clearing.",

        ],
        "Handle Content Loading Problem": [
            "If property images aren't loading, check your internet connection or try refreshing the page.",
            "If you can't see any properties in your area, try expanding your search radius or adjusting the filters.",
            "If reel videos aren't playing, ensure your device supports video playback or has a stable internet connection.",
            "If the property details page is blank, try refreshing the page or restarting the app.",
            "If the app is taking too long to load, try checking your internet connection or restarting the app.",
            "If property videos are stuck on buffering, ensure you have a stable internet connection.",
            "If property descriptions are missing, try refreshing the page or report the issue to support.",
            "For video playback issues, check your internet speed and try connecting to high speed internet.",
            "Low video quality might be due to auto-quality settings. Try manually connecting to Wifi/internet to get higher quality in video.",
            "Videos stopping midway often improve by closing other apps and ensuring stable internet connection.",
            "Clear app cache and ensure you have sufficient storage space for video previews to load properly.",
            "Check device volume and app permissions for audio issues. Restart app if needed.",
            "Freezing reels usually improve by updating the app or clearing temporary files.",
            "Choppy playback often resolves by closing background apps or switching to WiFi.",
            "Refresh the page or clear cache to fix title loading issues.",
            "Auto-pausing might be due to battery optimization. Check power settings.",
            "Error messages usually clear up after app restart or cache clearing.",
            "Blurry images might be due to slow internet. Wait for full resolution to load or check connection.",
            "Partial loading issues usually resolve with stronger internet connection.",
            "High-res photos need good internet connection. Try switching to WiFi.",
            "Swiping issues might need app permissions update or restart."
            "Refresh page to load missing amenities or check internet connection.",
            "Specifications might need page reload or app cache clearing.",
            "Similar properties need location services and search permissions.",
            "Status updates appear after refresh or cache clearing.",
            "Incomplete features list might need app update or refresh.",
            "Location details require GPS permissions and updated data.",
            "Video view information needs stable internet connection and updated app version.",
            "Neighborhood info requires location services and data update.",
            "PG listings may not show if not available in your area.",
            "Flat listings may not show if not available in your area.",
            "Villa listings may not show if not available in your area.",
            "Plot listings may not show if not available in your area.",
            "Commercial listings may not show if not available in your area.",
            "Rental listings may not show if not available in your area."
            "Blank reels and images on first app open usually resolve after app update or cache clearing.",
            "PG listings may not show in reels if not available in your area.",
            "Flat listings may not show in reels if not available in your area.",
            "Villa listings may not show in reels if not available in your area.",
            "Plot listings may not show in reels if not available in your area.",
            "Commercial listings may not show in reels if not available in your area.",
            "Rental listings may not show in reels if not available in your area."


             
        ],
        "Provide General Help": [
            "To use the app, start by browsing properties using the search bar and applying filters. You can also use the community section to join discussions and see property information.",
            "You can find saved properties in the 'Favorites' section of your profile.",
            "To contact property owners, click the 'Contact Seller' button on the property details page.",
            "The app features include property search, filtering by budget, saving properties, and contacting owners via the chat section.",
            "You can directly contact the owner of the property to schedule a visit.",
            "To filter properties by budget, use the price range slider in the search filters.",
            "To share property details, click the 'Share' button and choose your preferred method (e.g., email, social media).",
            "Carpet area refers to the usable floor space inside the walls, while built-up area includes the carpet area plus the thickness of walls and common areas.",
            "To get notifications for properties, enable push notifications in the app settings and set up your preferences."
        ]
    }

    chatbot = ReeltorChatbot(config, intents, responses)

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