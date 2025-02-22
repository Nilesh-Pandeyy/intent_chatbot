import requests
import time
import json
from typing import List, Dict
import random
from datetime import datetime

class ChatbotAPITester:
    def __init__(self, api_url: str = "http://localhost:5000/api/chat"):
        self.api_url = api_url
        self.test_results = []
        self.headers = {
            "Content-Type": "application/json",
            "Origin": "http://localhost:5000",
            "Accept": "application/json",
            "Access-Control-Allow-Origin": "*"
        }
        self.session = requests.Session()

    def send_message(self, message: str) -> Dict:
        """Send a message to the chatbot API and return the response"""
        try:
            response = self.session.post(
                self.api_url,
                json={"message": message},
                headers=self.headers,
                verify=False
            )
            
            if response.status_code == 403:
                print("Received 403 Forbidden - Checking server status...")
                return {"error": "403 Forbidden - Check server CORS settings"}
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            error_msg = "Connection Error - Make sure the Flask server is running"
            print(f"\nError: {error_msg}")
            return {"error": error_msg}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"\nError: {error_msg}")
            return {"error": error_msg}

    def run_test_cases(self, test_cases: List[Dict[str, str]]):
        """Run test cases and collect results"""
        print("\n=== Starting Chatbot API Tests ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total test cases: {len(test_cases)}")
        
        # Check server status
        try:
            print("\nChecking server status...")
            server_check = requests.get("http://localhost:5000/")
            if server_check.ok:
                print("Server is running and accessible!")
            else:
                print("Warning: Server response indicates issues")
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to server. Make sure Flask server is running.")
            return

        # Run tests
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}")
            print(f"Category: {test.get('category', 'Not specified')}")
            print(f"Query: {test['query']}")
            
            # Add random delay between requests
            time.sleep(random.uniform(0.5, 1.5))
            
            # Send request and measure response time
            start_time = time.time()
            response = self.send_message(test['query'])
            end_time = time.time()
            
            # Store test results
            result = {
                'test_number': i,
                'category': test.get('category', 'Not specified'),
                'query': test['query'],
                'expected_intent': test.get('expected_intent', 'Not specified'),
                'response': response.get('response', str(response)),
                'response_time': round(end_time - start_time, 3),
                'success': 'error' not in response
            }
            self.test_results.append(result)
            
            # Print response
            if 'response' in response:
                print(f"Response: {response['response'][:150]}..." if len(response['response']) > 150 else f"Response: {response['response']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
            print(f"Response time: {result['response_time']} seconds")

    def generate_report(self):
        """Generate a detailed test report"""
        if not self.test_results:
            print("\nNo test results to report - Tests may have failed to run")
            return
            
        print("\n=== Chatbot API Test Report ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        total_time = sum(r['response_time'] for r in self.test_results)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        # Calculate category statistics
        categories = {}
        for result in self.test_results:
            category = result['category']
            if category not in categories:
                categories[category] = {'total': 0, 'success': 0}
            categories[category]['total'] += 1
            if result['success']:
                categories[category]['success'] += 1
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total tests run: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Failed tests: {total_tests - successful_tests}")
        print(f"Success rate: {(successful_tests/total_tests*100):.1f}%")
        print(f"Average response time: {round(avg_time, 3)} seconds")
        
        # Print category statistics
        print("\nCategory Statistics:")
        for category, stats in categories.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{category}:")
            print(f"  Total tests: {stats['total']}")
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Print detailed results
        print("\nDetailed Results:")
        for result in self.test_results:
            print(f"\nTest {result['test_number']}:")
            print(f"Category: {result['category']}")
            print(f"Query: {result['query']}")
            print(f"Expected Intent: {result['expected_intent']}")
            print(f"Success: {'Yes' if result['success'] else 'No'}")
            print(f"Response time: {result['response_time']} seconds")
            if len(str(result['response'])) > 150:
                print(f"Response: {str(result['response'])[:150]}...")
            else:
                print(f"Response: {result['response']}")

def main():
    # Comprehensive test cases
    test_cases = [
        # Greetings
        {
            "category": "Greetings",
            "query": "Hello",
            "expected_intent": "Greeting"
        },
        {
            "category": "Greetings",
            "query": "Hi Reeltor",
            "expected_intent": "Greeting"
        },
        
        # Account Login Tests
        {
            "category": "Account Login",
            "query": "Unable to login to my account",
            "expected_intent": "Account_Login"
        },
        {
            "category": "Account Login",
            "query": "Gmail login not working",
            "expected_intent": "Account_Login"
        },
        {
            "category": "Account Login",
            "query": "Apple ID verification failed",
            "expected_intent": "Account_Login"
        },
        {
            "category": "Account Login",
            "query": "Not getting OTP for verification",
            "expected_intent": "Account_Login"
        },
        {
            "category": "Account Login",
            "query": "How to change my mobile number",
            "expected_intent": "Account_Login"
        },
        
        # Property Search Tests
        {
            "category": "Property Search",
            "query": "Search not working properly",
            "expected_intent": "Property_Search"
        },
        {
            "category": "Property Search",
            "query": "Can't find commercial properties",
            "expected_intent": "Property_Search"
        },
        {
            "category": "Property Search",
            "query": "No recommendations showing",
            "expected_intent": "Property_Search"
        },
        {
            "category": "Property Search",
            "query": "Filter not working",
            "expected_intent": "Property_Search"
        },
        {
            "category": "Property Search",
            "query": "Saved properties missing",
            "expected_intent": "Property_Search"
        },
        
        # Property Upload Tests
        {
            "category": "Property Upload",
            "query": "Unable to upload property photos",
            "expected_intent": "Property_Upload"
        },
        {
            "category": "Property Upload",
            "query": "Image quality reduced after uploading",
            "expected_intent": "Property_Upload"
        },
        {
            "category": "Property Upload",
            "query": "Property listing not visible",
            "expected_intent": "Property_Upload"
        },
        {
            "category": "Property Upload",
            "query": "How to upload property",
            "expected_intent": "Property_Upload"
        },
        
        # Reels Features Tests
        {
            "category": "Reels Features",
            "query": "Videos not playing in reels",
            "expected_intent": "Reels_Features"
        },
        {
            "category": "Reels Features",
            "query": "Reels stuck on loading",
            "expected_intent": "Reels_Features"
        },
        {
            "category": "Reels Features",
            "query": "No sound in property videos",
            "expected_intent": "Reels_Features"
        },
        {
            "category": "Reels Features",
            "query": "Can't swipe reels",
            "expected_intent": "Reels_Features"
        },
        
        # Chat Features Tests
        {
            "category": "Chat Features",
            "query": "Unable to send messages to owner",
            "expected_intent": "Chat_Features"
        },
        {
            "category": "Chat Features",
            "query": "Chat history disappeared",
            "expected_intent": "Chat_Features"
        },
        {
            "category": "Chat Features",
            "query": "Can't send images in chat",
            "expected_intent": "Chat_Features"
        },
        {
            "category": "Chat Features",
            "query": "Not getting chat notifications",
            "expected_intent": "Chat_Features"
        },
        
        # Location Issues Tests
        {
            "category": "Location Issues",
            "query": "Location access not working",
            "expected_intent": "Location_Issues"
        },
        {
            "category": "Location Issues",
            "query": "Wrong area showing in map",
            "expected_intent": "Location_Issues"
        },
        {
            "category": "Location Issues",
            "query": "Location accuracy very poor",
            "expected_intent": "Location_Issues"
        },
        {
            "category": "Location Issues",
            "query": "Can't search by location",
            "expected_intent": "Location_Issues"
        },
        
        # App Performance Tests
        {
            "category": "App Performance",
            "query": "App very slow and lagging",
            "expected_intent": "App_Performance"
        },
        {
            "category": "App Performance",
            "query": "App keeps crashing repeatedly",
            "expected_intent": "App_Performance"
        },
        {
            "category": "App Performance",
            "query": "Screen frozen and not responding",
            "expected_intent": "App_Performance"
        },
        {
            "category": "App Performance",
            "query": "Content not loading",
            "expected_intent": "App_Performance"
        },
        
        # Community Features Tests
        {
            "category": "Community Features",
            "query": "Unable to create new community",
            "expected_intent": "Community_Features"
        },
        {
            "category": "Community Features",
            "query": "Can't post in community section",
            "expected_intent": "Community_Features"
        },
        {
            "category": "Community Features",
            "query": "Community posts not loading",
            "expected_intent": "Community_Features"
        },
        {
            "category": "Community Features",
            "query": "Can't join local community",
            "expected_intent": "Community_Features"
        },
        
        # Notification Tests
        {
            "category": "Notifications",
            "query": "Not getting property alerts",
            "expected_intent": "Notifications"
        },
        {
            "category": "Notifications",
            "query": "Notification settings not saving",
            "expected_intent": "Notifications"
        },
        {
            "category": "Notifications",
            "query": "Missing chat notifications",
            "expected_intent": "Notifications"
        },
        {
            "category": "Notifications",
            "query": "How to manage notifications",
            "expected_intent": "Notifications"
        },
        
        # Complex Queries
        {
            "category": "Complex Queries",
            "query": "App crashes when uploading property photos",
            "expected_intent": "Property_Upload"
        },
        {
            "category": "Complex Queries",
            "query": "Location not accurate in property listing",
            "expected_intent": "Property_Upload"
        },
        {
            "category": "Complex Queries",
            "query": "Reels freeze when showing property details",
            "expected_intent": "Reels_Features"
        }
    ]
    
    # Initialize tester
    tester = ChatbotAPITester()
    
    # Run tests
    tester.run_test_cases(test_cases)
    
    # Generate report
    tester.generate_report()

if __name__ == "__main__":
    main()