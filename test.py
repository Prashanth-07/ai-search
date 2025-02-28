import requests
import json
from typing import Dict, Optional
import time
from datetime import datetime
import sys

class TestLogger:
    def __init__(self):
        self.test_results = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        print(formatted_message)
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        result = {"test": test_name, "passed": passed, "details": details}
        self.test_results.append(result)
        status = "PASSED" if passed else "FAILED"
        self.log(f"{test_name}: {status} - {details}", status)
        
    def summary(self):
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        self.log(f"\nTest Summary:", "INFO")
        self.log(f"Total Tests: {total}", "INFO")
        self.log(f"Passed: {passed}", "INFO")
        self.log(f"Failed: {total - passed}", "INFO")
        return passed == total

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.logger = TestLogger()
        self.stored_tool_id = None
        
    def check_server(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(self.base_url)
            self.logger.log_test(
                "Server Check",
                response.ok,
                "Server is running" if response.ok else "Server is not accessible"
            )
            return response.ok
        except requests.exceptions.ConnectionError:
            self.logger.log_test(
                "Server Check",
                False,
                f"Cannot connect to server at {self.base_url}"
            )
            return False

    def test_health(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            success = response.ok and response.json().get("status") == "healthy"
            self.logger.log_test(
                "Health Check",
                success,
                f"Response: {response.json() if response.ok else response.text}"
            )
            return success
        except Exception as e:
            self.logger.log_test("Health Check", False, f"Error: {str(e)}")
            return False

    def test_add_tool(self) -> bool:
        """Test adding a new tool"""
        test_tool = {
            "name": "Test AI Assistant",
            "description": "A powerful AI assistant for testing purposes",
            "pros": [
                "Fast response time",
                "High accuracy"
            ],
            "cons": [
                "Limited to test environment",
                "Requires setup"
            ],
            "categories": "Testing, AI, Automation",
            "usage": "Automated testing and quality assurance",
            "unique_features": "Real-time test feedback, Integration testing capabilities",
            "pricing": "Free for testing, $10/month for production"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/add-tool",
                json=test_tool
            )
            success = response.ok
            if success:
                self.stored_tool_id = response.json().get("id")
                
            self.logger.log_test(
                "Add Tool",
                success,
                f"Tool ID: {self.stored_tool_id if success else None}"
            )
            return success
        except Exception as e:
            self.logger.log_test("Add Tool", False, f"Error: {str(e)}")
            return False

    def test_query_tools(self) -> bool:
        """Test querying tools"""
        test_queries = [
            "I need an AI tool for testing",
            "What are some affordable AI assistants?",
            "Show me tools with good accuracy"
        ]
        
        all_successful = True
        for query in test_queries:
            try:
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query}
                )
                success = response.ok
                all_successful &= success
                self.logger.log_test(
                    f"Query Tools - '{query}'",
                    success,
                    f"Response received: {len(response.json().get('response', '')) > 0 if success else False}"
                )
            except Exception as e:
                self.logger.log_test(f"Query Tools - '{query}'", False, f"Error: {str(e)}")
                all_successful = False
        
        return all_successful

    def test_delete_tool(self) -> bool:
        """Test deleting a tool"""
        if not self.stored_tool_id:
            self.logger.log_test("Delete Tool", False, "No tool ID available for deletion")
            return False
            
        try:
            response = requests.delete(f"{self.base_url}/delete-tool/{self.stored_tool_id}")
            success = response.ok
            self.logger.log_test(
                "Delete Tool",
                success,
                f"Response: {response.json() if success else response.text}"
            )
            return success
        except Exception as e:
            self.logger.log_test("Delete Tool", False, f"Error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests in sequence"""
        self.logger.log("Starting API Tests...", "INFO")
        
        # Test 1: Check if server is running
        if not self.check_server():
            self.logger.log("Server not accessible. Stopping tests.", "ERROR")
            return False
            
        # Test 2: Health Check
        if not self.test_health():
            self.logger.log("Health check failed. Stopping tests.", "ERROR")
            return False
            
        # Test 3: Add Tool
        if not self.test_add_tool():
            self.logger.log("Adding tool failed. Stopping tests.", "ERROR")
            return False
            
        # Give the vector store a moment to index
        time.sleep(2)
        
        # Test 4: Query Tools
        if not self.test_query_tools():
            self.logger.log("Querying tools failed.", "ERROR")
            
        # Test 5: Delete Tool
        if not self.test_delete_tool():
            self.logger.log("Deleting tool failed.", "ERROR")
            
        # Print test summary
        return self.logger.summary()

def main():
    try:
        tester = APITester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# import requests
# import json
# import time

# # API base URL
# BASE_URL = "http://localhost:8000"

# # Tool data
# tool_data = [
#     {
#         "name": "VintageImageGen",
#         "description": "Generates vintage-style images.",
#         "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
#     },
#     {
#         "name": "ModernImageGen",
#         "description": "Generates modern images with high resolution",
#         "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"
#     },
#     {
#         "name": "AIChat",
#         "description": "An AI-powered chatbot for conversational purposes.",
#         "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"
#     },
#     {
#         "name": "ChatGPT",
#         "description": "A conversational AI developed by OpenAI for natural language understanding.",
#         "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"
#     },
#     {
#         "name": "DALL-E",
#         "description": "Generates creative images from textual descriptions using AI.",
#         "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"
#     },
#     {
#         "name": "Midjourney",
#         "description": "An AI tool that creates artistic images based on text prompts.",
#         "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"
#     },
#     {
#         "name": "StableDiffusion",
#         "description": "A latent diffusion model for generating detailed images from text.",
#         "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"
#     },
#     {
#         "name": "Copilot",
#         "description": "An AI pair programmer that assists with code completion and generation.",
#         "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"
#     },
#     {
#         "name": "DeepLTranslate",
#         "description": "An AI-powered translation service for multiple languages.",
#         "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"
#     },
#     {
#         "name": "VoiceClone",
#         "description": "Clones and synthesizes human voices using advanced AI techniques.",
#         "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"
#     },
#     {
#         "name": "SentimentAnalyzer",
#         "description": "Analyzes text to determine the sentiment using AI.",
#         "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"
#     },
#     {
#         "name": "RecommenderAI",
#         "description": "Provides personalized recommendations based on user data and AI analysis.",
#         "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"
#     },
#     {
#         "name": "FraudDetector",
#         "description": "Detects fraudulent activities using sophisticated AI algorithms.",
#         "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"
#     },
#     {
#         "name": "AnomalyFinder",
#         "description": "Identifies anomalies in datasets using high-sensitivity AI models.",
#         "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"
#     },
#     {
#         "name": "VirtualAssistant",
#         "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.",
#         "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"
#     }
# ]

# def test_initialize():
#     """Test the initialize endpoint"""
#     print("\nTesting Initialize Endpoint...")
#     url = f"{BASE_URL}/initialize"
#     response = requests.post(url, json={"tools": tool_data})
#     print(f"Status Code: {response.status_code}")
#     print(f"Response: {response.json()}")
#     return response.status_code == 200

# def test_add_tools():
#     """Test the add-tools endpoint"""
#     print("\nTesting Add Tools Endpoint...")
#     url = f"{BASE_URL}/add-tools"
#     # Adding just two tools for testing
#     new_tools = tool_data[:2]
#     response = requests.post(url, json={"tools": new_tools})
#     print(f"Status Code: {response.status_code}")
#     print(f"Response: {response.json()}")
#     return response.status_code == 200

# def test_query(query_text):
#     """Test the query endpoint"""
#     print(f"\nTesting Query Endpoint with: '{query_text}'")
#     url = f"{BASE_URL}/query"
#     response = requests.post(url, json={"query": query_text})
#     print(f"Status Code: {response.status_code}")
#     print(f"Response: {response.json()}")
#     return response.status_code == 200

# def main():
#     print("Starting API Tests...")
    
#     # Test initialize endpoint
#     if not test_initialize():
#         print("Initialize test failed!")
#         return
    
#     # Wait a moment for the vector store to be ready
#     time.sleep(2)
    
#     # Test add-tools endpoint
#     if not test_add_tools():
#         print("Add tools test failed!")
#         return
    
#     # Wait a moment for the vector store to update
#     time.sleep(2)
    
#     # Test queries
#     test_queries = [
#         "I need an AI tool for generating images",
#         "What's the best tool for code completion?",
#         "I need a translation service",
#         "Recommend a tool for fraud detection"
#     ]
    
#     for query in test_queries:
#         if not test_query(query):
#             print(f"Query test failed for: {query}")
#             return
#         time.sleep(1)  # Wait between queries
    
#     print("\nAll tests completed successfully!")

# if __name__ == "__main__":
#     main()
# # import requests

# # url = "http://localhost:8000/rag/"
# # headers = {
# #     "Content-Type": "application/json"
# # }
# # data = {
# #     "user_query": "I need to learn dancing",
# # #     "tool_data": [
# # #     {
# # #         "name": "danceAI",
# # #         "description": "Generates videos to help in dancing.",
# # #         "api_details": "Endpoint: /api/dance, Method: POST, Params: {'style': 'vintage'}"
# # #     }
# # # ]
# # }

# # response = requests.post(url, json=data, headers=headers)
# # print(response.status_code)
# # print(response.json())


# # # curl -X POST http://localhost:8000/rag/ \
# # #   -H "Content-Type: application/json" \
# # #   -d '{
# # #     "tool_data": [
# # #       {
# # #         "name": "ImageGen",
# # #         "description": "AI image generator",
# # #         "api_details": "POST /generate"
# # #       }
# # #     ],
# # #     "user_query": "I need an image generation tool"
# # #   }' 

# # # [
# # #     {
# # #         "name": "VintageImageGen",
# # #         "description": "Generates vintage-style images using AI algorithms.",
# # #         "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
# # #     },
# # #     {
# # #         "name": "ModernImageGen",
# # #         "description": "Generates modern images with high resolution and clarity.",
# # #         "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"
# # #     },
# # #     {
# # #         "name": "AIChat",
# # #         "description": "An AI-powered chatbot for conversational purposes.",
# # #         "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"
# # #     },
# # #     {
# # #         "name": "ChatGPT",
# # #         "description": "A conversational AI developed by OpenAI for natural language understanding.",
# # #         "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"
# # #     },
# # #     {
# # #         "name": "DALL-E",
# # #         "description": "Generates creative images from textual descriptions using AI.",
# # #         "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"
# # #     },
# # #     {
# # #         "name": "Midjourney",
# # #         "description": "An AI tool that creates artistic images based on text prompts.",
# # #         "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"
# # #     },
# # #     {
# # #         "name": "StableDiffusion",
# # #         "description": "A latent diffusion model for generating detailed images from text.",
# # #         "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"
# # #     },
# # #     {
# # #         "name": "Copilot",
# # #         "description": "An AI pair programmer that assists with code completion and generation.",
# # #         "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"
# # #     },
# # #     {
# # #         "name": "DeepLTranslate",
# # #         "description": "An AI-powered translation service for multiple languages.",
# # #         "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"
# # #     },
# # #     {
# # #         "name": "VoiceClone",
# # #         "description": "Clones and synthesizes human voices using advanced AI techniques.",
# # #         "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"
# # #     },
# # #     {
# # #         "name": "SentimentAnalyzer",
# # #         "description": "Analyzes text to determine the sentiment using AI.",
# # #         "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"
# # #     },
# # #     {
# # #         "name": "RecommenderAI",
# # #         "description": "Provides personalized recommendations based on user data and AI analysis.",
# # #         "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"
# # #     },
# # #     {
# # #         "name": "FraudDetector",
# # #         "description": "Detects fraudulent activities using sophisticated AI algorithms.",
# # #         "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"
# # #     },
# # #     {
# # #         "name": "AnomalyFinder",
# # #         "description": "Identifies anomalies in datasets using high-sensitivity AI models.",
# # #         "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"
# # #     },
# # #     {
# # #         "name": "VirtualAssistant",
# # #         "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.",
# # #         "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"
# # #     }
# # # ]