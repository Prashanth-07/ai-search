import requests

url = "http://localhost:8000/rag/"
headers = {
    "Content-Type": "application/json"
}
data = {
    "user_query": "I need to learn dancing",
#     "tool_data": [
#     {
#         "name": "danceAI",
#         "description": "Generates videos to help in dancing.",
#         "api_details": "Endpoint: /api/dance, Method: POST, Params: {'style': 'vintage'}"
#     }
# ]
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code)
print(response.json())


# curl -X POST http://localhost:8000/rag/ \
#   -H "Content-Type: application/json" \
#   -d '{
#     "tool_data": [
#       {
#         "name": "ImageGen",
#         "description": "AI image generator",
#         "api_details": "POST /generate"
#       }
#     ],
#     "user_query": "I need an image generation tool"
#   }' 

# [
#     {
#         "name": "VintageImageGen",
#         "description": "Generates vintage-style images using AI algorithms.",
#         "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
#     },
#     {
#         "name": "ModernImageGen",
#         "description": "Generates modern images with high resolution and clarity.",
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