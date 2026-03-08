import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    print("Testing Gemini API...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            print("✅ Gemini API is working!")
        else:
            print(f"❌ Gemini API failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ Gemini API error: {e}")

def test_unstructured():
    print("\nTesting Unstructured API...")
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        print("❌ UNSTRUCTURED_API_KEY not found in .env")
        return
    
    url = os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructured.io/general/v0/general")
    headers = {"unstructured-api-key": api_key}
    # Just try a simple heartbeat/unauthorized check if it supports it, 
    # or just check if we can reach the endpoint.
    try:
        # We send an empty request to see if we get a 400 (Bad Request) vs 401 (Unauthorized)
        res = requests.post(url, headers=headers)
        if res.status_code == 401:
            print("❌ Unstructured API key is INVALID (401 Unauthorized)")
        elif res.status_code == 400:
             print("✅ Unstructured API key is VALID (Received 400 as expected for empty request)")
        else:
            print(f"ℹ️ Unstructured API returned: {res.status_code}")
    except Exception as e:
        print(f"❌ Unstructured API error: {e}")

if __name__ == "__main__":
    test_gemini()
    test_unstructured()
