import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    print("Testing Gemini API (SDK)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return
    
    try:
        # Use the official SDK to test
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents='Hello',
        )
        if response.text:
            print(f"✅ Gemini API is working! Output: {response.text[:20]}...")
        else:
            print("❌ Gemini API is working but returned empty text.")
    except Exception as e:
        print(f"❌ Gemini SDK error: {e}")

if __name__ == "__main__":
    test_gemini()
