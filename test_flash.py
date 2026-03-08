import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini_flash():
    print("Testing Gemini 1.5 Flash...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content("Say 'Hello'")
        print(f"✅ Gemini 1.5 Flash is working! Response: {response.text}")
    except Exception as e:
        print(f"❌ Gemini 1.5 Flash failed: {e}")

if __name__ == "__main__":
    test_gemini_flash()
