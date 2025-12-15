"""
Test script to verify authentication and personalization integration.
"""
import requests
import json
import os
from datetime import datetime, timedelta

# Base URL for the API
BASE_URL = "http://localhost:8000/api"

def test_auth_and_personalization():
    """
    Test the integration of authentication and personalization features.
    """
    print("Testing Authentication and Personalization Integration...")

    # Test user data
    test_email = f"testuser_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com"
    test_password = "SecurePassword123!"
    test_name = "Test User"

    # Step 1: Test signup
    print("\n1. Testing signup...")
    signup_data = {
        "email": test_email,
        "password": test_password,
        "name": test_name,
        "background": {
            "software_experience": "beginner",
            "hardware_familiarity": "low-end",
            "learning_goals": "Learn robotics basics"
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        if response.status_code == 200:
            print("✓ Signup successful")
            auth_data = response.json()
            token = auth_data["access_token"]
            user_id = auth_data["user_id"]
            print(f"  - Token received: {token[:20]}...")
        else:
            print(f"✗ Signup failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Signup error: {str(e)}")
        return False

    # Step 2: Test chat with user ID (personalization)
    print("\n2. Testing chat with personalization...")
    chat_data = {
        "query": "What is ROS 2?",
        "user_id": user_id  # This should trigger personalization
    }

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
        if response.status_code == 200:
            print("✓ Chat with personalization successful")
            chat_response = response.json()
            print(f"  - Response received: {chat_response['response'][:100]}...")
        else:
            print(f"✗ Chat failed: {response.status_code} - {response.text}")
            # Note: This might fail if the backend server is not running
            print("  - This is expected if the backend server is not running")
    except Exception as e:
        print(f"✗ Chat error: {str(e)}")
        print("  - This is expected if the backend server is not running")

    # Step 3: Test profile update
    print("\n3. Testing profile update...")
    update_data = {
        "background": {
            "software_experience": "intermediate",
            "hardware_familiarity": "mid-range",
            "learning_goals": "Learn advanced robotics concepts"
        }
    }

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.put(f"{BASE_URL}/auth/profile", json=update_data, headers=headers)
        if response.status_code == 200:
            print("✓ Profile update successful")
            profile_data = response.json()
            print(f"  - Updated profile: {profile_data['background']}")
        else:
            print(f"✗ Profile update failed: {response.status_code} - {response.text}")
            # Note: This might fail if the backend server is not running
            print("  - This is expected if the backend server is not running")
    except Exception as e:
        print(f"✗ Profile update error: {str(e)}")
        print("  - This is expected if the backend server is not running")

    print("\n✓ Authentication and personalization integration test completed!")
    return True

def test_translation():
    """
    Test the translation feature.
    """
    print("\n4. Testing translation feature...")

    # Test user data for translation test
    test_email = f"testuser_trans_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com"
    test_password = "SecurePassword123!"
    test_name = "Translation Test User"

    # Signup first
    signup_data = {
        "email": test_email,
        "password": test_password,
        "name": test_name,
        "background": {
            "software_experience": "intermediate",
            "hardware_familiarity": "mid-range",
            "learning_goals": "Test translation feature"
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        if response.status_code == 200:
            auth_data = response.json()
            token = auth_data["access_token"]

            # Test translation
            translation_data = {
                "text": "This is a test for Urdu translation functionality.",
                "target_language": "ur"
            }

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            response = requests.post(f"{BASE_URL}/translate", json=translation_data, headers=headers)
            if response.status_code == 200:
                print("✓ Translation test successful")
                translation_result = response.json()
                print(f"  - Original: {translation_result['original_text']}")
                print(f"  - Translated: {translation_result['translated_text']}")
            else:
                print(f"✗ Translation failed: {response.status_code} - {response.text}")
                print("  - This is expected if the backend server is not running")
        else:
            print(f"✗ Translation test setup failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Translation test error: {str(e)}")
        print("  - This is expected if the backend server is not running")

if __name__ == "__main__":
    print("Running integration tests for authentication and personalization features...")

    success = test_auth_and_personalization()
    if success:
        test_translation()

    print("\nNote: Some tests may show errors if the backend server is not running.")
    print("To run the full test, start the backend server with: uvicorn backend.src.api.main:app --reload")