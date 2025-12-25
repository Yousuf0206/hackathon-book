# Authentication and Personalization Setup Guide

## Overview
This guide explains how to set up and run the authentication, personalization, and Urdu translation features for the Physical AI & Humanoid Robotics book application.

## Prerequisites
- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip package manager
- Git

## Backend Setup

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file in the `backend` directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
AUTH_SECRET_KEY=your_secret_key_for_jwt_tokens
ACCESS_TOKEN_EXPIRE_MINUTES=30
DATABASE_PATH=users.db
```

### 3. Run the Backend Server
```bash
cd backend
uvicorn src.api.main:app --reload
```
The backend will run on `http://localhost:8000`

## Frontend Setup

### 1. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 2. Run the Frontend Development Server
```bash
cd frontend
npm start
```

## Docusaurus Integration

### 1. Install Docusaurus Dependencies
```bash
npm install
```

### 2. Run Docusaurus Development Server
```bash
npm start
```
This will run the Docusaurus site on `http://localhost:3000`

## API Endpoints

### Authentication Endpoints
- `POST /api/auth/signup` - Create a new user account
- `POST /api/auth/signin` - Authenticate a user
- `GET /api/auth/profile` - Get current user profile (requires authentication)
- `PUT /api/auth/profile` - Update user profile (requires authentication)
- `POST /api/auth/logout` - Logout user

### Translation Endpoints
- `POST /api/translate` - Translate content to Urdu (requires authentication)

### Chat Endpoints
- `POST /api/chat` - Chat with personalization support (pass user_id for personalization)

## Using the Features

### 1. User Registration
- Navigate to the signup page (you can create a simple HTML page or integrate with your Docusaurus site)
- Fill in the registration form including background information:
  - Full name
  - Email address
  - Password
  - Software experience level (beginner/intermediate/advanced)
  - Hardware familiarity (low-end/mid-range/high-performance)
  - Learning goals (optional)

### 2. User Login
- Navigate to the signin page
- Enter your email and password
- Your session will be stored in localStorage

### 3. Chapter Personalization
- On any chapter page with the `<ChapterControls>` component
- Use the "Personalize Content" dropdown to select your preferred complexity level
- Click "Personalize Content" to adjust the material to your background

### 4. Urdu Translation
- On any chapter page with the `<ChapterControls>` component
- Click "Translate to Urdu" to convert the content
- Use "Reset to Original" to return to the English content

### 5. Personalized Chat
- The RAG chatbot will automatically personalize responses based on your profile when you're logged in
- The system considers your software experience and hardware familiarity when generating responses

## Testing the Integration

A test script is provided to verify the functionality:

```bash
cd backend
python test_auth_integration.py
```

Note: This test requires the backend server to be running.

## Security Notes

- JWT tokens are stored in localStorage (consider using httpOnly cookies in production)
- Passwords are hashed using bcrypt
- All sensitive API endpoints require authentication
- Rate limiting is implemented to prevent abuse

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**: Ensure the database file path is writable
2. **Authentication Fails**: Check that the AUTH_SECRET_KEY is consistent between runs
3. **Translation Not Working**: Verify that the user is authenticated before attempting translation
4. **Personalization Not Applying**: Check that user_id is being passed correctly to the chat endpoint

### Debugging Tips:
- Check the browser console for frontend errors
- Check the backend server logs for API errors
- Verify environment variables are properly set
- Ensure the database file has proper permissions

## Production Deployment Considerations

- Use a proper database (PostgreSQL, MySQL) instead of SQLite
- Implement httpOnly cookies for JWT tokens
- Set up proper SSL certificates
- Configure proper CORS settings
- Implement proper logging and monitoring
- Set up backup strategies for user data