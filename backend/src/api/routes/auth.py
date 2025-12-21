"""
Authentication API routes for signup, signin, and profile management.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
import os

from ...auth.auth_system import auth_system, UserCreate, UserUpdate, Token, get_current_user, UserInDB
from ...models.chat import ChatSession  # Using existing model structure

router = APIRouter()


@router.post("/auth/signup", response_model=Token)
async def signup(user_create: UserCreate):
    """
    Register a new user with background information.
    """
    try:
        # Create the user in the database
        user = auth_system.create_user(user_create)

        # Create access token
        token_data = {
            "user_id": user.id,
            "email": user.email,
            "name": user.name
        }
        access_token = auth_system.create_access_token(data=token_data)

        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=user.id,
            email=user.email,
            name=user.name
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during signup: {str(e)}"
        )


@router.post("/auth/signin", response_model=Token)
async def signin(email: str, password: str):
    """
    Authenticate a user and return an access token.
    """
    try:
        # Authenticate the user
        user = auth_system.authenticate_user(email, password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )

        # Create access token
        token_data = {
            "user_id": user.id,
            "email": user.email,
            "name": user.name
        }
        access_token = auth_system.create_access_token(data=token_data)

        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=user.id,
            email=user.email,
            name=user.name
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during signin: {str(e)}"
        )


@router.get("/auth/profile", response_model=UserInDB)
async def get_profile(current_user: UserInDB = Depends(get_current_user)):
    """
    Get the current user's profile information.
    """
    return current_user


@router.put("/auth/profile", response_model=UserInDB)
async def update_profile(
    user_update: UserUpdate,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Update the current user's profile information.
    """
    try:
        updated_user = auth_system.update_user_profile(current_user.id, user_update)
        return updated_user
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during profile update: {str(e)}"
        )


@router.post("/auth/logout")
async def logout():
    """
    Logout endpoint (client-side token removal is sufficient).
    """
    # In a stateless JWT system, logout is handled on the client side
    # by removing the token from local storage. This endpoint can be
    # used to perform any cleanup operations if needed.
    return {"message": "Successfully logged out"}