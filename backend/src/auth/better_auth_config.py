"""
Better Auth configuration for the Physical AI & Humanoid Robotics book application.
This configuration sets up authentication with user background information collection.
"""
from better_automation.auth import Auth, UserSchema
from pydantic import BaseModel, Field
from typing import Optional
import os


class UserBackground(BaseModel):
    """
    Extended user profile with background information collected during signup.
    """
    software_experience: Optional[str] = Field(
        default=None,
        description="User's software experience level (beginner/intermediate/advanced)"
    )
    hardware_familiarity: Optional[str] = Field(
        default=None,
        description="User's hardware familiarity (low-end/mid-range/high-performance systems)"
    )
    learning_goals: Optional[str] = Field(
        default=None,
        description="User's learning goals or objectives"
    )
    created_at: Optional[str] = Field(
        default=None,
        description="Timestamp of profile creation"
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="Timestamp of last profile update"
    )


# Extend the default UserSchema with background information
class ExtendedUserSchema(UserSchema):
    background: Optional[UserBackground] = Field(
        default=None,
        description="Extended user background information"
    )


# Initialize Better Auth with custom user schema
auth = Auth(
    secret=os.getenv("BETTER_AUTH_SECRET", "your-secret-key-change-in-production"),
    database_url=os.getenv("DATABASE_URL", "sqlite:///./test.db"),  # Use proper DB in production
    user_schema=ExtendedUserSchema,
    # Additional configuration options
    rate_limiting={
        "enabled": True,
        "window": 60,  # 60 seconds
        "max_requests": 100
    },
    email_verification={
        "enabled": False,  # Enable in production
        "from_email": os.getenv("EMAIL_FROM", "noreply@example.com")
    }
)


def get_auth():
    """
    Returns the configured auth instance.
    """
    return auth