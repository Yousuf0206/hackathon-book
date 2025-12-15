"""
Custom authentication system for the Physical AI & Humanoid Robotics book application.
This implements user authentication with background information collection.
"""
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import sqlite3
import json
from contextlib import contextmanager


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserBackground(BaseModel):
    """
    User background information collected during signup.
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


class UserCreate(BaseModel):
    """
    Schema for user creation (signup).
    """
    email: str
    password: str
    name: str
    background: Optional[UserBackground] = None


class UserUpdate(BaseModel):
    """
    Schema for user profile updates.
    """
    name: Optional[str] = None
    background: Optional[UserBackground] = None


class UserInDB(BaseModel):
    """
    Schema for user stored in database.
    """
    id: int
    email: str
    name: str
    hashed_password: str
    background: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    is_active: bool = True


class Token(BaseModel):
    """
    JWT token schema.
    """
    access_token: str
    token_type: str
    user_id: int
    email: str
    name: str


class TokenData(BaseModel):
    """
    Decoded token data schema.
    """
    user_id: int
    email: str


class AuthSystem:
    """
    Custom authentication system for the application.
    """

    def __init__(self):
        self.secret_key = os.getenv("AUTH_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.db_path = os.getenv("DATABASE_PATH", "users.db")

        # Initialize database
        self._init_db()

    def _init_db(self):
        """
        Initialize the database with users table.
        """
        with self._get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    hashed_password TEXT NOT NULL,
                    background TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            conn.commit()

    @contextmanager
    def _get_db_connection(self):
        """
        Get database connection with proper context management.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against the hashed password.
        """
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """
        Hash a password.
        """
        return pwd_context.hash(password)

    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Get a user by email from the database.
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT id, email, name, hashed_password, background, created_at, updated_at, is_active FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()

            if row:
                background = json.loads(row[4]) if row[4] else None
                return UserInDB(
                    id=row[0],
                    email=row[1],
                    name=row[2],
                    hashed_password=row[3],
                    background=background,
                    created_at=row[5],
                    updated_at=row[6],
                    is_active=bool(row[7])
                )
            return None

    def create_user(self, user_create: UserCreate) -> UserInDB:
        """
        Create a new user in the database.
        """
        # Check if user already exists
        existing_user = self.get_user_by_email(user_create.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )

        # Hash the password
        hashed_password = self.get_password_hash(user_create.password)

        # Prepare background data
        background_json = json.dumps(user_create.background.dict()) if user_create.background else None

        # Current timestamp
        now = datetime.utcnow().isoformat()

        with self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, name, hashed_password, background, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_create.email,
                    user_create.name,
                    hashed_password,
                    background_json,
                    now,
                    now
                )
            )
            user_id = cursor.lastrowid
            conn.commit()

        # Return the created user
        return UserInDB(
            id=user_id,
            email=user_create.email,
            name=user_create.name,
            hashed_password=hashed_password,
            background=user_create.background.dict() if user_create.background else None,
            created_at=now,
            updated_at=now,
            is_active=True
        )

    def authenticate_user(self, email: str, password: str) -> Optional[UserInDB]:
        """
        Authenticate a user by email and password.
        """
        user = self.get_user_by_email(email)
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        return user

    def create_access_token(self, data: dict) -> str:
        """
        Create a JWT access token.
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify a JWT token and return the user data.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: int = payload.get("user_id")
            email: str = payload.get("email")
            if user_id is None or email is None:
                return None
            return TokenData(user_id=user_id, email=email)
        except JWTError:
            return None

    def update_user_profile(self, user_id: int, user_update: UserUpdate) -> UserInDB:
        """
        Update user profile information.
        """
        # Get current user
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT id, email, name, hashed_password, background, created_at, updated_at, is_active FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )

            # Prepare updated data
            name = user_update.name if user_update.name else row[2]
            background = user_update.background.dict() if user_update.background else json.loads(row[4]) if row[4] else None
            background_json = json.dumps(background) if background else None
            updated_at = datetime.utcnow().isoformat()

            # Update the user
            conn.execute(
                """
                UPDATE users
                SET name = ?, background = ?, updated_at = ?
                WHERE id = ?
                """,
                (name, background_json, updated_at, user_id)
            )
            conn.commit()

        # Return updated user
        return UserInDB(
            id=row[0],
            email=row[1],
            name=name,
            hashed_password=row[3],
            background=background,
            created_at=row[5],
            updated_at=updated_at,
            is_active=bool(row[7])
        )


# Initialize the auth system
auth_system = AuthSystem()


# Security scheme for API endpoints
security = HTTPBearer()


async def get_current_user(request: Request) -> Optional[UserInDB]:
    """
    Dependency to get the current authenticated user from the token.
    """
    credentials: HTTPAuthorizationCredentials = await security(request)
    token_data = auth_system.verify_token(credentials.credentials)

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_system.get_user_by_email(token_data.email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user