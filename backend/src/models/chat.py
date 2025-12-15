"""
Model for chat sessions and related entities.
Based on the ChatSession entity from the data model.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4


class Message(BaseModel):
    """
    Represents a single message in a chat session.
    """
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatSession(BaseModel):
    """
    Represents a chat session with conversation history.
    """
    id: str
    created_at: datetime
    updated_at: datetime
    user_id: Optional[str] = None
    messages: List[Message] = []

    def __init__(self, **kwargs):
        super().__init__(
            id=kwargs.get('id', str(uuid4())),
            created_at=kwargs.get('created_at', datetime.now()),
            updated_at=kwargs.get('updated_at', datetime.now()),
            **{k: v for k, v in kwargs.items() if k not in ['id', 'created_at', 'updated_at']}
        )

    def add_message(self, role: str, content: str):
        """
        Add a message to the session.
        """
        message = Message(
            id=str(uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_messages(self) -> List[Message]:
        """
        Get all messages in the session.
        """
        return self.messages