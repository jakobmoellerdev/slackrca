# rca/models.py
from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    text: str

class Thread(BaseModel):
    channel_id: str
    thread_ts: str
    messages: List[Message]
