
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.types import TokenGen
from private_gpt.server.utils.chunk import Chunk
from pydantic import BaseModel
from dataclasses import dataclass

class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None

class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        # Detect if there is a system message, extract the last message and chat history
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        # Remove from messages list the system message and last message,
        # if they exist. The rest is the chat history.
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )
