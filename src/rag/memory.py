"""
Conversation memory helpers (LangChain ConversationBufferWindowMemory with
a manually enforced window size).
"""
from langchain_classic.memory import ConversationBufferWindowMemory

from src.utils.config import MEMORY_WINDOW_K


def get_empty_memory() -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(k=MEMORY_WINDOW_K, return_messages=True, memory_key="history")


def save_to_memory(memory: ConversationBufferWindowMemory, user_msg: str, assistant_msg: str) -> None:
    """Save a turn and manually enforce the k window (2 messages per turn)."""
    memory.save_context({"input": user_msg}, {"output": assistant_msg})
    limit = MEMORY_WINDOW_K * 2
    if len(memory.chat_memory.messages) > limit:
        memory.chat_memory.messages = memory.chat_memory.messages[-limit:]


def history_as_messages(memory: ConversationBufferWindowMemory) -> list[dict]:
    """Convert stored memory into OpenAI-style chat messages."""
    chat_history = memory.load_memory_variables({})["history"]
    return [
        {"role": ("user" if m.type == "human" else "assistant"), "content": m.content}
        for m in chat_history
    ]
