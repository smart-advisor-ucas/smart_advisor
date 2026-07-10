"""
/chat endpoints — the frontend talks to these.
"""
from fastapi import APIRouter

from src.utils.schemas import ChatRequest, ChatResponse
from src.rag.services.chat_service import session_store, process_message

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session = session_store.get(req.session_id)
    reply = process_message(session, req.message)
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        onboarding=session.onboarding,
        awaiting_info=session.awaiting_info,
    )


@router.post("/reset", response_model=ChatResponse)
def reset_chat(req: ChatRequest) -> ChatResponse:
    """Start a new conversation for this session_id (e.g. 'محادثة جديدة' button)."""
    session = session_store.reset(req.session_id)
    greeting = (
        "مرحباً! أنا المستشار الأكاديمي الذكي في UCAS.\n"
        "قبل أن نبدأ، أودّ معرفة بعض المعلومات عنك لأتمكن من مساعدتك بشكل أفضل.\n"
        "للبدء، ما معدلك في الثانوية العامة (التوجيهي)؟\n"
        "هذا يساعدني في معرفة البرامج التي تؤهل للقبول.\n"
    )
    session.history.append({"role": "assistant", "content": greeting})
    return ChatResponse(
        session_id=req.session_id,
        reply=greeting,
        onboarding=session.onboarding,
        awaiting_info=session.awaiting_info,
    )


@router.get("/history/{session_id}")
def get_history(session_id: str):
    session = session_store.get(session_id)
    return {"session_id": session_id, "history": session.history}
