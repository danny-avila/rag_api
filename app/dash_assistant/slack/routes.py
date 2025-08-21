# app/dash_assistant/slack/routes.py
"""Slack integration routes for dash assistant."""
import hashlib
import hmac
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Form, status
from pydantic import BaseModel

from app.config import logger, get_env_variable
from app.dash_assistant.db import DashAssistantDB
from app.dash_assistant.serving.retriever import DashRetriever
from app.dash_assistant.serving.answer_builder import build_answer
from .blocks import build_slack_response_for_query


# Slack configuration
SLACK_SIGNING_SECRET = get_env_variable("SLACK_SIGNING_SECRET", "test_secret_for_development")


class SlackCommandRequest(BaseModel):
    """Slack slash command request model."""
    token: str
    team_id: str
    team_domain: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    command: str
    text: str
    response_url: str
    trigger_id: str


# Router
router = APIRouter(prefix="/slack", tags=["slack-integration"])


def verify_slack_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    """Verify Slack request signature.
    
    Args:
        request_body: Raw request body
        timestamp: Request timestamp
        signature: Slack signature from headers
        
    Returns:
        bool: True if signature is valid
    """
    if not SLACK_SIGNING_SECRET or SLACK_SIGNING_SECRET == "test_secret_for_development":
        # In development/test mode, skip signature verification
        logger.warning("Slack signature verification skipped (development mode)")
        return True
    
    # Check timestamp to prevent replay attacks
    current_time = int(time.time())
    if abs(current_time - int(timestamp)) > 300:  # 5 minutes
        logger.warning("Slack request timestamp too old")
        return False
    
    # Verify signature
    sig_basestring = f"v0:{timestamp}:{request_body.decode()}"
    expected_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)


@router.post("/command")
async def handle_slash_command(
    request: Request,
    token: str = Form(...),
    team_id: str = Form(...),
    team_domain: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    command: str = Form(...),
    text: str = Form(...),
    response_url: str = Form(...),
    trigger_id: str = Form(...)
):
    """Handle Slack slash command for dashboard search.
    
    This endpoint handles slash commands like /dash-search <query>
    and returns formatted Slack blocks with search results.
    """
    logger.info(f"Slack command received: {command} {text} from user {user_name}")
    
    # Verify Slack signature
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    
    if not verify_slack_signature(body, timestamp, signature):
        logger.error("Invalid Slack signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid request signature"
        )
    
    # Check database health
    if not await DashAssistantDB.health_check():
        logger.error("Database health check failed")
        return {
            "response_type": "ephemeral",
            "text": "❌ Сервис временно недоступен. Попробуйте позже."
        }
    
    # Validate command
    if not text.strip():
        return {
            "response_type": "ephemeral",
            "text": "❓ Пожалуйста, укажите поисковый запрос. Например: `/dash-search revenue analytics`"
        }
    
    try:
        # Log query to database
        qid = await _log_slack_query(user_id, user_name, text, channel_id)
        
        # Perform search
        retriever = DashRetriever()
        candidates = await retriever.search(query=text, top_k=5)
        
        # Build answer
        answer = build_answer(text, candidates)
        
        # Build Slack response
        slack_response = build_slack_response_for_query(text, answer)
        
        # Add query ID to metadata for feedback tracking
        slack_response["metadata"]["qid"] = qid
        
        logger.info(f"Slack search completed: {len(answer['results'])} results for '{text}'")
        return slack_response
        
    except Exception as e:
        logger.error(f"Slack command processing failed: {e}")
        return {
            "response_type": "ephemeral",
            "text": f"❌ Произошла ошибка при поиске: {str(e)}"
        }


@router.post("/interactive")
async def handle_interactive_component(request: Request):
    """Handle Slack interactive components (button clicks).
    
    This endpoint handles feedback button clicks from Slack messages.
    """
    logger.info("Slack interactive component received")
    
    # Verify Slack signature
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    
    if not verify_slack_signature(body, timestamp, signature):
        logger.error("Invalid Slack signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid request signature"
        )
    
    try:
        # Parse payload (Slack sends form-encoded JSON)
        form_data = await request.form()
        payload_str = form_data.get("payload", "")
        
        if not payload_str:
            raise ValueError("No payload in request")
        
        import json
        payload = json.loads(payload_str)
        
        # Extract action information
        user = payload.get("user", {})
        actions = payload.get("actions", [])
        
        if not actions:
            raise ValueError("No actions in payload")
        
        action = actions[0]
        action_id = action.get("action_id", "")
        value = action.get("value", "")
        
        logger.info(f"Interactive action: {action_id} = {value} from user {user.get('id')}")
        
        # Process feedback action
        if action_id.startswith("feedback_"):
            await _process_feedback_action(action_id, value, user)
            
            # Return updated message or acknowledgment
            return {
                "text": f"Спасибо за обратную связь! {value == 'up' and '👍' or '👎'}"
            }
        
        return {"text": "Действие обработано"}
        
    except Exception as e:
        logger.error(f"Interactive component processing failed: {e}")
        return {
            "text": "❌ Ошибка при обработке действия"
        }


async def _log_slack_query(user_id: str, user_name: str, query_text: str, channel_id: str) -> int:
    """Log Slack query to database.
    
    Args:
        user_id: Slack user ID
        user_name: Slack user name
        query_text: Search query
        channel_id: Slack channel ID
        
    Returns:
        int: Query ID (qid)
    """
    try:
        # Insert query log
        qid = await DashAssistantDB.fetch_value("""
            INSERT INTO query_log (user_id, query_text, intent_json)
            VALUES ($1, $2, $3)
            RETURNING qid
        """, user_id, query_text, {
            "source": "slack",
            "user_name": user_name,
            "channel_id": channel_id
        })
        
        logger.debug(f"Logged Slack query: qid={qid}")
        return qid
        
    except Exception as e:
        logger.error(f"Failed to log Slack query: {e}")
        # Return dummy ID if logging fails
        return -1


async def _process_feedback_action(action_id: str, value: str, user: Dict[str, Any]):
    """Process feedback action from Slack interactive component.
    
    Args:
        action_id: Action ID (e.g., "feedback_up_123")
        value: Feedback value ("up" or "down")
        user: Slack user information
    """
    try:
        # Extract entity_id from action_id
        # Format: feedback_{up|down}_{entity_id}
        parts = action_id.split("_")
        if len(parts) >= 3:
            entity_id = int(parts[2])
        else:
            logger.warning(f"Invalid action_id format: {action_id}")
            return
        
        # For now, we don't have qid in the action, so we'll skip the database update
        # In a full implementation, you'd need to store qid in the action or message metadata
        logger.info(f"Feedback received: entity_id={entity_id}, feedback={value}, user={user.get('id')}")
        
        # TODO: Update query_log with feedback when qid is available
        # This would require storing qid in the Slack message metadata
        
    except Exception as e:
        logger.error(f"Failed to process feedback action: {e}")


@router.get("/health")
async def slack_health_check():
    """Health check for Slack integration."""
    return {
        "status": "healthy",
        "service": "slack-integration",
        "signing_secret_configured": bool(SLACK_SIGNING_SECRET and SLACK_SIGNING_SECRET != "test_secret_for_development")
    }
