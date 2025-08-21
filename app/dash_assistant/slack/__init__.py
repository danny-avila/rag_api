# app/dash_assistant/slack/__init__.py
"""Slack integration module for dash assistant."""

from .blocks import build_slack_blocks, build_slack_response_for_query
from .routes import router

__all__ = ["build_slack_blocks", "build_slack_response_for_query", "router"]
