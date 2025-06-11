"""
Base storage class with common functionality for all storage providers
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class BaseFileStorage:
    """Base storage class with common functionality"""

    def sanitize_path_component(self, component: str) -> str:
        """Sanitize agentID/userID/filename for safe storage"""
        # Remove path traversal attempts
        component = component.replace("..", "").replace("/", "").replace("\\", "")

        # Replace problematic characters
        component = re.sub(r'[<>:"|?*]', "_", component)

        # Limit length (S3 key max is 1024 chars, folder + filename)
        if len(component) > 100:
            name, ext = os.path.splitext(component)
            component = name[:95] + ext

        return component

    def generate_storage_key(
        self, folder_name: str, filename: str, file_id: str
    ) -> str:
        """
        Generate unique storage key with folder structure

        Args:
            folder_name: "agent123", "user456", or "public"
            filename: "document.pdf"
            file_id: Unique file identifier

        Returns:
            "agent123/document_file-123_20241201_143022.pdf"
        """
        # Sanitize inputs
        folder_name = self.sanitize_path_component(folder_name)
        filename = self.sanitize_path_component(filename)

        # Generate unique filename to prevent overwrites
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{name}_{file_id[:8]}_{timestamp}{ext}"

        return f"{folder_name}/{unique_filename}"

    def get_folder_name(self, user_id: str, agent_id: Optional[str] = None) -> str:
        """
        Determine folder name based on agentID/userID priority

        Priority:
        1. agentID (if provided)
        2. userID (if not "public")
        3. "public" (fallback)

        Args:
            user_id: User ID from authentication
            agent_id: Agent ID from request (optional)

        Returns:
            Folder name to use
        """
        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"get_folder_name called with user_id='{user_id}', agent_id='{agent_id}'"
            )

        if agent_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using agent_id as folder: {agent_id}")
            return agent_id
        elif user_id and user_id != "public":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using user_id as folder: {user_id}")
            return user_id
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using 'public' as folder (user_id was '{user_id}')")
            return "public"
