"""
Hugging Face Mention Bot with Ollama (Notification-Based)
Monitors Hugging Face NOTIFICATIONS for mentions of the authenticated user
and automatically replies using Ollama.

This is MORE EFFICIENT than polling all discussions because it only
processes discussions that have triggered notifications.

Requirements:
    pip install huggingface_hub requests python-dateutil

Configuration:
    Token file:  /home/nilky/rlk.txt (contains HF_TOKEN, one token per line)
    OLLAMA_HOST: Ollama API endpoint (default: http://localhost:11434)
    OLLAMA_MODEL: Model for generating replies (default: llama3.2:3b-instruct-q6_K)
    POLL_INTERVAL: Seconds between checks (default: 60)
    HF_USERNAME: Your HF username for posting (optional)

Usage:
    # Create token file with your read-write HF token
    echo "hf_..." > /home/nilky/rlk.txt

    # Run the bot
    python hf_mention_bot_notifications.py

    # Or with custom settings
    export OLLAMA_MODEL=mistral
    export POLL_INTERVAL=120
    python hf_mention_bot_notifications.py
"""

import os
import re
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path

import requests
from huggingface_hub import HfApi, login
from dateutil.parser import parse as parse_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hf_mention_bot")


class HFNotificationMentionBot:
    """
    Bot that monitors Hugging Face NOTIFICATIONS for mentions and replies using Ollama.

    This approach is more efficient than polling because:
     - Only processes discussions that have new notifications
     - Doesn't need to scan all repos
     - Leverages HF's built-in notification system

    Features:
     - Monitors notifications for the authenticated user
     - Detects when you're mentioned in discussions/PRs/issues
     - Generates contextual replies with Ollama
     - Tracks processed notifications to avoid duplicates
     - Persistent state across restarts
    """

    def __init__(
        self,
        hf_token: str,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b-instruct-q6_K",
        mention_username: Optional[str] = None,
        hf_username: Optional[str] = None,
        poll_interval: int = 60,
        state_file: str = "hf_mention_bot_notifications_state.json"
    ):
        """
        Initialize the bot.

        Args:
            hf_token: Hugging Face API token
            ollama_host: Ollama API host URL
            ollama_model: Ollama model for generating replies
            mention_username: Username to monitor (defaults to authenticated user)
            hf_username: Your Hugging Face username
            poll_interval: Seconds between polling cycles
            state_file: File to store processed notification IDs
        """
        self.hf_token = hf_token
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.mention_username = mention_username
        self.hf_username = hf_username
        self.poll_interval = poll_interval
        self.state_file = state_file
        self.HF_API_BASE = "https://huggingface.co"

        # Authenticate with Hugging Face
        login(token=hf_token)
        self.hf_api = HfApi(token=hf_token)

        # Get the authenticated user's username if mention_username not specified
        if not self.mention_username:
            try:
                user = self.hf_api.whoami()
                self.mention_username = user['name']
            except Exception as e:
                logger.error(f"Failed to get authenticated user: {e}")
                self.mention_username = "unknown"

        # Compile mention regex pattern
        self.mention_pattern = re.compile(
            rf'@{re.escape(self.mention_username)}\b',
            re.IGNORECASE
        )

        # Track processed notification event IDs AND discussion replies
        self.processed_notifications: Set[str] = set()
        self.processed_discussions: Set[str] = set()  # discussion:repo:num to avoid replying to same discussion
        self.load_state()

        logger.info(f"Bot initialized for @{self.mention_username}")
        logger.info(f"Using Ollama: {ollama_model} at {ollama_host}")
        logger.info(f"Monitoring notifications (not polling all discussions)")

    def load_state(self):
        """Load processed notification IDs from state file."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_notifications = set(state.get('processed_notifications', []))
                    self.processed_discussions = set(state.get('processed_discussions', []))
                logger.info(f"Loaded state with {len(self.processed_notifications)} notifications and {len(self.processed_discussions)} discussions")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Save processed notification IDs to state file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'processed_notifications': list(self.processed_notifications),
                    'processed_discussions': list(self.processed_discussions)
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def hf_get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a GET request to the Hugging Face API."""
        url = f"{self.HF_API_BASE}{endpoint}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            logger.info(f"HF API GET {endpoint} - Status Code: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HF API GET error {endpoint}: {e}")
            return None

    def hf_post(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Make a POST request to the Hugging Face API."""
        url = f"{self.HF_API_BASE}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            logger.info(f"HF API POST {endpoint} - Status Code: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HF API POST error {endpoint}: {e}")
            return None

    def check_mention(self, text: str) -> bool:
        """Check if text contains a mention of the target username."""
        if not text:
            return False
        return bool(self.mention_pattern.search(text))

    def generate_ollama_reply(self, context: str, mentioned_by: str) -> Optional[str]:
        """
        Generate a reply using Ollama API.

        Args:
            context: The text that contains the mention
            mentioned_by: Username who mentioned us

        Returns:
            Generated reply text, or None if failed
        """
        prompt = f"""You are @{self.mention_username}, a helpful AI assistant on Hugging Face.
You were mentioned by @{mentioned_by} in the following context:

{context[:1000]}

Respond in a helpful, friendly manner. Keep your reply concise (1-3 sentences).
Stay on topic and be relevant to the conversation.

Your response:"""

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 256
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            reply = result.get('response', '').strip()

            if reply:
                logger.info(f"Generated reply: {reply}")
                return reply

            logger.warning("Ollama returned empty response")
            return None

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error - is Ollama running at {self.ollama_host}? Error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return None

    # ============================================================
    # THIS IS THE ONLY METHOD THAT CHANGED
    # Uses huggingface_hub's built-in comment_discussion() instead
    # of raw HTTP POST, because HF doesn't expose a public REST
    # endpoint for posting discussion comments.
    # ============================================================
    def post_comment(self, repo_id: str, repo_type: str, discussion_num: int, content: str) -> bool:
        """
        Post a comment to a discussion using huggingface_hub's built-in method.

        Args:
            repo_id: Repository ID (e.g., username/repo)
            repo_type: 'model', 'dataset', or 'space'
            discussion_num: Discussion number
            content: Comment content

        Returns:
            True if successful
        """
        try:
            self.hf_api.comment_discussion(
                repo_id=repo_id,
                repo_type=repo_type,
                discussion_num=discussion_num,
                comment=content,
                token=self.hf_token
            )
            logger.info(f"Posted comment to {repo_id}/discussion#{discussion_num}")
            return True
        except Exception as e:
            logger.error(f"Failed to post comment via HfApi.comment_discussion(): {e}")
            return False
    # ============================================================

    def get_notifications(self) -> List[Dict]:
        """Get notifications from Hugging Face API."""
        endpoint = "/api/notifications"
        data = self.hf_get(endpoint)
        if data and 'notifications' in data:
            return data['notifications']
        return []

    def get_discussion_details(self, repo_id: str, repo_type: str, discussion_num: int) -> Optional[Dict]:
        """Get details for a specific discussion using the REST API."""
        endpoint = f"/api/{repo_type}s/{repo_id}/discussions/{discussion_num}"
        data = self.hf_get(endpoint)
        if data:
            return data
        return None

    def process_notification(self, notification: Dict) -> bool:
        """
        Process a single notification for mentions.

        Returns:
            True if a reply was posted
        """
        # Skip already read notifications (HF marks them as read once viewed in UI)
        if notification.get('read', True):
            logger.debug(f"Skipping read notification")
            return False

        # Create unique ID for this notification
        notif_id = f"notif:{notification.get('discussionEventId', notification.get('id', ''))}"
        if notif_id in self.processed_notifications:
            logger.debug(f"Skipping already processed notification: {notif_id}")
            return False

        logger.info(f"Processing notification: {notif_id}")

        # Extract repo and discussion info
        repo_info = notification.get('repo', {})
        repo_id = repo_info.get('name')
        repo_type = repo_info.get('type', 'model')
        discussion = notification.get('discussion', {})
        discussion_num = discussion.get('num')

        # If this is a comment notification, try to extract from comment data
        if not repo_id and notification.get('type') == 'comment':
            comment_data = notification.get('comment', {})
            if comment_data:
                repo_info = comment_data.get('repo', {})
                repo_id = repo_info.get('name')
                repo_type = repo_info.get('type', 'model')
                discussion = comment_data.get('discussion', {})
                discussion_num = discussion.get('num')

        if not repo_id or not discussion_num:
            logger.debug(f"Notification structure: {json.dumps(notification, indent=2, default=str)}")
            logger.warning(f"Notification missing repo or discussion info: {notif_id}")
            # Mark as processed to avoid re-checking
            self.processed_notifications.add(notif_id)
            self.save_state()
            return False

        # Track at discussion level to avoid replying multiple times
        disc_id = f"disc:{repo_id}:{discussion_num}"
        if disc_id in self.processed_discussions:
            logger.debug(f"Already replied to discussion {disc_id}")
            return False

        # Get full discussion details
        details = self.get_discussion_details(repo_id, repo_type, discussion_num)
        if not details:
            logger.warning(f"Failed to get discussion details for {repo_id}#{discussion_num}")
            return False

        # Track if we found any mention
        mention_found = False

        # Check if we're mentioned in the discussion title or description
        title = details.get('title', '')
        # Description might be in 'description' or in the first comment
        description = details.get('description', '') or details.get('comment', '')
        author_name = details.get('author', {}).get('name', 'unknown') if isinstance(details.get('author'), dict) else details.get('author', 'unknown')

        if self.check_mention(title) or self.check_mention(description):
            mention_found = True
            context = f"Discussion: {title}\n{description[:500]}"
            logger.info(f"Found mention in discussion #{discussion_num} at {repo_id}")

            reply = self.generate_ollama_reply(context, author_name)
            if reply:
                if self.post_comment(repo_id, repo_type, discussion_num, reply):
                    self.processed_discussions.add(disc_id)
                    self.processed_notifications.add(notif_id)
                    self.save_state()
                    return True

        # Check comments in this discussion
        if not mention_found:
            events = details.get('events', [])
            for event in events:
                if isinstance(event, dict) and event.get('type') == 'comment':
                    event_id = event.get('id', '')

                    # Track at comment level
                    comment_id = f"comment:{event_id}"
                    if comment_id in self.processed_notifications:
                        continue

                    # Extract content from the correct path in HF API
                    # Content can be in: event['content'] OR event['data']['latest']['raw']
                    content = event.get('content', '') or event.get('data', {}).get('latest', {}).get('raw', '')
                    comment_author = event.get('author', {}).get('name', 'unknown') if isinstance(event.get('author'), dict) else event.get('author', 'unknown')

                    if self.check_mention(content):
                        mention_found = True
                        logger.info(f"Found mention in comment {event_id} at {repo_id}#{discussion_num}")

                        reply = self.generate_ollama_reply(content, comment_author)
                        if reply:
                            if self.post_comment(repo_id, repo_type, discussion_num, reply):
                                self.processed_discussions.add(disc_id)
                                self.processed_notifications.add(notif_id)
                                self.processed_notifications.add(comment_id)
                                self.save_state()
                                return True
                            # If post fails, don't mark as processed - will retry next time

        # Only mark as processed if we couldn't find any mention
        if not mention_found:
            self.processed_notifications.add(notif_id)
            self.save_state()

        return False

    def run(self):
        """Main monitoring loop."""
        logger.info(f"Starting notification-based mention bot for @{self.mention_username}")

        while True:
            try:
                start_time = datetime.now(timezone.utc)
                logger.info(f"Starting poll cycle at {start_time.isoformat()}")

                # Get notifications
                notifications = self.get_notifications()
                logger.info(f"Found {len(notifications)} notifications")

                for notification in notifications:
                    try:
                        self.process_notification(notification)
                        # Rate limiting between notifications
                        time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error processing notification: {e}")

                # Save state periodically
                self.save_state()

                # Wait for next poll
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                sleep_time = max(0, self.poll_interval - elapsed)
                if sleep_time > 0:
                    logger.info(f"Poll completed in {elapsed:.1f}s. Sleeping {sleep_time:.1f}s...")
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(min(60, self.poll_interval))  # Wait before retry


def read_token_from_file(filepath: str = '/home/nilky/rlk.txt') -> str:
    """Read HF token from file."""
    try:
        with open(filepath, 'r') as f:
            token = f.read().strip()
        if not token:
            raise ValueError(f"Token file {filepath} is empty")
        if not token.startswith('hf_'):
            raise ValueError(f"Token in {filepath} doesn't look like a valid HF token")
        return token
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Token file not found at {filepath}. "
            "Create it with your read-write HF token."
        )
    except Exception as e:
        raise ValueError(f"Failed to read token file: {e}")


def main():
    """Entry point."""
    # Load configuration
    try:
        hf_token = read_token_from_file()
    except Exception as e:
        logger.error(f"Failed to load HF token: {e}")
        return

    # Optional configuration from environment
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:3b-instruct-q6_K')
    mention_username = os.getenv('MENTION_USERNAME')  # Optional - defaults to authenticated user
    hf_username = os.getenv('HF_USERNAME')
    poll_interval = int(os.getenv('POLL_INTERVAL', '60'))

    # Create and run bot
    try:
        bot = HFNotificationMentionBot(
            hf_token=hf_token,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            mention_username=mention_username,
            hf_username=hf_username,
            poll_interval=poll_interval
        )
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()