"""
Hugging Face Mention Bot with Ollama (General Version)
====================================================

Monitors Hugging Face NOTIFICATIONS for mentions of the authenticated user
and automatically replies using Ollama.

Usage:
    python hf_mention_bot_general.py --ollama-model llama3.2:3b-instruct-q6_K --token-file /path/to/token.txt

Arguments:
    --ollama-model     Ollama model to use (default: llama3.2:3b-instruct-q6_K)
    --token-file       Path to file containing HF token (default: /home/nilky/rlk.txt)
    --ollama-host      Ollama API endpoint (default: http://localhost:11434)
    --poll-interval    Seconds between checks (default: 60)
    --mention-user     Username to monitor (default: authenticated user)
"""

import os
import re
import time
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Set

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
    def __init__(
        self,
        hf_token: str,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b-instruct-q6_K",
        mention_username: Optional[str] = None,
        hf_username: Optional[str] = None,
        poll_interval: int = 60,
        state_file: str = "hf_mention_bot_state.json"
    ):
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
        
        # Track processed notifications
        self.processed_notifications: Set[str] = set()
        self.processed_discussions: Set[str] = set()
        self.load_state()
        
        logger.info(f"Bot initialized for @{self.mention_username}")
        logger.info(f"Using Ollama: {ollama_model} at {ollama_host}")
        logger.info(f"Monitoring notifications (not polling all discussions)")
    
    def load_state(self):
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
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'processed_notifications': list(self.processed_notifications),
                    'processed_discussions': list(self.processed_discussions)
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def hf_get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
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
        if not text:
            return False
        return bool(self.mention_pattern.search(text))
    
    def generate_ollama_reply(self, context: str, mentioned_by: str) -> Optional[str]:
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
    
    def post_comment(self, repo_id: str, repo_type: str, discussion_num: int, content: str) -> bool:
        endpoint = f"/api/{repo_type}s/{repo_id}/discussions/{discussion_num}/comment"
        data = {"content": content}
        
        logger.info(f"Attempting to post comment to {repo_id}/discussion#{discussion_num}")
        
        result = self.hf_post(endpoint, data)
        if result:
            logger.info(f"Posted comment to {repo_id}/discussion#{discussion_num}")
            return True
        else:
            logger.warning(f"Failed to post comment. Endpoint: {endpoint}")
        return False
    
    def get_notifications(self) -> List[Dict]:
        endpoint = "/api/notifications"
        data = self.hf_get(endpoint)
        if data and 'notifications' in data:
            return data['notifications']
        return []
    
    def get_discussion_details(self, repo_id: str, repo_type: str, discussion_num: int) -> Optional[Dict]:
        endpoint = f"/api/{repo_type}s/{repo_id}/discussions/{discussion_num}"
        data = self.hf_get(endpoint)
        if data:
            return data
        return None
    
    def process_notification(self, notification: Dict) -> bool:
        # Skip already read notifications
        if notification.get('read', True):
            logger.debug(f"Skipping read notification")
            return False
        
        notif_id = f"notif:{notification.get('discussionEventId', notification.get('id', ''))}"
        
        if notif_id in self.processed_notifications:
            logger.debug(f"Skipping already processed notification: {notif_id}")
            return False
        
        logger.info(f"Processing notification: {notif_id}")
        
        repo_info = notification.get('repo', {})
        repo_id = repo_info.get('name')
        repo_type = repo_info.get('type', 'model')
        discussion = notification.get('discussion', {})
        discussion_num = discussion.get('num')
        
        if not repo_id or not discussion_num:
            logger.warning(f"Notification missing repo or discussion info: {notif_id}")
            return False
        
        disc_id = f"disc:{repo_id}:{discussion_num}"
        if disc_id in self.processed_discussions:
            logger.debug(f"Already replied to discussion {disc_id}")
            return False
        
        details = self.get_discussion_details(repo_id, repo_type, discussion_num)
        if not details:
            logger.warning(f"Failed to get discussion details for {repo_id}#{discussion_num}")
            return False
        
        mention_found = False
        
        title = details.get('title', '')
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
        
        if not mention_found:
            events = details.get('events', [])
            for event in events:
                if isinstance(event, dict) and event.get('type') == 'comment':
                    event_id = event.get('id', '')
                    comment_id = f"comment:{event_id}"
                    if comment_id in self.processed_notifications:
                        continue
                    
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
        
        if not mention_found:
            self.processed_notifications.add(notif_id)
            self.save_state()
        return False
    
    def run(self):
        logger.info(f"Starting notification-based mention bot for @{self.mention_username}")
        
        while True:
            try:
                start_time = datetime.now(timezone.utc)
                logger.info(f"Starting poll cycle at {start_time.isoformat()}")
                
                notifications = self.get_notifications()
                logger.info(f"Found {len(notifications)} notifications")
                
                for notification in notifications:
                    try:
                        self.process_notification(notification)
                        time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error processing notification: {e}")
                
                self.save_state()
                
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
                time.sleep(min(60, self.poll_interval))


def read_token_from_file(filepath: str) -> str:
    try:
        with open(filepath, 'r') as f:
            token = f.read().strip()
            if not token:
                raise ValueError(f"Token file {filepath} is empty")
            if not token.startswith('hf_'):
                raise ValueError(f"Token in {filepath} doesn't look like a valid HF token")
            return token
    except FileNotFoundError:
        raise FileNotFoundError(f"Token file not found at {filepath}. Create it with your read-write HF token.")
    except Exception as e:
        raise ValueError(f"Failed to read token file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Hugging Face Mention Bot with Ollama')
    parser.add_argument('--token-file', default='/home/nilky/rlk.txt',
                        help='Path to file containing Hugging Face token (one token per line)')
    parser.add_argument('--ollama-model', default='llama3.2:3b-instruct-q6_K',
                        help='Ollama model to use for generating replies')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                        help='Ollama API endpoint URL')
    parser.add_argument('--poll-interval', type=int, default=60,
                        help='Seconds between polling cycles')
    parser.add_argument('--mention-user',
                        help='Specific username to monitor for mentions (default: authenticated user)')
    
    args = parser.parse_args()
    
    try:
        hf_token = read_token_from_file(args.token_file)
    except Exception as e:
        logger.error(f"Failed to load HF token: {e}")
        return
    
    try:
        bot = HFNotificationMentionBot(
            hf_token=hf_token,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            mention_username=args.mention_user,
            poll_interval=args.poll_interval
        )
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
