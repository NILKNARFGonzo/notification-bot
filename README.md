# Hugging Face Mention Bot with Ollama

A Python bot that monitors Hugging Face notifications for mentions of your username and automatically replies using Ollama.

## Features

- **Notification-based**: Efficiently monitors HF notifications instead of polling all discussions
- **Ollama integration**: Generates contextual replies using your local Ollama models
- **State persistence**: Tracks processed notifications to avoid duplicates
- **Retry logic**: Automatically retries failed posts (e.g., due to rate limits)
- **Status logging**: Clear logging of all operations including HTTP status codes

## Requirements

- Python 3.7+
- Hugging Face account with read-write token
- Ollama running locally with a compatible model

### Python packages
```bash
pip install huggingface_hub requests python-dateutil
```

## Quick Start

### 1. Get a Hugging Face token

1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Write** permissions
3. Save the token to a file (e.g., `~/.hf_token`)

**Note**: New accounts have a 0/hour comment creation quota that increases over time.

### 2. Pull an Ollama model

```bash
ollama pull llama3.2:3b-instruct-q6_K
```

### 3. Run the bot

```bash
# Basic usage (uses default token file and model)
python hf_mention_bot_general.py

# With custom arguments
python hf_mention_bot_general.py \
    --token-file ~/.hf_token \
    --ollama-model llama3.2:3b-instruct-q6_K \
    --ollama-host http://localhost:11434 \
    --poll-interval 60
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--token-file` | `/home/nilky/rlk.txt` | Path to file containing HF token |
| `--ollama-model` | `llama3.2:3b-instruct-q6_K` | Ollama model for generating replies |
| `--ollama-host` | `http://localhost:11434` | Ollama API endpoint |
| `--poll-interval` | `60` | Seconds between notification checks |
| `--mention-user` | *authenticated user* | Specific username to monitor for mentions |

## How It Works

1. **Poll Notifications**: Checks Hugging Face API for new notifications every `--poll-interval` seconds
2. **Detect Mentions**: Scans discussion titles, descriptions, and comments for `@YourUsername`
3. **Generate Reply**: Uses Ollama to create a contextual response
4. **Post Comment**: Attempts to post the reply to the discussion
5. **Track State**: Saves processed notifications to avoid duplicates
6. **Retry**: If posting fails (e.g., rate limit), it will retry on the next poll

## Rate Limits

New Hugging Face accounts have a **0/hour comment creation quota**. The bot will:
- Continue generating replies with Ollama
- Attempt to post on each poll cycle
- Succeed automatically once your quota increases

Quotas increase progressively over time. For immediate access, contact Hugging Face support.

## State File

The bot saves state to `hf_mention_bot_state.json` in the working directory. This tracks:
- Processed notification IDs
- Discussions already replied to

Delete this file to reset the bot's memory.

## Example Output

```
2026-09-20 01:37:14,390 - INFO - Bot initialized for @RogueLlama32
2026-09-20 01:37:14,390 - INFO - Using Ollama: llama3.2:3b-instruct-q6_K at http://localhost:11434
2026-09-20 01:37:14,556 - INFO - Found mention in comment abc123 at user/repo#3
2026-09-20 01:37:20,999 - INFO - Generated reply: Easy peasy! The answer to 1+1 is 2...
2026-09-20 01:37:20,999 - INFO - Attempting to post comment to user/repo/discussion#3
2026-09-20 01:37:21,069 - INFO - HF API POST /api/models/user/repo/discussions/3/comment - Status Code: 429
2026-09-20 01:37:21,069 - ERROR - HF API POST error: 429 Client Error: Too Many Requests
```

## Troubleshooting

### Ollama connection errors
- Ensure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama pull llama3.2:3b-instruct-q6_K`
- Check the host/port: `curl http://localhost:11434/api/tags`

### HF API errors
- Verify your token has **Write** permissions
- Check token file path and permissions
- New accounts: wait for quota to increase

### No mentions detected
- Ensure you're mentioned with `@Username` format
- Check the bot is running for the correct username
- Verify you have unread notifications on Hugging Face

## License

CC BY-SA 4.0 - Creative Commons Attribution-ShareAlike 4.0 International Public License

See [LICENSE](LICENSE) for full text.
