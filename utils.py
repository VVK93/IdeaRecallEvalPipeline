# utils.py
import logging
import os
import re
from pathlib import Path
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import Optional

# Import constants needed for setup or utility functions
from constants import (
    API_KEY, ERROR_API_KEY_MISSING, ERROR_OPENAI_INIT, TOKENIZER_MODEL,
    ERROR_TOKENIZER_INIT, ERROR_TOKEN_COUNT, ERROR_FETCH_TRANSCRIPT,
    ERROR_TRANSCRIPT_DISABLED, ERROR_NO_TRANSCRIPT
)

# --- Setup Logging ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Create a logger instance that modules can import
logger = logging.getLogger(__name__) # Use __name__ for module-specific logging context

# --- Load Environment Variables ---
env_path = Path(__file__).resolve().parent / '.env' # Use resolve() for robustness
load_dotenv(dotenv_path=env_path)
logger.info(f"Attempting to load .env from: {env_path}")

# --- Initialize OpenAI Client ---
client: Optional[OpenAI] = None
if not API_KEY:
    logger.critical(ERROR_API_KEY_MISSING)
    # Error will be raised in app.py if client remains None
else:
    try:
        client = OpenAI(api_key=API_KEY)
        client.models.list() # Quick check to validate key early
        logger.info("OpenAI client initialized successfully.")
    except OpenAIError as e:
        logger.critical(ERROR_OPENAI_INIT.format(e), exc_info=True)
        # Error will be raised in app.py if client remains None
    except Exception as e: # Catch potential network or other issues
        logger.critical(f"Unexpected error initializing OpenAI client: {e}", exc_info=True)
        # Error will be raised in app.py if client remains None

# --- Initialize Tokenizer ---
tokenizer = None
try:
    tokenizer = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    logger.info(f"Tokenizer for model '{TOKENIZER_MODEL}' initialized.")
except Exception as e:
    logger.critical(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, e), exc_info=True)
    # Error will be raised in app.py if tokenizer remains None

# --- Utility Functions ---
def count_tokens(text: str) -> int:
    """Counts tokens using the global tokenizer."""
    if tokenizer is None:
        logger.error("Tokenizer not initialized, cannot count tokens.")
        return -1 # Indicate error
    if not text: return 0
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(ERROR_TOKEN_COUNT.format(e), exc_info=True)
        return -1 # Indicate error

def extract_video_id(url: str) -> Optional[str]:
    """Extracts YouTube video ID."""
    patterns = [
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.info(f"Extracted video ID: {video_id} from URL")
            return video_id
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

def clean_transcript(text: str) -> str:
    """Basic cleaning of transcript text."""
    original_len = len(text)
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}(?:.\d{3})?\]', '', text) # Timestamps
    text = re.sub(r'Speaker \d+:', '', text) # Speaker tags
    text = re.sub(r'\[Music\]', '', text, flags=re.IGNORECASE) # Music cues
    text = re.sub(r'\s+', ' ', text).strip() # Collapse whitespace
    # Example: Remove specific ad phrases (use carefully)
    text = re.sub(r'Link in the description to apply.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'apply for a spot.*', '', text, flags=re.IGNORECASE)
    cleaned_len = len(text)
    logger.info(f"Transcript cleaned. Original length: {original_len} -> Cleaned length: {cleaned_len} chars.")
    return text

def fetch_transcript(video_id: str) -> str:
    """Fetches and concatenates transcript text."""
    try:
        # You might want to specify language preferences here, e.g., languages=['en']
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        logger.info(f"Fetched transcript for video {video_id}. Length: {len(full_transcript)} chars.")
        return full_transcript
    except TranscriptsDisabled:
        logger.error(f"Transcripts disabled for video: {video_id}")
        raise ValueError(ERROR_TRANSCRIPT_DISABLED)
    except NoTranscriptFound:
        logger.error(f"No transcript found for video: {video_id}")
        raise ValueError(ERROR_NO_TRANSCRIPT)
    except Exception as e:
        logger.error(f"Error fetching transcript for video {video_id}: {e}", exc_info=True)
        raise RuntimeError(ERROR_FETCH_TRANSCRIPT.format(e))