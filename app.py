import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from evaluation.evaluator import Evaluator
import asyncio
import time
from datetime import datetime
from bert_score import score
import json
import tiktoken
from constants import *

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Validate OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OpenAI API key not found in environment variables")
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()
logging.info(f"API Key loaded successfully (length: {len(api_key)})")

# Initialize OpenAI client, evaluator, and tokenizer
client = OpenAI(api_key=api_key)
evaluator = Evaluator()
tokenizer = tiktoken.encoding_for_model("gpt-4")

def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
        r'youtube\.com\/embed\/([^&\n?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except TranscriptsDisabled:
        logging.error(f"Transcripts disabled for video {video_id}")
        raise Exception(ERROR_TRANSCRIPT_DISABLED)
    except NoTranscriptFound:
        logging.error(f"No transcript found for video {video_id}")
        raise Exception(ERROR_NO_TRANSCRIPT)
    except Exception as e:
        logging.error(f"Error fetching transcript for video {video_id}: {str(e)}")
        raise Exception(ERROR_FETCH_TRANSCRIPT.format(str(e)))

def generate_summary(transcript: str) -> tuple[str, float]:
    """Generate summary using OpenAI API with token-based chunking."""
    try:
        start_time = time.time()
        tokens = tokenizer.encode(transcript)
        max_tokens = 120000  # Adjust based on model limits (gpt-4-turbo supports ~128k)

        if len(tokens) <= max_tokens:
            logging.info("Processing transcript in one call")
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                    {"role": "user", "content": USER_PROMPT_SUMMARY.format(transcript)}
                ],
                temperature=0.7
            )
            summary = response.choices[0].message.content
        else:
            chunks = [transcript[i:i + max_tokens] for i in range(0, len(transcript), max_tokens)]
            summaries = []
            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                        {"role": "user", "content": USER_PROMPT_SUMMARY.format(chunk)}
                    ],
                    temperature=0.7
                )
                summaries.append(response.choices[0].message.content)
            combined = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMBINE},
                    {"role": "user", "content": USER_PROMPT_COMBINE.format(' '.join(summaries))}
                ],
                temperature=0.7
            )
            summary = combined.choices[0].message.content

        generation_time = round(time.time() - start_time, 2)
        return summary, generation_time
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        raise Exception(ERROR_GENERATE_SUMMARY.format(str(e)))

def validate_json_structure(summary: str) -> bool:
    """Validate if the input string is a valid JSON"""
    summary = summary.strip()
    logging.info(f"Attempting to validate JSON: {repr(summary)}")
    try:
        json.loads(summary)
        return True
    except json.JSONDecodeError as e:
        logging.error(f"JSON validation failed: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in JSON validation: {str(e)}")
        return False

async def evaluate_bert_score(transcript: str, summary: str) -> float:
    """Calculate BERT score between transcript and summary."""
    try:
        P, R, F1 = score([summary], [transcript], lang='en', verbose=False)
        return F1.mean().item()
    except Exception as e:
        logging.error(f"BERT score evaluation failed: {str(e)}")
        raise Exception(ERROR_EVALUATE_BERT.format(str(e)))

async def evaluate_summary(transcript: str, summary: str, video_id: str) -> dict:
    """Evaluate the summary using BERT score and JSON validation."""
    try:
        is_valid_json = validate_json_structure(summary)
        bert_score = await evaluate_bert_score(transcript, summary)
        return {
            "bert_score": bert_score,
            "passed_bert": bert_score >= BERT_THRESHOLD_VALUE,
            "is_valid_json": is_valid_json,
            "generation_id": video_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logging.error(f"Error in evaluation: {str(e)}")
        raise Exception(ERROR_EVALUATION.format(str(e)))

def display_metrics(metrics: dict, generation_time: float) -> None:
    """Display evaluation metrics in Streamlit."""
    st.subheader(METRICS_HEADER)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            BERT_LABEL,
            f"{metrics['bert_score']:.3f}",
            f"{'✅' if metrics['passed_bert'] else '❌'} {BERT_THRESHOLD}"
        )
    with col2:
        st.metric(
            "JSON Structure",
            "Valid" if metrics['is_valid_json'] else "Invalid",
            "✅" if metrics['is_valid_json'] else "❌"
        )
    with col3:
        st.metric(GENERATION_TIME_LABEL, f"{generation_time}s")
    with st.expander(DETAILED_METRICS_HEADER):
        st.write(f"Generation ID: {metrics['generation_id']}")
        st.write(f"Timestamp: {metrics['timestamp']}")
        st.write(f"BERT Score: {metrics['bert_score']:.3f}")
        st.write(f"JSON Structure Valid: {metrics['is_valid_json']}")
        st.write(f"Passed BERT Threshold: {metrics['passed_bert']}")
        st.write(f"{GENERATION_TIME_LABEL}: {generation_time}s")

def main():
    """Main function to run the Streamlit app."""
    st.title(TITLE)
    st.write(DESCRIPTION)

    # Input field for YouTube URL
    url = st.text_input(INPUT_LABEL)

    if url:
        try:
            # Extract video ID
            video_id = extract_video_id(url)
            
            if video_id:
                logging.info(f"Processing video ID: {video_id}")
                
                # Create containers for status updates
                status_container = st.empty()
                step_container = st.empty()
                
                # Step 1: Get transcript
                step_container.info(STATUS_FETCHING_TRANSCRIPT)
                transcript = fetch_transcript(video_id)
                
                if transcript:
                    # Show transcript length and collapsible transcript
                    step_container.empty()  # Clear step message
                    status_container.success(STATUS_TRANSCRIPT_SUCCESS)
                    st.info(TRANSCRIPT_INFO.format(len(transcript)))
                    
                    # Show transcript in expandable section
                    with st.expander("📝 View Original Transcript", expanded=False):
                        st.text(transcript)
                    
                    # Step 2: Generate summary
                    step_container.info(STATUS_GENERATING_SUMMARY)
                    summary, generation_time = generate_summary(transcript)
                    
                    if summary:
                        step_container.empty()  # Clear step message
                        status_container.success(STATUS_SUMMARY_SUCCESS)
                        
                        # Show generated output in expandable section
                        st.markdown(SUMMARY_HEADER.format(generation_time))
                        with st.expander("✨ View Generated Output", expanded=False):
                            st.write(summary)
                        
                        # Step 3: Run evaluation
                        step_container.info(STATUS_EVALUATING)
                        with st.spinner("Evaluating summary..."):
                            metrics = asyncio.run(evaluate_summary(transcript, summary, video_id))
                            
                            # Display evaluation metrics
                            st.subheader(METRICS_HEADER)
                            
                            # Step 1: Automatic Checks
                            st.markdown("### Step 1: Automatic Checks")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    BERT_LABEL,
                                    f"{metrics['bert_score']:.3f}",
                                    f"{'✅' if metrics['passed_bert'] else '❌'} {BERT_THRESHOLD}"
                                )
                            with col2:
                                st.metric(
                                    "JSON Structure",
                                    "Valid" if metrics['is_valid_json'] else "Invalid",
                                    "✅" if metrics['is_valid_json'] else "❌"
                                )
                            
                            # Show detailed metrics
                            with st.expander(DETAILED_METRICS_HEADER):
                                st.write(f"Generation ID: {metrics['generation_id']}")
                                st.write(f"Timestamp: {metrics['timestamp']}")
                                st.write(f"BERT Score: {metrics['bert_score']:.3f}")
                                st.write(f"JSON Structure Valid: {metrics['is_valid_json']}")
                                st.write(f"Passed BERT Threshold: {metrics['passed_bert']}")
                                st.write(f"{GENERATION_TIME_LABEL}: {generation_time}s")
                            
                            # Final status
                            step_container.empty()  # Clear step message
                            if metrics['passed_bert'] and metrics['is_valid_json']:
                                status_container.success(STATUS_ALL_DONE)
                            else:
                                status_container.warning(STATUS_WARNING)
                    else:
                        step_container.empty()  # Clear step message
                        status_container.error(ERROR_GENERATE_SUMMARY)
                else:
                    step_container.empty()  # Clear step message
                    status_container.error(ERROR_TRANSCRIPT_DISABLED)
            else:
                step_container.empty()  # Clear step message
                status_container.error(ERROR_INVALID_URL)
                logging.error(f"Invalid URL format: {url}")
        except Exception as e:
            step_container.empty()  # Clear step message
            status_container.error(f"❌ Error: {str(e)}")
            logging.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()