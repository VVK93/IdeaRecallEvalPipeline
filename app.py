import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
import asyncio
import time
from datetime import datetime
from bert_score import score
import json
import tiktoken
from constants import *
from typing import Tuple, Dict, Any, Optional

# Setup logging (same as before)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error(ERROR_API_KEY_MISSING)
    st.error(ERROR_API_KEY_MISSING)
    st.stop()

try:
    client = OpenAI(api_key=api_key)
    client.models.list() # Test API key early
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"OpenAI API Key validation failed: {e}")
    st.error(f"OpenAI API Key validation failed: {e}. Check key and account status.")
    st.stop()

try:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
except Exception as e:
    logging.error(f"Failed to initialize tokenizer: {e}")
    st.error(f"Failed to initialize tokenizer: {e}")
    st.stop()

# --- Utility Functions --- (Unchanged core logic)

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a string using the global tokenizer."""
    if not text: return 0
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Could not count tokens: {e}")
        return -1 # Indicate error

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logging.info(f"Extracted video ID: {video_id} from URL: {url}")
            return video_id
    logging.warning(f"Could not extract video ID from URL: {url}")
    return None

# --- Core Logic Functions --- (Unchanged core logic)

def fetch_transcript(video_id: str) -> str:
    """Fetch and concatenate transcript text from YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        logging.info(f"Fetched transcript for video {video_id}. Length: {len(full_transcript)} chars.")
        return full_transcript
    except TranscriptsDisabled:
        logging.error(f"Transcripts disabled for video {video_id}")
        raise ValueError(ERROR_TRANSCRIPT_DISABLED)
    except NoTranscriptFound:
        logging.error(f"No transcript found for video {video_id}")
        raise ValueError(ERROR_NO_TRANSCRIPT)
    except Exception as e:
        logging.error(f"Error fetching transcript for video {video_id}: {str(e)}")
        raise RuntimeError(ERROR_FETCH_TRANSCRIPT.format(str(e)))

def generate_summary(transcript: str) -> Tuple[str, float]:
    """Generate summary using OpenAI API with basic token-based chunking."""
    start_time = time.time()
    model_name = "gpt-4-turbo"
    max_model_tokens = 120000 
    prompt_overhead = 200 # Estimate

    try:
        transcript_tokens = count_tokens(transcript)
        logging.info(f"Transcript token count: {transcript_tokens}")

        if transcript_tokens <= (max_model_tokens - prompt_overhead):
            logging.info("Processing transcript in a single API call.")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                    {"role": "user", "content": USER_PROMPT_SUMMARY.format(transcript)}
                ],
                temperature=0.7, # From original code
                # Consider adding response_format={"type": "json_object"} if desired
            )
            summary = response.choices[0].message.content
        else:
            # --- Basic Chunking Logic (from original code) ---
            logging.warning("Transcript exceeds token limit, attempting basic chunking.")
            # Using character limit based on max_tokens, not ideal but matches original
            # This calculation might be flawed if transcript encoding is different
            # A better approach involves tiktoken for chunking, but sticking to original logic:
            # The original code used max_tokens directly as char limit, which is incorrect.
            # Let's *try* to approximate based on average chars/token, but it's risky.
            avg_chars_per_token = 4 # Rough estimate
            estimated_chunk_char_size = int((max_model_tokens - prompt_overhead - 1000) * avg_chars_per_token)

            chunks = [transcript[i:i + estimated_chunk_char_size] for i in range(0, len(transcript), estimated_chunk_char_size)]
            logging.info(f"Splitting transcript into {len(chunks)} chunks.")

            summaries = []
            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")
                # Check actual chunk tokens (optional but recommended)
                # chunk_tokens = count_tokens(SYSTEM_PROMPT_SUMMARY + USER_PROMPT_SUMMARY.format(chunk))
                # if chunk_tokens >= max_model_tokens: log warning/error

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                        {"role": "user", "content": USER_PROMPT_SUMMARY.format(chunk)}
                    ],
                    temperature=0.7,
                )
                summaries.append(response.choices[0].message.content)

            logging.info("Combining chunk summaries.")
            # Ensure combined summaries fit (check tokens ideally)
            combined_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMBINE},
                    {"role": "user", "content": USER_PROMPT_COMBINE.format(' \n\n---\n\n '.join(summaries))} # Join summaries
                ],
                temperature=0.7,
            )
            summary = combined_response.choices[0].message.content
            # --- End Chunking Logic ---

        generation_time = round(time.time() - start_time, 2)
        logging.info(f"Summary generated successfully in {generation_time}s.")
        return summary, generation_time

    except (Exception) as e:
        logging.error(ERROR_GENERATE_SUMMARY.format(str(e)))
        raise RuntimeError(ERROR_GENERATE_SUMMARY.format(str(e)))


# --- Evaluation Functions --- (Unchanged core logic, added type hints)

def validate_json_structure(summary: str) -> bool:
    """Validate if the input string is valid JSON."""
    # summary = summary.strip() # Keep strip if needed, but OpenAI JSON mode helps
    try:
        json.loads(summary)
        logging.info("JSON structure validation passed.")
        return True
    except json.JSONDecodeError as e:
        logging.warning(f"JSON validation failed: {str(e)}. Summary: {repr(summary[:100])}...")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during JSON validation: {str(e)}")
        return False

async def evaluate_bert_score(transcript: str, summary: str) -> float:
    """Calculate BERT score F1 between transcript and summary text."""
    try:
        P, R, F1 = await asyncio.to_thread(
            score, [summary], [transcript], lang='en', verbose=False
        )
        f1_score = float(F1.mean())
        logging.info(f"BERT score calculated: {f1_score:.4f}")
        return f1_score
    except Exception as e:
        logging.error(f"Error calculating BERT score: {str(e)}")
        return 0.0

async def evaluate_with_ai_judge(transcript: str, summary: str) -> Dict[str, Any]:
    """Evaluate summary using OpenAI as a judge. (Transcript provided for context if needed)"""
    max_retries = 2
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # Pass the raw summary string to the prompt template
            prompt = USER_PROMPT_AI_JUDGE.format(summary)

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_AI_JUDGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Keep low temp for consistency
                max_tokens=500, 
                response_format={"type": "json_object"} # Enforce JSON output
            )

            result_text = response.choices[0].message.content
            logging.info(f"AI Judge raw response (attempt {attempt+1}): {result_text}")

            try:
                result = json.loads(result_text)
                required_keys = ["overall_score", "relevance_score", "fluency_score", "critique"]
                if not all(key in result for key in required_keys):
                    raise ValueError("AI Judge response missing required keys.")
                if not all(isinstance(result.get(key, 0), int) for key in required_keys if key != "critique"):
                     # Allow potentially missing scores to default to 0 if that's acceptable
                     logging.warning("AI Judge scores missing or not integers, defaulting to 0.")
                     result = {k: result.get(k, 0) if k != "critique" else result.get(k, "Critique missing.") for k in required_keys}
                     # raise ValueError("AI Judge scores are not integers.") # Stricter check

                # Add pass/fail status (using AI_JUDGE_THRESHOLD constant)
                threshold = 80 # Match original code's hardcoded value
                result["overall_pass"] = result.get("overall_score", 0) >= threshold
                result["relevance_pass"] = result.get("relevance_score", 0) >= threshold
                result["fluency_pass"] = result.get("fluency_score", 0) >= threshold
                logging.info(f"AI Judge evaluation successful: {result}")
                return result

            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Attempt {attempt+1}: Failed to parse AI judge response or structure incorrect: {e}")
                if attempt == max_retries - 1: # Last attempt failed
                     return {
                         "error": "Failed to parse AI judge response after retries.",
                         "raw_response": result_text, # Include raw response for debugging
                         "overall_score": 0, "relevance_score": 0, "fluency_score": 0,
                         "critique": f"Error: Could not parse judge response. Raw: {result_text}",
                         "overall_pass": False, "relevance_pass": False, "fluency_pass": False
                     }
                await asyncio.sleep(retry_delay) # Wait before retrying

        except (Exception) as e:
            logging.exception(f"Unexpected error in AI judge evaluation (attempt {attempt+1})") # Use exception for stack trace
            if attempt == max_retries - 1:
                return {
                    "error": f"Unexpected error: {str(e)}",
                    "overall_score": 0, "relevance_score": 0, "fluency_score": 0,
                    "critique": f"Error: {str(e)}",
                    "overall_pass": False, "relevance_pass": False, "fluency_pass": False
                }
            await asyncio.sleep(retry_delay)

    # Fallback if loop finishes unexpectedly (shouldn't happen with returns inside)
    return {"error": "AI Judge evaluation failed unexpectedly after retries.", "overall_pass": False, "relevance_pass": False, "fluency_pass": False}


# --- UI Display Function ---

def display_final_results(
    transcript: str,
    summary: str,
    metrics: Dict[str, Any],
    generation_time: float
    ) -> None:
    """Displays the final results in a structured layout."""

    st.header(RESULTS_HEADER)
    st.divider()

    # --- Transcript and Summary Side-by-Side ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(TRANSCRIPT_HEADER)
        tran_len = len(transcript)
        tran_tok = count_tokens(transcript)
        st.caption(f"{LENGTH_LABEL.format(tran_len)} | {TOKENS_LABEL.format(tran_tok) if tran_tok != -1 else 'Token count error'}")
        with st.expander(VIEW_TRANSCRIPT_EXPANDER, expanded=False):
            st.text_area("Transcript_Text", transcript, height=300, key="transcript_display", disabled=True)

    with col2:
        st.subheader(SUMMARY_HEADER)
        sum_len = len(summary)
        sum_tok = count_tokens(summary)
        st.caption(f"{LENGTH_LABEL.format(sum_len)} | {TOKENS_LABEL.format(sum_tok) if sum_tok != -1 else 'Token count error'}")
        with st.expander(VIEW_SUMMARY_EXPANDER, expanded=True): # Show summary expanded by default
            is_json = metrics.get("is_valid_json", False)
            if is_json:
                try:
                    st.json(json.loads(summary), expanded=True)
                except Exception as e:
                    logging.warning(f"Failed to render summary as JSON even though validated: {e}")
                    st.text_area("Summary_JSON_Error", summary, height=300, key="summary_display_json_error", disabled=True)
            else:
                 st.warning(f"{INFO_ICON} Summary is not valid JSON.")
                 st.text_area("Summary_Text", summary, height=300, key="summary_display_text", disabled=True)

    st.divider()

    # --- Evaluation Metrics ---
    st.subheader(EVALUATION_HEADER)

    # Generation Time Metric
    st.metric(GENERATION_TIME_LABEL, f"{generation_time:.2f} s")
    st.divider()

    # Automatic Checks Section
    st.markdown(f"**{METRICS_AUTO_HEADER}**")
    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        json_valid = metrics.get("is_valid_json", False)
        json_status = f"{PASS_ICON} {JSON_VALID_LABEL}" if json_valid else f"{FAIL_ICON} {JSON_INVALID_LABEL}"
        st.metric(JSON_VALIDATION_LABEL, json_status)
    with col_auto2:
        bert_score = metrics.get("bert_score", 0.0)
        bert_threshold = 0.8 # From original conditional logic
        passed_bert = bert_score >= bert_threshold
        bert_pass_fail_icon = f"{PASS_ICON}" if passed_bert else f"{FAIL_ICON}"
        # Use BERT_THRESHOLD_NOTE which includes the value
        st.metric(BERT_LABEL, f"{bert_score:.3f}", f"{bert_pass_fail_icon} {BERT_THRESHOLD_NOTE}")
    st.divider()

    # AI Judge Section
    st.markdown(f"**{METRICS_AI_HEADER}**")
    ai_results = metrics.get("ai_judge")

    if ai_results:
        if "error" in ai_results:
            st.error(f"{FAIL_ICON} AI Judge evaluation failed: {ai_results['error']}")
            if "raw_response" in ai_results:
                 with st.expander("View Raw AI Judge Error Response"):
                     st.code(ai_results["raw_response"], language=None)
        else:
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            ai_threshold = 80 # From original code
            with col_ai1:
                ov_score = ai_results.get('overall_score', 0)
                ov_pass_icon = f"{PASS_ICON}" if ov_score >= ai_threshold else f"{WARN_ICON}"
                st.metric(AI_OVERALL_LABEL, f"{ov_score}{SCORE_SUFFIX}", ov_pass_icon)
            with col_ai2:
                rel_score = ai_results.get('relevance_score', 0)
                rel_pass_icon = f"{PASS_ICON}" if rel_score >= ai_threshold else f"{WARN_ICON}"
                st.metric(AI_RELEVANCE_LABEL, f"{rel_score}{SCORE_SUFFIX}", rel_pass_icon)
            with col_ai3:
                flu_score = ai_results.get('fluency_score', 0)
                flu_pass_icon = f"{PASS_ICON}" if flu_score >= ai_threshold else f"{WARN_ICON}"
                st.metric(AI_FLUENCY_LABEL, f"{flu_score}{SCORE_SUFFIX}", flu_pass_icon)

            critique = ai_results.get("critique", "No critique provided.")
            with st.expander(VIEW_CRITIQUE_EXPANDER):
                st.markdown(critique)

            # --- Overall Status Indicator --- (Based on all checks)
            all_auto_passed = json_valid and passed_bert
            # Check if AI judge passed all its metrics (if it ran successfully)
            ai_judge_passed_all = (
                "error" not in ai_results and
                ai_results.get('overall_pass', False) and
                ai_results.get('relevance_pass', False) and
                ai_results.get('fluency_pass', False)
            )

            if all_auto_passed and ai_judge_passed_all:
                st.success(f"{PASS_ICON} {SUCCESS_MESSAGE}")
            else:
                st.warning(f"{WARN_ICON} {WARNING_MESSAGE}")

    # Handle case where AI judge was skipped
    elif metrics.get("ai_judge") is None: # Check if explicitly None (skipped)
         st.info(f"{INFO_ICON} {INFO_AI_JUDGE_SKIPPED}")
         # Still show overall status based on auto checks
         if json_valid and passed_bert:
              st.success(f"{PASS_ICON} Automatic checks passed, AI Judge skipped.")
         else:
              st.warning(f"{WARN_ICON} Automatic checks failed.")
    else:
         # Should not happen if ai_judge is always set to None or a dict
         st.info(INFO_AI_JUDGE_NO_RESULTS)

    st.divider()
    # Detailed Metrics Expander (Raw Data)
    with st.expander(VIEW_DETAILS_EXPANDER):
        st.json(metrics)


# --- Main Streamlit App Function ---

def main():
    st.set_page_config(page_title="Idea Recall", page_icon="🎥", layout="wide")

    st.title(APP_TITLE)
    st.caption(APP_SUBHEADER)

    url = st.text_input(URL_INPUT_LABEL, placeholder=URL_PLACEHOLDER, key="youtube_url_input")

    # Only proceed if URL is entered
    if url:
        video_id = None
        transcript = ""
        summary = ""
        final_metrics = {}
        generation_time = 0.0
        processing_successful = False

        # Use st.status for better feedback during the multi-step process
        with st.status(STATUS_IN_PROGRESS, expanded=True) as status:
            try:
                # --- Step 1: Extract ID & Fetch Transcript ---
                status.update(label=STATUS_STEP_1)
                start_time = time.time()
                video_id = extract_video_id(url)
                if not video_id:
                    raise ValueError(ERROR_INVALID_URL)

                transcript = fetch_transcript(video_id) # Raises ValueError or RuntimeError on failure
                step_1_time = time.time() - start_time
                tran_len = len(transcript)
                tran_tok = count_tokens(transcript)
                status.update(label=STATUS_STEP_1_COMPLETE.format(tran_len, tran_tok))
                logging.info(f"Step 1 took {step_1_time:.2f}s")


                # --- Step 2: Generate Summary ---
                status.update(label=STATUS_STEP_2)
                # generation_time is captured inside generate_summary
                summary, generation_time = generate_summary(transcript) # Raises RuntimeError on failure
                sum_len = len(summary)
                sum_tok = count_tokens(summary)
                status.update(label=STATUS_STEP_2_COMPLETE.format(generation_time, sum_len, sum_tok))
                logging.info(f"Step 2 took {generation_time:.2f}s")


                # --- Step 3: Automatic Evaluation (Using Evaluator) ---
                status.update(label=STATUS_STEP_3)
                start_time = time.time()
                # Run JSON validation and BERT score evaluation separately
                json_valid = validate_json_structure(summary)
                bert_score = asyncio.run(evaluate_bert_score(transcript, summary))
                
                # Combine results into expected dictionary structure
                auto_eval_results = {
                    "json_valid": json_valid,
                    "bert_score": bert_score
                }
                
                if not isinstance(auto_eval_results, dict) or 'json_valid' not in auto_eval_results or 'bert_score' not in auto_eval_results:
                    logging.error(f"Evaluator output format incorrect: {auto_eval_results}")
                    raise TypeError("Evaluator did not return the expected dictionary structure ('json_valid', 'bert_score').")
                step_3_time = time.time() - start_time
                status.update(label=STATUS_STEP_3_COMPLETE)
                logging.info(f"Step 3 took {step_3_time:.2f}s")


                # --- Step 4: AI Judge Evaluation (Conditional) ---
                ai_judge_results = None
                bert_threshold = 0.8 # From original logic
                run_ai_judge = auto_eval_results.get("json_valid", False) and auto_eval_results.get("bert_score", 0.0) >= bert_threshold

                if run_ai_judge:
                    status.update(label=STATUS_STEP_4)
                    start_time = time.time()
                    # Run the async function using asyncio.run()
                    ai_judge_results = asyncio.run(evaluate_with_ai_judge(transcript, summary))
                    step_4_time = time.time() - start_time
                    status.update(label=STATUS_STEP_4_COMPLETE)
                    logging.info(f"Step 4 took {step_4_time:.2f}s")
                else:
                    status.update(label=STATUS_STEP_4_SKIPPED)
                    logging.info("AI Judge skipped due to auto-eval results.")


                # --- Combine Results ---
                final_metrics = {
                    "is_valid_json": auto_eval_results.get("json_valid"),
                    "bert_score": auto_eval_results.get("bert_score"),
                    # Adding passed_bert here for clarity in final dict, based on threshold used
                    "passed_bert": auto_eval_results.get("bert_score", 0.0) >= bert_threshold,
                    "generation_id": video_id,
                    "timestamp": datetime.now().isoformat(), # Use current timestamp
                    "ai_judge": ai_judge_results # Will be None if skipped or dict if ran
                }

                status.update(label=STATUS_ALL_COMPLETE, state="complete", expanded=False) # Collapse status on success
                processing_successful = True

            except (ValueError, RuntimeError, TypeError, Exception) as e:
                # Catch specific errors from functions and general errors
                error_message = str(e)
                status.update(label=STATUS_ERROR.format(error_message), state="error", expanded=True)
                logging.exception("Error during processing pipeline.") # Log full traceback
                # Display error prominently outside the status box as well
                st.error(f"{FAIL_ICON} Processing failed: {error_message}")

        # --- Display Final Results --- (Only if processing completed successfully)
        if processing_successful:
            display_final_results(transcript, summary, final_metrics, generation_time)

    # Optionally add a placeholder or message if no URL is entered
    # else:
    #    st.info("Enter a YouTube URL above to begin.")


if __name__ == "__main__":
    main()