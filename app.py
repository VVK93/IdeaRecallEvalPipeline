import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI, OpenAIError # Import specific error type
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
from constants import * # Import all constants
from typing import Tuple, Dict, Any, Optional, List

# --- Setup Logging ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# Load .env file relative to this script file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- Initialize OpenAI Client and Tokenizer ---
# API key is now loaded via constants.py which gets it from os.getenv
if not API_KEY:
    logger.critical(ERROR_API_KEY_MISSING)
    st.error(ERROR_API_KEY_MISSING)
    st.stop()

try:
    client = OpenAI(api_key=API_KEY)
    # Quick check to validate key early
    client.models.list()
    logger.info("OpenAI client initialized successfully.")
except OpenAIError as e:
    logger.critical(ERROR_OPENAI_INIT.format(e), exc_info=True)
    st.error(ERROR_OPENAI_INIT.format(e))
    st.stop()
except Exception as e: # Catch potential network or other issues
    logger.critical(f"Unexpected error initializing OpenAI client: {e}", exc_info=True)
    st.error(f"Unexpected error initializing OpenAI client: {e}")
    st.stop()

try:
    # Use model name specified in constants for tokenizer
    tokenizer = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    logger.info(f"Tokenizer for model '{TOKENIZER_MODEL}' initialized.")
except Exception as e:
    logger.critical(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, e), exc_info=True)
    st.error(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, e))
    st.stop()

# --- Utility Functions ---

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a string using the global tokenizer."""
    if not text: return 0
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(ERROR_TOKEN_COUNT.format(e), exc_info=True)
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
            logger.info(f"Extracted video ID: {video_id} from URL: {url}")
            return video_id
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

# --- Core Logic Functions ---

def fetch_transcript(video_id: str) -> str:
    """Fetch and concatenate transcript text from YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        logger.info(f"Fetched transcript for video {video_id}. Length: {len(full_transcript)} chars.")
        return full_transcript
    except TranscriptsDisabled:
        logger.error(f"Transcripts disabled for video {video_id}")
        raise ValueError(ERROR_TRANSCRIPT_DISABLED)
    except NoTranscriptFound:
        logger.error(f"No transcript found for video {video_id}")
        raise ValueError(ERROR_NO_TRANSCRIPT)
    except Exception as e:
        logger.error(f"Error fetching transcript for video {video_id}: {e}", exc_info=True)
        raise RuntimeError(ERROR_FETCH_TRANSCRIPT.format(e))

def _generate_summary_single_call(transcript: str) -> str:
    """Helper for single API call summary generation."""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            {"role": "user", "content": USER_PROMPT_SUMMARY.format(transcript)}
        ],
        temperature=0.5, # Slightly lower temp for consistency
        response_format={"type": "json_object"} # Enforce JSON output
    )
    return response.choices[0].message.content

def _generate_summary_map_reduce(transcript: str, status_update_func) -> str:
    """Helper for Map-Reduce summary generation using token chunking."""
    # 1. Chunking based on tokens
    tokens = tokenizer.encode(transcript)
    chunks = []
    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start_index:end_index]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        # Move start index for next chunk, considering overlap
        start_index += CHUNK_SIZE - CHUNK_OVERLAP
        if start_index >= len(tokens): # Prevent infinite loop if overlap is large
             break

    logger.info(f"Split transcript into {len(chunks)} chunks for Map-Reduce.")

    # 2. Map Step: Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        status_update_func(label=STATUS_STEP_2_MAP.format(i=i+1, n=len(chunks)))
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            chunk_summary_json = _generate_summary_single_call(chunk)
            # Extract summary text from JSON - crucial step
            try:
                chunk_summary_data = json.loads(chunk_summary_json)
                summary_text = chunk_summary_data.get("summary", "")
                if not summary_text:
                    logger.warning(f"Chunk {i+1} summary JSON missing 'summary' key or value empty.")
                    # Optionally: Fallback or skip this chunk's summary
                    continue # Skip this chunk if summary is empty/missing
                chunk_summaries.append(summary_text)
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON from chunk {i+1} summary: {chunk_summary_json[:100]}...")
                 # Optionally: raise error or skip
                 continue # Skip if JSON is invalid
        except Exception as e:
            logger.error(ERROR_GENERATE_SUMMARY_CHUNK.format(i=i+1, n=len(chunks), e=e), exc_info=True)
            # Decide: raise error immediately or try to continue and combine partial results?
            # Raising error is safer for ensuring full coverage potential.
            raise RuntimeError(ERROR_GENERATE_SUMMARY_CHUNK.format(i=i+1, n=len(chunks), e=e))

    if not chunk_summaries:
        raise RuntimeError("Map phase failed: No valid chunk summaries were generated.")

    # 3. Reduce Step: Combine chunk summaries
    status_update_func(label=STATUS_STEP_2_REDUCE)
    logger.info("Combining chunk summaries in Reduce step.")
    combined_text = "\n\n---\n\n".join(chunk_summaries) # Simple joiner

    # Check token count of combined summaries before final call (optional but good practice)
    combined_tokens = count_tokens(SYSTEM_PROMPT_COMBINE + USER_PROMPT_COMBINE.format(combined_text))
    if combined_tokens > MAX_MODEL_TOKENS_SUMMARY: # Use same limit as generation
        logger.warning(f"Combined chunk summaries ({combined_tokens} tokens) exceed model limit. Combining might fail or truncate.")
        # Potentially implement recursive reduction if needed, but keep simple for now

    try:
        final_summary_json = client.chat.completions.create(
            model=GENERATION_MODEL, # Use the main generation model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COMBINE},
                {"role": "user", "content": USER_PROMPT_COMBINE.format(combined_text)}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        return final_summary_json
    except Exception as e:
        logger.error(ERROR_GENERATE_SUMMARY_COMBINE.format(e=e), exc_info=True)
        raise RuntimeError(ERROR_GENERATE_SUMMARY_COMBINE.format(e=e))


def generate_summary(transcript: str, status_update_func) -> Tuple[str, float]:
    """
    Generate summary using OpenAI API. Uses Map-Reduce with token chunking if transcript is too long.
    Args:
        transcript: The video transcript text.
        status_update_func: A function (like st.status.update) to report progress.
    Returns:
        Tuple containing the summary (as a JSON string) and generation time.
    """
    start_time = time.time()
    try:
        # Calculate tokens needed for a single call
        # Add estimate for system/user prompt wrapper around the transcript
        prompt_tokens = count_tokens(SYSTEM_PROMPT_SUMMARY + USER_PROMPT_SUMMARY.format(""))
        transcript_tokens = count_tokens(transcript)
        total_tokens_single_call = prompt_tokens + transcript_tokens

        logger.info(f"Transcript token count: {transcript_tokens}")
        logger.info(f"Estimated single call tokens: {total_tokens_single_call} vs Max: {MAX_MODEL_TOKENS_SUMMARY}")

        if total_tokens_single_call <= MAX_MODEL_TOKENS_SUMMARY:
            # --- Single Call ---
            status_update_func(label=STATUS_STEP_2_SINGLE)
            logger.info("Processing transcript in a single API call.")
            summary_json = _generate_summary_single_call(transcript)
        else:
            # --- Map-Reduce Call ---
            logger.warning("Transcript exceeds token limit, initiating Map-Reduce process.")
            num_chunks = (transcript_tokens // (CHUNK_SIZE - CHUNK_OVERLAP)) + 1 # Estimate
            status_update_func(label=STATUS_STEP_2_CHUNK.format(num_chunks=num_chunks))
            summary_json = _generate_summary_map_reduce(transcript, status_update_func)

        generation_time = round(time.time() - start_time, 2)
        logger.info(f"Summary generation completed in {generation_time}s.")

        # Basic validation that we got *some* string back
        if not summary_json or not isinstance(summary_json, str):
             raise ValueError("LLM returned empty or invalid summary content.")

        return summary_json, generation_time

    except (OpenAIError, ValueError, RuntimeError) as e:
        logger.error(ERROR_GENERATE_SUMMARY_GENERAL.format(e), exc_info=True)
        raise RuntimeError(ERROR_GENERATE_SUMMARY_GENERAL.format(e)) # Re-raise as runtime error
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during summary generation: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected summary generation error: {e}")


# --- Evaluation Functions ---

def validate_json_structure(summary_json: str) -> bool:
    """Validate if the input string is valid JSON."""
    try:
        json.loads(summary_json)
        logger.info("JSON structure validation passed.")
        return True
    except json.JSONDecodeError as e:
        logger.warning(f"JSON validation failed: {e}. Summary start: {repr(summary_json[:100])}...")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during JSON validation: {e}", exc_info=True)
        return False

async def evaluate_bert_score(transcript: str, summary_text: str) -> float:
    """Calculate BERT score F1 between transcript and extracted summary text."""
    if not summary_text: # Handle case where summary text extraction failed
        logger.warning("BERT score evaluation skipped: summary text is empty.")
        return 0.0
    try:
        # Run blocking BERT score calculation in a separate thread
        P, R, F1 = await asyncio.to_thread(
            score, [summary_text], [transcript], lang='en', verbose=False, model_type='bert-base-uncased' # Specify model for consistency
        )
        f1_score = float(F1.mean())
        logger.info(f"BERT score calculated: {f1_score:.4f}")
        return f1_score
    except Exception as e:
        logger.error(ERROR_EVALUATE_BERT.format(e), exc_info=True)
        return 0.0 # Return 0 on error

async def evaluate_with_ai_judge(summary_text: str) -> Dict[str, Any]:
    """Evaluate summary using OpenAI as a judge."""
    if not summary_text:
        logger.warning("AI Judge evaluation skipped: summary text is empty.")
        return {"error": "Summary text was empty.", "overall_pass": False} # Indicate skip reason

    max_retries = 2
    retry_delay = 5
    final_result = {}

    for attempt in range(max_retries):
        try:
            # Pass the extracted summary text to the judge prompt
            prompt = USER_PROMPT_AI_JUDGE.format(summary_text)

            # Run blocking API call in thread
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=AI_JUDGE_MODEL, # Use judge model from constants
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_AI_JUDGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Low temp for deterministic judging
                max_tokens=500,
                response_format={"type": "json_object"} # Enforce JSON output
            )

            result_text = response.choices[0].message.content
            logger.info(f"AI Judge raw response (attempt {attempt+1}): {result_text}")

            # Parse and validate the JSON response
            try:
                result = json.loads(result_text)
                required_keys = ["overall_score", "relevance_score", "fluency_score", "critique"]

                # Use .get() for safer access and check types
                scores = {k: result.get(k) for k in required_keys if k != "critique"}
                critique = result.get("critique", "Critique missing from response.") # Default critique

                if not all(isinstance(scores.get(k), int) for k in scores):
                    # Raise error if scores are missing or not integers
                    raise ValueError("AI Judge response missing scores or scores are not integers.")

                # Add pass/fail status based on threshold from constants
                threshold = AI_JUDGE_THRESHOLD
                final_result = {
                    "overall_score": scores.get("overall_score", 0),
                    "relevance_score": scores.get("relevance_score", 0),
                    "fluency_score": scores.get("fluency_score", 0),
                    "critique": critique,
                    "overall_pass": scores.get("overall_score", 0) >= threshold,
                    "relevance_pass": scores.get("relevance_score", 0) >= threshold,
                    "fluency_pass": scores.get("fluency_score", 0) >= threshold,
                    "error": None # Explicitly set error to None on success
                }
                logger.info(f"AI Judge evaluation successful: {final_result}")
                return final_result # Success, exit loop and function

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(ERROR_AI_JUDGE_PARSE.format(attempt=attempt+1, max_retries=max_retries, e=e))
                if attempt == max_retries - 1: # Last attempt failed
                     final_result = {
                         "error": ERROR_AI_JUDGE_RETRY_FAIL.format(max_retries=max_retries),
                         "raw_response": result_text,
                         "overall_score": 0, "relevance_score": 0, "fluency_score": 0,
                         "critique": f"Error parsing judge response after {max_retries} attempts.",
                         "overall_pass": False, "relevance_pass": False, "fluency_pass": False
                     }
                     return final_result
                await asyncio.sleep(retry_delay) # Wait before retrying

        except OpenAIError as e:
            logger.error(f"OpenAI API error during AI judge evaluation (attempt {attempt+1}): {e}", exc_info=True)
            if attempt == max_retries - 1:
                 final_result = {"error": ERROR_EVALUATE_AI_JUDGE.format(e), "overall_pass": False}
                 return final_result
            await asyncio.sleep(retry_delay)
        except Exception as e: # Catch any other unexpected errors
            logger.exception(f"Unexpected error in AI judge evaluation (attempt {attempt+1})")
            if attempt == max_retries - 1:
                final_result = {"error": f"Unexpected error: {str(e)}", "overall_pass": False}
                return final_result
            await asyncio.sleep(retry_delay)

    # Fallback if loop finishes unexpectedly (should have returned earlier)
    if not final_result:
         final_result = {"error": "AI Judge evaluation failed unexpectedly.", "overall_pass": False}
    return final_result


# --- UI Display Function ---

def display_final_results(
    transcript: str,
    summary_json: str, # Now explicitly the JSON string
    metrics: Dict[str, Any],
    generation_time: float
    ) -> None:
    """Displays the final results in a structured layout."""

    st.header(RESULTS_HEADER)
    st.divider()

    # --- Attempt to extract summary text for display ---
    summary_text = ""
    summary_data = {}
    is_valid_json_for_display = False
    try:
        summary_data = json.loads(summary_json)
        summary_text = summary_data.get("summary", "*Summary text not found in JSON*")
        is_valid_json_for_display = True
    except json.JSONDecodeError:
        summary_text = summary_json # Display raw output if not valid JSON
        is_valid_json_for_display = False
    except Exception: # Catch any other parsing issues
         summary_text = summary_json
         is_valid_json_for_display = False


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
        sum_len = len(summary_text) # Length of extracted text
        sum_tok = count_tokens(summary_text) # Tokens of extracted text
        st.caption(f"{LENGTH_LABEL.format(sum_len)} | {TOKENS_LABEL.format(sum_tok) if sum_tok != -1 else 'Token count error'}")

        if is_valid_json_for_display:
            with st.expander(VIEW_SUMMARY_EXPANDER, expanded=True):
                st.json(summary_data, expanded=True) # Display the parsed JSON object
        else:
             st.warning(f"{INFO_ICON} Summary output was not valid JSON. Displaying raw output.")
             st.text_area("Summary_Raw_Output", summary_text, height=300, key="summary_display_raw", disabled=True)


    st.divider()

    # --- Evaluation Metrics ---
    st.subheader(EVALUATION_HEADER)

    st.metric(GENERATION_TIME_LABEL, f"{generation_time:.2f} s")
    st.divider()

    # Automatic Checks Section
    st.markdown(f"**{METRICS_AUTO_HEADER}**")
    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        # Use the definitive validation result stored in metrics
        json_valid = metrics.get("is_valid_json", False)
        json_status = f"{PASS_ICON} {JSON_VALID_LABEL}" if json_valid else f"{FAIL_ICON} {JSON_INVALID_LABEL}"
        st.metric(JSON_VALIDATION_LABEL, json_status)
    with col_auto2:
        bert_score_val = metrics.get("bert_score", 0.0)
        # Use constant for threshold check
        passed_bert = bert_score_val >= BERT_SCORE_THRESHOLD
        bert_pass_fail_icon = f"{PASS_ICON}" if passed_bert else f"{FAIL_ICON}"
        # Use constant for threshold note display
        st.metric(BERT_LABEL, f"{bert_score_val:.3f}", f"{bert_pass_fail_icon} {BERT_THRESHOLD_NOTE}")
    st.divider()

    # AI Judge Section
    st.markdown(f"**{METRICS_AI_HEADER}**") # Include model name via constant formatting
    ai_results = metrics.get("ai_judge")

    if ai_results:
        # Check if evaluation failed vs. just didn't pass thresholds
        if ai_results.get("error"):
            st.error(f"{FAIL_ICON} AI Judge evaluation failed: {ai_results['error']}")
            if "raw_response" in ai_results:
                 with st.expander("View Raw AI Judge Error Response"):
                     st.code(ai_results["raw_response"], language="json") # Assume JSON-like if available
        else:
            # Display scores if evaluation succeeded
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            threshold = AI_JUDGE_THRESHOLD # Use constant
            with col_ai1:
                ov_score = ai_results.get('overall_score', 0)
                ov_pass_icon = f"{PASS_ICON}" if ov_score >= threshold else f"{WARN_ICON}"
                st.metric(AI_OVERALL_LABEL, f"{ov_score}{SCORE_SUFFIX}", ov_pass_icon)
            with col_ai2:
                rel_score = ai_results.get('relevance_score', 0)
                rel_pass_icon = f"{PASS_ICON}" if rel_score >= threshold else f"{WARN_ICON}"
                st.metric(AI_RELEVANCE_LABEL, f"{rel_score}{SCORE_SUFFIX}", rel_pass_icon)
            with col_ai3:
                flu_score = ai_results.get('fluency_score', 0)
                flu_pass_icon = f"{PASS_ICON}" if flu_score >= threshold else f"{WARN_ICON}"
                st.metric(AI_FLUENCY_LABEL, f"{flu_score}{SCORE_SUFFIX}", flu_pass_icon)

            critique = ai_results.get("critique", "No critique provided.")
            with st.expander(VIEW_CRITIQUE_EXPANDER):
                st.markdown(critique)

            # Check if all *individual* AI metrics passed (if evaluation ran ok)
            ai_judge_passed_all_metrics = (
                ai_results.get('overall_pass', False) and
                ai_results.get('relevance_pass', False) and
                ai_results.get('fluency_pass', False)
            )
            # Overall Status Indicator logic depends on whether AI judge ran and passed
            auto_checks_passed = json_valid and passed_bert

            if auto_checks_passed and ai_judge_passed_all_metrics:
                 st.success(f"{PASS_ICON} {SUCCESS_MESSAGE}")
            elif auto_checks_passed and not ai_judge_passed_all_metrics:
                 st.warning(f"{WARN_ICON} Automatic checks passed, but AI Judge metrics below threshold. Review critique.")
            else: # Automatic checks failed (AI judge might have been skipped or also failed)
                 st.warning(f"{WARN_ICON} {WARNING_MESSAGE}")


    # Handle case where AI judge was explicitly skipped (metrics["ai_judge"] is None)
    elif metrics.get("ai_judge") is None:
         st.info(f"{INFO_ICON} {INFO_AI_JUDGE_SKIPPED}")
         # Show overall status based only on auto checks
         if json_valid and passed_bert:
              st.success(f"{PASS_ICON} Automatic checks passed. AI Judge skipped.")
         else:
              st.warning(f"{WARN_ICON} Automatic checks failed.")
    else:
         # Should not happen if ai_judge is always None or a dict
         st.error("Internal state error: Unexpected value for AI judge results.")

    st.divider()
    # Detailed Metrics Expander (Raw Data)
    with st.expander(VIEW_DETAILS_EXPANDER):
        # Display the structured final_metrics dictionary
        st.json(metrics)


# --- Main Async Pipeline Function ---

async def run_pipeline(url: str, status) -> Optional[Tuple[str, str, Dict, float]]:
    """
    Runs the full pipeline: fetch, summarize, evaluate.
    Args:
        url: The YouTube URL.
        status: The st.status object for updates.
    Returns:
        A tuple (transcript, summary_json, final_metrics, generation_time) on success, None on failure.
    """
    transcript = ""
    summary_json = ""
    generation_time = 0.0
    final_metrics = {}
    video_id = None

    try:
        # --- Step 1: Extract ID & Fetch Transcript ---
        status.update(label=STATUS_STEP_1)
        start_time = time.time()
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError(ERROR_INVALID_URL)

        transcript = await asyncio.to_thread(fetch_transcript, video_id)
        step_1_time = time.time() - start_time
        tran_len = len(transcript)
        tran_tok = count_tokens(transcript)
        status.update(label=STATUS_STEP_1_COMPLETE.format(tran_len=tran_len, tran_tok=tran_tok))
        logger.info(f"Step 1 (Fetch Transcript) took {step_1_time:.2f}s")

        # --- Step 2: Generate Summary ---
        # Pass the status update function to generate_summary for Map-Reduce progress
        summary_json, generation_time = await asyncio.to_thread(
            generate_summary, transcript, status.update
        )
        sum_len_estimate = len(summary_json) # Estimate length based on JSON string
        sum_tok_estimate = count_tokens(summary_json) # Estimate tokens based on JSON string
        status.update(label=STATUS_STEP_2_COMPLETE.format(gen_time=generation_time, sum_len=sum_len_estimate, sum_tok=sum_tok_estimate))
        logger.info(f"Step 2 (Generate Summary) took {generation_time:.2f}s")

        # --- Step 3: Automatic Evaluation (JSON Validation & BERT Score) ---
        status.update(label=STATUS_STEP_3)
        start_time = time.time()

        # 3a: Validate JSON Structure of the summary
        is_valid_json = validate_json_structure(summary_json)

        # 3b: Extract summary text IF JSON is valid (needed for BERT & AI Judge)
        summary_text_for_eval = ""
        if is_valid_json:
            try:
                summary_data = json.loads(summary_json)
                summary_text_for_eval = summary_data.get("summary", "")
                if not summary_text_for_eval:
                    logger.warning("JSON is valid, but 'summary' key is missing or empty.")
            except Exception as e:
                 logger.error(f"Error extracting summary text from valid JSON: {e}")
                 is_valid_json = False # Treat as invalid if extraction fails

        # 3c: Calculate BERT Score (only if summary text exists)
        # Run BERT score calculation asynchronously
        bert_score_val = await evaluate_bert_score(transcript, summary_text_for_eval)

        step_3_time = time.time() - start_time
        status.update(label=STATUS_STEP_3_COMPLETE.format(bert_score=bert_score_val))
        logger.info(f"Step 3 (Auto Eval) took {step_3_time:.2f}s")


        # --- Step 4: AI Judge Evaluation (Conditional) ---
        # Condition: Run AI Judge only if JSON is valid AND BERT score meets threshold
        ai_judge_results = None # Default to None (skipped)
        run_ai_judge_condition = is_valid_json and bert_score_val >= BERT_SCORE_THRESHOLD

        if run_ai_judge_condition:
            status.update(label=STATUS_STEP_4)
            logger.info("Conditions met, running AI Judge evaluation.")
            start_time = time.time()
            # Pass the extracted summary text
            ai_judge_results = await evaluate_with_ai_judge(summary_text_for_eval)
            step_4_time = time.time() - start_time
            status.update(label=STATUS_STEP_4_COMPLETE)
            logger.info(f"Step 4 (AI Judge) took {step_4_time:.2f}s")
        else:
            status.update(label=STATUS_STEP_4_SKIPPED)
            logger.info(f"AI Judge skipped. JSON valid: {is_valid_json}, BERT Score: {bert_score_val:.3f} (Threshold: {BERT_SCORE_THRESHOLD})")


        # --- Combine Final Metrics ---
        final_metrics = {
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "generation_model": GENERATION_MODEL,
            "generation_time_sec": generation_time,
            "transcript_chars": tran_len,
            "transcript_tokens": tran_tok,
            "summary_json_valid": is_valid_json,
            "summary_bert_score": bert_score_val,
            "summary_bert_passed": bert_score_val >= BERT_SCORE_THRESHOLD,
            "ai_judge_model": AI_JUDGE_MODEL,
            "ai_judge_skipped": not run_ai_judge_condition,
            "ai_judge_results": ai_judge_results # Contains scores, critique, pass status, or error info
        }

        return transcript, summary_json, final_metrics, generation_time

    except (ValueError, RuntimeError, OpenAIError) as e:
        error_message = str(e)
        logger.exception(f"Pipeline error for URL {url}: {error_message}")
        status.update(label=STATUS_ERROR.format(error_message), state="error", expanded=True)
        st.error(f"{FAIL_ICON} Processing failed: {error_message}")
        return None
    except Exception as e: # Catch-all for unexpected errors
        error_message = ERROR_UNKNOWN.format(e)
        logger.exception(f"Unexpected pipeline error for URL {url}")
        status.update(label=STATUS_ERROR.format(error_message), state="error", expanded=True)
        st.error(f"{FAIL_ICON} An unexpected error occurred: {e}")
        return None


# --- Main Streamlit App Function ---

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    st.title(APP_TITLE)
    st.caption(APP_SUBHEADER)

    # Display the informational box about the demo's scope
    st.info(DEMO_SCOPE_INFO, icon=INFO_ICON)

    url = st.text_input(URL_INPUT_LABEL, placeholder=URL_PLACEHOLDER, key="youtube_url_input")

    if url:
        results = None
        # Use st.status for the entire async pipeline execution
        with st.status(STATUS_IN_PROGRESS, expanded=True) as status:
            # Run the main async pipeline function
            results = asyncio.run(run_pipeline(url, status))

        # If run_pipeline completed successfully (didn't return None)
        if results:
            transcript, summary_json, final_metrics, generation_time = results
            status.update(label=STATUS_ALL_COMPLETE, state="complete", expanded=False)
            # Display results outside the status box
            display_final_results(transcript, summary_json, final_metrics, generation_time)
        # Error handling is done inside run_pipeline and reported via status/st.error


if __name__ == "__main__":
    main()