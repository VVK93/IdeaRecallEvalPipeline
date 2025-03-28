# pipeline.py
import logging
import asyncio
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from openai import OpenAIError

# Import constants and shared logger
from constants import *
from utils import logger as root_logger, extract_video_id, fetch_transcript, clean_transcript, count_tokens 

# Import specific functions from other modules
from generation import generate_content
from evaluation import validate_json_structure, _run_async_evaluations

# Get a logger specific to this module
logger = logging.getLogger(__name__)

def run_pipeline(url: str, status_update_func) -> Optional[Tuple[str, str, Dict, float]]:
    """
    Runs the full pipeline synchronously, orchestrating calls to other modules.
    Uses asyncio.run() only for the concurrent evaluation steps.

    Args:
        url (str): The YouTube URL input by the user.
        status_update_func: A function (like st.status.update) to report progress.

    Returns:
        Optional[Tuple[str, str, Dict, float]]: A tuple containing:
            - transcript_raw (str): The original fetched transcript.
            - content_json (str): The raw JSON string output from the LLM.
            - final_metrics (Dict): A dictionary containing all collected metrics.
            - generation_time (float): The time taken for content generation.
        Returns None if a critical error occurs during the pipeline execution.
    """
    # Initialize variables
    transcript_raw = ""
    transcript_cleaned = ""
    content_json = ""
    generation_time = 0.0
    final_metrics: Dict[str, Any] = {}
    video_id = None
    parsed_content_data: Optional[Dict] = None
    is_valid_json = False
    summary_text_for_eval = ""
    bert_score_val = 0.0
    ai_judge_results: Optional[Dict] = None
    ai_judge_skipped = True
    passed_bert_indicator = False

    try:
        # --- Step 1: Fetch & Clean Transcript ---
        if status_update_func: status_update_func(label=STATUS_STEP_1)
        step_1_start_time = time.time()
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError(ERROR_INVALID_URL) # Use specific error from constants

        # Fetch raw transcript first
        transcript_raw = fetch_transcript(video_id)
        # Clean the transcript for processing
        transcript_cleaned = clean_transcript(transcript_raw)

        step_1_time = time.time() - step_1_start_time
        # Calculate length/tokens based on the *cleaned* transcript for metrics
        tran_len = len(transcript_cleaned)
        tran_tok = count_tokens(transcript_cleaned) # count_tokens imported from utils
        if tran_tok == -1: logger.warning("Failed to count tokens for cleaned transcript.")

        if status_update_func: status_update_func(label=STATUS_STEP_1_COMPLETE.format(tran_len=tran_len, tran_tok=tran_tok if tran_tok !=-1 else "N/A"))
        logger.info(f"Step 1 (Fetch & Clean) completed in {step_1_time:.2f}s")

        # --- Step 2: Generate Content ---
        # generate_content is imported from generation module
        content_json, generation_time = generate_content(transcript_cleaned, status_update_func)
        if status_update_func: status_update_func(label=STATUS_STEP_2_COMPLETE.format(gen_time=generation_time))
        logger.info(f"Step 2 (Generate Content) completed in {generation_time:.2f}s")

        # --- Steps 3 & 4: Evaluation ---
        if status_update_func: status_update_func(label=STATUS_STEP_3) # Indicate start of evaluation
        eval_start_time = time.time()

        # 3a: Validate JSON Structure (imported from evaluation)
        is_valid_json, parsed_content_data = validate_json_structure(content_json)

        # 3b: Extract summary text for evaluation (only if JSON is valid)
        summary_text_for_eval = ""
        if is_valid_json and parsed_content_data:
            summary_text_for_eval = parsed_content_data.get("summary", "")
            if not summary_text_for_eval:
                logger.warning("JSON is valid but 'summary' key is missing or empty. Evaluation might be skipped.")
        elif is_valid_json and not parsed_content_data:
            # This case indicates an internal issue with validate_json_structure
            logger.error("Internal Error: JSON validation reported success but returned no data.")
            is_valid_json = False # Force to invalid state
        else: # JSON is invalid
            logger.warning("JSON is invalid. Summary text cannot be extracted for evaluation.")

        # 3c & 4: Run Async Evaluations using asyncio.run
        # _run_async_evaluations is imported from evaluation
        try:
            bert_score_val, ai_judge_results, ai_judge_skipped, passed_bert_indicator = asyncio.run(
                 _run_async_evaluations(transcript_cleaned, summary_text_for_eval) # Pass cleaned transcript
            )
        except RuntimeError as e:
             # Handle potential nested asyncio loop issues
             if "cannot run loop while another loop is running" in str(e):
                  logger.error("Asyncio loop conflict detected during evaluation.", exc_info=True)
                  # Use specific error constant
                  raise RuntimeError(ERROR_ASYNC_EVAL_LOOP) from e
             else:
                  raise # Re-raise other runtime errors
        except Exception as e: # Catch any other error during async execution
             logger.error(f"Unexpected error during async evaluations: {e}", exc_info=True)
             # Decide how to handle - potentially set flags indicating eval failure
             ai_judge_skipped = True # Mark as skipped/failed
             ai_judge_results = {"error": f"Async evaluation failed: {e}", "overall_pass": False}

        eval_time_total = time.time() - eval_start_time

        # Update Streamlit status based on evaluation completion
        if status_update_func:
            status_update_func(label=STATUS_STEP_3_COMPLETE.format(bert_score=bert_score_val)) # Auto eval complete
            if not ai_judge_skipped:
                status_update_func(label=STATUS_STEP_4_COMPLETE) # AI judge ran
            else:
                status_update_func(label=STATUS_STEP_4_SKIPPED) # AI judge skipped/failed

        logger.info(f"Steps 3 & 4 (Evaluation) completed in {eval_time_total:.2f}s")

        # --- Combine Final Metrics ---
        final_metrics = {
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "generation_model": GENERATION_MODEL,
            "generation_time_sec": generation_time,
            "transcript_chars": tran_len, # Based on cleaned transcript
            "transcript_tokens": tran_tok, # Based on cleaned transcript
            "parsed_content": parsed_content_data if is_valid_json else None,
            "content_json_valid": is_valid_json,
            "summary_bert_score": bert_score_val,
            "summary_bert_passed": passed_bert_indicator, # Based on >= BERT_SCORE_FAIL_THRESHOLD
            "ai_judge_model": AI_JUDGE_MODEL,
            "ai_judge_skipped": ai_judge_skipped,
            "ai_judge_results": ai_judge_results # Contains scores/critique or error info
        }

        # Return raw transcript for display, plus other results
        return transcript_raw, content_json, final_metrics, generation_time

    # --- Error Handling ---
    except (ValueError, RuntimeError, OpenAIError) as e: # Handle specific known errors first
        error_message = str(e)
        # Log exception traceback for detailed debugging
        logger.exception(f"Pipeline error for URL {url}: {error_message}")
        # Update Streamlit status and show error
        if status_update_func: status_update_func(label=STATUS_ERROR.format(error_message), state="error", expanded=True)
        # st.error is handled in main app loop based on return value
        return None # Indicate failure
    except Exception as e: # Catch-all for unexpected errors
        error_message = ERROR_UNKNOWN.format(e)
        logger.exception(f"Unexpected pipeline error for URL {url}")
        if status_update_func: status_update_func(label=STATUS_ERROR.format(error_message), state="error", expanded=True)
        # st.error is handled in main app loop
        return None # Indicate failure