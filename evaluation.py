# evaluation.py
import logging
import json
import asyncio
import time
from typing import Tuple, Optional, Dict, Any
from openai import OpenAIError
from bert_score import score

# Import shared client and constants
from utils import client, logger as root_logger
from constants import (
    AI_JUDGE_MODEL, AI_JUDGE_THRESHOLD, SYSTEM_PROMPT_AI_JUDGE, USER_PROMPT_AI_JUDGE,
    ERROR_EVALUATE_BERT, ERROR_EVALUATE_AI_JUDGE, ERROR_AI_JUDGE_PARSE,
    ERROR_AI_JUDGE_RETRY_FAIL, BERT_SCORE_FAIL_THRESHOLD
)

# Get a logger specific to this module
logger = logging.getLogger(__name__)

def validate_json_structure(content_json: str) -> Tuple[bool, Optional[Dict]]:
    """Validates if the input string is valid JSON and returns parsed data."""
    if not content_json:
        logger.warning("validate_json_structure called with empty input.")
        return False, None
    try:
        data = json.loads(content_json)
        logger.info("JSON validation passed.")
        return True, data
    except json.JSONDecodeError as e:
        logger.warning(f"JSON validation failed: {e}. Content start: {repr(content_json[:100])}...")
        return False, None
    except Exception as e: # Catch other potential errors like TypeError
        logger.error(f"Unexpected error during JSON validation: {e}", exc_info=True)
        return False, None

async def evaluate_bert_score(transcript: str, summary_text: str) -> float:
    """Calculates BERT score F1 between transcript and summary text (Async)."""
    if not summary_text:
        logger.warning("BERT score evaluation skipped: summary text is empty.")
        return 0.0
    if not transcript:
         logger.warning("BERT score evaluation skipped: reference transcript is empty.")
         return 0.0
    try:
        # BERT score is CPU bound, run in thread
        # Ensure inputs are lists of strings
        candidates = [str(summary_text)]
        references = [str(transcript)]
        P, R, F1 = await asyncio.to_thread(
            score, candidates, references, lang='en', verbose=False, model_type='bert-base-uncased'
        )
        # F1 is a tensor, get the scalar value
        f1_score = float(F1.mean().item()) # Use .item() to get Python float
        logger.info(f"BERT score calculated: {f1_score:.4f}")
        return f1_score
    except Exception as e:
        logger.error(ERROR_EVALUATE_BERT.format(e), exc_info=True)
        return 0.0 # Return 0 on error

async def evaluate_with_ai_judge(summary_text: str) -> Dict[str, Any]:
    """Evaluates summary using AI Judge (Async with retries)."""
    if client is None: raise RuntimeError("OpenAI client not initialized.") # Add check
    if not summary_text:
        logger.warning("AI Judge evaluation skipped: summary text is empty.")
        return {"error": "Summary text was empty for evaluation.", "overall_pass": False}

    max_retries = 2
    retry_delay = 5 # seconds
    final_result: Dict[str, Any] = {}

    for attempt in range(max_retries):
        logger.info(f"Attempting AI Judge evaluation (Attempt {attempt + 1}/{max_retries})")
        try:
            prompt = USER_PROMPT_AI_JUDGE.format(summary_text)

            # OpenAI call is IO bound, run in thread
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=AI_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_AI_JUDGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Low temp for deterministic judging
                max_tokens=500, # Limit response size
                response_format={"type": "json_object"} # Enforce JSON output
            )

            result_text = response.choices[0].message.content
            if not result_text: # Handle case where API returns empty content
                logger.error(f"AI Judge API returned empty content (Attempt {attempt+1}).")
                raise ValueError("AI Judge API returned empty content.")

            logger.info(f"AI Judge raw response received (Attempt {attempt+1})")
            # logger.debug(f"AI Judge raw response content: {result_text}") # Optionally log full response in debug

            # Parse and validate the JSON response
            try:
                result = json.loads(result_text)
                required_keys = ["overall_score", "relevance_score", "fluency_score", "critique"]
                scores = {k: result.get(k) for k in required_keys if k != "critique"}
                critique = result.get("critique", "Critique missing from response.")

                # Check if all required score keys are present and are integers
                if not all(k in scores and isinstance(scores[k], int) for k in scores):
                     error_msg = "AI Judge response missing required scores or scores are not integers."
                     logger.error(f"{error_msg} Response: {result}")
                     raise ValueError(error_msg)

                threshold = AI_JUDGE_THRESHOLD
                # Construct successful result dictionary
                final_result = {
                    "overall_score": scores["overall_score"],
                    "relevance_score": scores["relevance_score"],
                    "fluency_score": scores["fluency_score"],
                    "critique": critique,
                    "error": None, # Explicitly set error to None on success
                    "overall_pass": scores["overall_score"] >= threshold,
                    "relevance_pass": scores["relevance_score"] >= threshold,
                    "fluency_pass": scores["fluency_score"] >= threshold
                }
                logger.info("AI Judge evaluation successful.")
                return final_result # Success, exit loop and function

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(ERROR_AI_JUDGE_PARSE.format(attempt=attempt+1, max_retries=max_retries, e=e), exc_info=True)
                # If it's the last attempt, return the failure details
                if attempt == max_retries - 1:
                     final_result = {"error": ERROR_AI_JUDGE_RETRY_FAIL.format(max_retries=max_retries), "raw_response": result_text, "overall_pass": False}
                     return final_result
                # Otherwise, wait and retry
                await asyncio.sleep(retry_delay)

        except OpenAIError as e:
            logger.error(f"OpenAI API error during AI judge evaluation (Attempt {attempt+1}): {e}", exc_info=True)
            if attempt == max_retries - 1:
                 final_result = {"error": ERROR_EVALUATE_AI_JUDGE.format(e), "overall_pass": False}
                 return final_result
            await asyncio.sleep(retry_delay) # Wait before retrying API error

        except ValueError as ve: # Catch empty content error specifically
             logger.error(f"AI Judge returned empty content (Attempt {attempt+1}): {ve}")
             if attempt == max_retries - 1:
                  final_result = {"error": str(ve), "overall_pass": False}
                  return final_result
             await asyncio.sleep(retry_delay) # Wait before retrying

        except Exception as e: # Catch any other unexpected errors during the attempt
            logger.exception(f"Unexpected error in AI judge evaluation attempt {attempt+1}")
            if attempt == max_retries - 1:
                final_result = {"error": f"Unexpected error during evaluation: {str(e)}", "overall_pass": False}
                return final_result
            await asyncio.sleep(retry_delay) # Wait before retrying

    # Fallback if loop finishes unexpectedly (e.g., max_retries is 0)
    # Should have returned within the loop in normal operation.
    if not final_result:
         final_result = {"error": "AI Judge evaluation failed unexpectedly after retries.", "overall_pass": False}
    return final_result


# --- Async Evaluation Helper ---
async def _run_async_evaluations(transcript: str, summary_text_for_eval: str) -> Tuple[float, Optional[Dict[str, Any]], bool, bool]:
    """Runs BERT score and AI judge concurrently. Returns BERT score, AI Judge results, AI Judge skipped status, BERT pass indicator."""
    bert_score_val = 0.0
    ai_judge_results = None
    ai_judge_skipped = True # Assume skipped until attempted
    passed_bert_indicator = False # Default

    tasks_to_run = []
    bert_task_index = -1
    judge_task_index = -1

    # Schedule tasks only if summary text exists
    if summary_text_for_eval:
        # Schedule BERT Score calculation
        tasks_to_run.append(evaluate_bert_score(transcript, summary_text_for_eval))
        bert_task_index = len(tasks_to_run) - 1

        # Schedule AI Judge evaluation
        tasks_to_run.append(evaluate_with_ai_judge(summary_text_for_eval))
        judge_task_index = len(tasks_to_run) - 1
        ai_judge_skipped = False # Mark as attempted
        logger.info("Summary text exists, scheduling BERT and AI Judge evaluations.")
    else:
        logger.info(f"BERT score and AI Judge skipped: summary text empty."); ai_judge_skipped = True

    # Run tasks concurrently if any were scheduled
    if tasks_to_run:
        eval_start_time = time.time()
        # Use return_exceptions=True to handle potential failures in one task without stopping the other
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        logger.info(f"Async evaluations gather() call took {time.time() - eval_start_time:.2f}s")

        # Process BERT results if it ran
        if bert_task_index != -1:
            bert_result = results[bert_task_index]
            if isinstance(bert_result, Exception):
                logger.error(f"BERT Score evaluation failed within gather: {bert_result}", exc_info=bert_result)
                bert_score_val = 0.0 # Assign default value on error
            elif bert_result is not None: # Check if it returned a valid score
                bert_score_val = float(bert_result)
            else: # Should not happen if evaluate_bert_score always returns float
                 logger.error("BERT Score evaluation returned None unexpectedly.")
                 bert_score_val = 0.0

        # Process AI Judge results if it ran
        if judge_task_index != -1:
            judge_result = results[judge_task_index]
            if isinstance(judge_result, Exception):
                logger.error(f"AI Judge evaluation failed within gather: {judge_result}", exc_info=judge_result)
                # Ensure error is captured in the results dict format
                ai_judge_results = {"error": f"Async gather failed: {judge_result}", "overall_pass": False}
                # ai_judge_skipped remains False because it was attempted
            elif isinstance(judge_result, dict): # Check if it returned a dictionary
                 ai_judge_results = judge_result
                 # Log if the dictionary contains an internal error reported by the function
                 if ai_judge_results.get("error"):
                     logger.warning(f"AI Judge ran but returned an internal error state: {ai_judge_results.get('error')}")
            else: # Unexpected return type
                 logger.error(f"AI Judge evaluation returned unexpected type: {type(judge_result)}")
                 ai_judge_results = {"error": "AI Judge returned unexpected data type", "overall_pass": False}
                 # ai_judge_skipped remains False

    # Calculate the pass/fail indicator based on the FAIL threshold (< 0.5)
    # Passed is True if score >= 0.5
    passed_bert_indicator = bert_score_val >= BERT_SCORE_FAIL_THRESHOLD
    logger.info(f"BERT Score: {bert_score_val:.3f}, Pass Threshold: {BERT_SCORE_FAIL_THRESHOLD:.2f}, Passed Indicator: {passed_bert_indicator}")

    return bert_score_val, ai_judge_results, ai_judge_skipped, passed_bert_indicator