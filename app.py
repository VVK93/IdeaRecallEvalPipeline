# app.py
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI, OpenAIError
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
import nest_asyncio # Import nest_asyncio

# --- Setup Logging ---
log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
logging.basicConfig(filename=log_dir / "app.log", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nest_asyncio.apply() # Apply patch for nested asyncio loops

# --- Load Environment Variables & Initialize Clients ---
env_path = Path(__file__).parent / '.env'; load_dotenv(dotenv_path=env_path)
if not API_KEY: logger.critical(ERROR_API_KEY_MISSING); st.error(ERROR_API_KEY_MISSING); st.stop()
try:
    client = OpenAI(api_key=API_KEY); client.models.list(); logger.info("OpenAI client initialized.")
except OpenAIError as e: logger.critical(ERROR_OPENAI_INIT.format(e), exc_info=True); st.error(ERROR_OPENAI_INIT.format(e)); st.stop()
except Exception as e: logger.critical(f"Unexpected init error: {e}", exc_info=True); st.error(f"Unexpected init error: {e}"); st.stop()
try:
    tokenizer = tiktoken.encoding_for_model(TOKENIZER_MODEL); logger.info(f"Tokenizer '{TOKENIZER_MODEL}' initialized.")
except Exception as e: logger.critical(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, e), exc_info=True); st.error(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, e)); st.stop()

# --- Utility Functions ---
def count_tokens(text: str) -> int:
    if not text: return 0
    try: return len(tokenizer.encode(text))
    except Exception as e: logger.warning(ERROR_TOKEN_COUNT.format(e), exc_info=True); return -1
def extract_video_id(url: str) -> Optional[str]:
    patterns = [ r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)', r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([^&\n?#]+)' ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: video_id = match.group(1); logger.info(f"Extracted video ID: {video_id}"); return video_id
    logger.warning(f"Could not extract video ID from URL: {url}"); return None

# --- Core Logic Functions ---
def fetch_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        logger.info(f"Fetched transcript {video_id}. Length: {len(full_transcript)} chars.")
        return full_transcript
    except TranscriptsDisabled: logger.error(f"Transcripts disabled: {video_id}"); raise ValueError(ERROR_TRANSCRIPT_DISABLED)
    except NoTranscriptFound: logger.error(f"No transcript found: {video_id}"); raise ValueError(ERROR_NO_TRANSCRIPT)
    except Exception as e: logger.error(f"Error fetching transcript: {e}", exc_info=True); raise RuntimeError(ERROR_FETCH_TRANSCRIPT.format(e))
def clean_transcript(text: str) -> str:
    original_len = len(text)
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}(?:.\d{3})?\]', '', text) # Timestamps
    text = re.sub(r'Speaker \d+:', '', text) # Speaker tags
    text = re.sub(r'\[Music\]', '', text, flags=re.IGNORECASE) # Music cues
    text = re.sub(r'\s+', ' ', text).strip() # Whitespace
    text = re.sub(r'Link in the description to apply.*', '', text, flags=re.IGNORECASE) # Ad
    text = re.sub(r'apply for a spot.*', '', text, flags=re.IGNORECASE) # Ad
    cleaned_len = len(text)
    logger.info(f"Transcript cleaned. Original length: {original_len} -> Cleaned length: {cleaned_len} chars.")
    return text

# --- Content Generation Functions ---
def _generate_content_single_call(transcript: str) -> str:
    response = client.chat.completions.create( model=GENERATION_MODEL, messages=[ {"role": "system", "content": SYSTEM_PROMPT_GENERATION}, {"role": "user", "content": USER_PROMPT_GENERATION.format(transcript)} ], temperature=0.5, response_format={"type": "json_object"} )
    content = response.choices[0].message.content
    if not content: logger.error("OpenAI API returned empty content for single call."); raise ValueError("LLM returned empty content.")
    return content
def _generate_content_map_reduce(transcript: str, status_update_func) -> str:
    tokens = tokenizer.encode(transcript); chunks = []; start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + CHUNK_SIZE, len(tokens)); chunk_text = tokenizer.decode(tokens[start_index:end_index])
        chunks.append(chunk_text); start_index += CHUNK_SIZE - CHUNK_OVERLAP
        if start_index >= len(tokens): break
    logger.info(f"Split into {len(chunks)} chunks.")
    chunk_content_jsons = []
    for i, chunk in enumerate(chunks):
        status_update_func(label=STATUS_STEP_2_MAP.format(i=i+1, n=len(chunks))); logger.info(f"Map: Chunk {i+1}/{len(chunks)}")
        try:
            chunk_json_str = _generate_content_single_call(chunk)
            try: json.loads(chunk_json_str); chunk_content_jsons.append(chunk_json_str); logger.debug(f"Chunk {i+1} valid JSON.")
            except json.JSONDecodeError: logger.warning(f"Chunk {i+1} invalid JSON response, skipping."); continue
        except ValueError as ve: logger.error(f"Chunk {i+1} empty content, skipping. Error: {ve}"); continue
        except Exception as e: logger.error(ERROR_GENERATE_CONTENT_CHUNK.format(i=i+1, n=len(chunks), e=e), exc_info=True); logger.warning(f"Skipping chunk {i+1} due to error.")
    if not chunk_content_jsons: raise RuntimeError("Map phase failed: No valid JSON generated.")
    status_update_func(label=STATUS_STEP_2_REDUCE); logger.info("Reduce: Combining.")
    combined_input_str = "\n\n".join(chunk_content_jsons)
    combined_tokens = count_tokens(SYSTEM_PROMPT_COMBINE + USER_PROMPT_COMBINE.format(combined_input_str))
    if combined_tokens > MAX_MODEL_TOKENS_SUMMARY: logger.warning(f"Combined input for Reduce ({combined_tokens} tokens) may exceed model limit.")
    try:
        final_content_json = client.chat.completions.create( model=GENERATION_MODEL, messages=[ {"role": "system", "content": SYSTEM_PROMPT_COMBINE}, {"role": "user", "content": USER_PROMPT_COMBINE.format(combined_input_str)} ], temperature=0.5, response_format={"type": "json_object"} ).choices[0].message.content
        if not final_content_json: logger.error("Reduce step returned empty content."); raise ValueError("LLM returned empty content during Reduce phase.")
        return final_content_json
    except ValueError as ve: logger.error(f"Reduce step failed: {ve}"); raise RuntimeError(f"Reduce step failed: {ve}") from ve
    except Exception as e: logger.error(ERROR_GENERATE_CONTENT_COMBINE.format(e=e), exc_info=True); raise RuntimeError(ERROR_GENERATE_CONTENT_COMBINE.format(e=e))
def generate_content(transcript: str, status_update_func) -> Tuple[str, float]:
    start_time = time.time()
    try:
        prompt_tokens = count_tokens(SYSTEM_PROMPT_GENERATION + USER_PROMPT_GENERATION.format("")); transcript_tokens = count_tokens(transcript)
        total_tokens_single_call = prompt_tokens + transcript_tokens
        logger.info(f"Cleaned transcript token count: {transcript_tokens}"); logger.info(f"Est. single call tokens: {total_tokens_single_call} vs Max: {MAX_MODEL_TOKENS_SUMMARY}")
        if total_tokens_single_call <= MAX_MODEL_TOKENS_SUMMARY:
            status_update_func(label=STATUS_STEP_2_SINGLE); logger.info("Processing in single call.")
            content_json = _generate_content_single_call(transcript)
        else:
            logger.warning("Using Map-Reduce."); num_chunks_est = (transcript_tokens // (CHUNK_SIZE - CHUNK_OVERLAP)) + 1
            status_update_func(label=STATUS_STEP_2_CHUNK.format(num_chunks=num_chunks_est))
            content_json = _generate_content_map_reduce(transcript, status_update_func)
        generation_time = round(time.time() - start_time, 2)
        logger.info(f"Content gen completed in {generation_time}s.")
        if not content_json or not isinstance(content_json, str): logger.error("Final content empty/invalid."); raise ValueError("LLM generation resulted in empty/invalid final content.")
        return content_json, generation_time
    except (OpenAIError, ValueError, RuntimeError) as e: logger.error(ERROR_GENERATE_CONTENT_GENERAL.format(e), exc_info=True); raise RuntimeError(ERROR_GENERATE_CONTENT_GENERAL.format(e)) from e
    except Exception as e: logger.error(f"Unexpected content gen error: {e}", exc_info=True); raise RuntimeError(f"Unexpected content gen error: {e}") from e

# --- Evaluation Functions ---
def validate_json_structure(content_json: str) -> Tuple[bool, Optional[Dict]]:
    if not content_json: return False, None
    try: data = json.loads(content_json); logger.info("JSON validation passed."); return True, data
    except json.JSONDecodeError as e: logger.warning(f"JSON validation failed: {e}. Start: {repr(content_json[:100])}..."); return False, None
    except Exception as e: logger.error(f"Unexpected JSON validation error: {e}", exc_info=True); return False, None
async def evaluate_bert_score(transcript: str, summary_text: str) -> float:
    if not summary_text: logger.warning("BERT score skipped: summary empty."); return 0.0
    try:
        P, R, F1 = await asyncio.to_thread(score, [summary_text], [transcript], lang='en', verbose=False, model_type='bert-base-uncased')
        f1_score = float(F1.mean()); logger.info(f"BERT score: {f1_score:.4f}"); return f1_score
    except Exception as e: logger.error(ERROR_EVALUATE_BERT.format(e), exc_info=True); return 0.0
async def evaluate_with_ai_judge(summary_text: str) -> Dict[str, Any]:
    if not summary_text: logger.warning("AI Judge skipped: summary empty."); return {"error": "Summary text empty.", "overall_pass": False}
    max_retries = 2; retry_delay = 5; final_result = {}
    for attempt in range(max_retries):
        try:
            prompt = USER_PROMPT_AI_JUDGE.format(summary_text)
            response = await asyncio.to_thread( client.chat.completions.create, model=AI_JUDGE_MODEL, messages=[ {"role": "system", "content": SYSTEM_PROMPT_AI_JUDGE}, {"role": "user", "content": prompt} ], temperature=0.0, max_tokens=500, response_format={"type": "json_object"} )
            result_text = response.choices[0].message.content
            if not result_text: logger.error(f"AI Judge empty content (Attempt {attempt+1})."); raise ValueError("AI Judge API returned empty content.")
            logger.info(f"AI Judge raw response (attempt {attempt+1}): {result_text}")
            try:
                result = json.loads(result_text); required_keys = ["overall_score", "relevance_score", "fluency_score", "critique"]
                scores = {k: result.get(k) for k in required_keys if k != "critique"}; critique = result.get("critique", "Critique missing.")
                if not all(k in scores and isinstance(scores[k], int) for k in scores): raise ValueError("Missing/invalid scores.")
                threshold = AI_JUDGE_THRESHOLD
                final_result = { "overall_score": scores["overall_score"], "relevance_score": scores["relevance_score"], "fluency_score": scores["fluency_score"], "critique": critique, "error": None, "overall_pass": scores["overall_score"] >= threshold, "relevance_pass": scores["relevance_score"] >= threshold, "fluency_pass": scores["fluency_score"] >= threshold }
                logger.info(f"AI Judge evaluation successful."); return final_result
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(ERROR_AI_JUDGE_PARSE.format(attempt=attempt+1, max_retries=max_retries, e=e))
                if attempt == max_retries - 1: final_result = {"error": ERROR_AI_JUDGE_RETRY_FAIL.format(max_retries=max_retries), "raw_response": result_text, "overall_pass": False}; return final_result
                await asyncio.sleep(retry_delay)
        except OpenAIError as e:
            logger.error(f"OpenAI API error AI judge (attempt {attempt+1}): {e}", exc_info=True)
            if attempt == max_retries - 1: final_result = {"error": ERROR_EVALUATE_AI_JUDGE.format(e), "overall_pass": False}; return final_result
            await asyncio.sleep(retry_delay)
        except ValueError as ve:
             logger.error(f"AI Judge empty content error (Attempt {attempt+1}): {ve}")
             if attempt == max_retries - 1: final_result = {"error": str(ve), "overall_pass": False}; return final_result
             await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.exception(f"Unexpected error AI judge (attempt {attempt+1})")
            if attempt == max_retries - 1: final_result = {"error": f"Unexpected error: {str(e)}", "overall_pass": False}; return final_result
            await asyncio.sleep(retry_delay)
    if not final_result: final_result = {"error": "AI Judge failed unexpectedly after retries.", "overall_pass": False}
    return final_result

# --- UI Display Function ---
def display_final_results( transcript: str, content_json: str, metrics: Dict[str, Any], generation_time: float ) -> None:
    """Displays the final results with updated BERT interpretation and overall status logic."""
    st.header(RESULTS_HEADER)
    st.divider()

    # --- Parse Content ---
    parsed_data = metrics.get("parsed_content")
    summary_text = ""
    flashcards = []
    is_valid_json_display = metrics.get("content_json_valid", False)
    if is_valid_json_display and parsed_data:
        summary_text = parsed_data.get("summary", "*Summary text not found in JSON*")
        flashcards = parsed_data.get("flashcards", [])
        if not isinstance(flashcards, list):
             logger.warning("Parsed 'flashcards' is not a list.")
             flashcards = []
    elif not is_valid_json_display:
        summary_text = content_json # Show raw if invalid JSON

    # --- Transcript Display ---
    st.subheader(TRANSCRIPT_HEADER)
    tran_len = len(transcript); tran_tok = count_tokens(transcript)
    st.caption(f"{LENGTH_LABEL.format(tran_len)} | {TOKENS_LABEL.format(tran_tok) if tran_tok != -1 else 'Token error'}")
    with st.expander(VIEW_TRANSCRIPT_EXPANDER, expanded=False):
        st.text_area("Transcript_Text", transcript, height=300, key="transcript_display", disabled=True)
    st.divider()

    # --- Generated Content Display ---
    st.subheader(GENERATED_CONTENT_HEADER)
    col1, col2 = st.columns([2, 1])
    with col1: # Summary and Flashcards
        st.markdown(f"**{SUMMARY_SUBHEADER}**")
        sum_len = len(summary_text); sum_tok = count_tokens(summary_text)
        st.caption(f"{LENGTH_LABEL.format(sum_len)} | {TOKENS_LABEL.format(sum_tok) if sum_tok != -1 else 'Token error'}")
        st.markdown(summary_text if summary_text else "*Summary empty or generation failed.*")

        st.markdown(f"**{FLASHCARDS_SUBHEADER}**")
        st.caption(FLASHCARD_COUNT_LABEL.format(count=len(flashcards)))
        if flashcards:
            for i, card in enumerate(flashcards):
                q = card.get("question", f"Card {i+1}: Q missing"); a = card.get("answer", f"Card {i+1}: A missing")
                with st.expander(f"Flashcard {i+1}: {q[:50]}..."):
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
        elif is_valid_json_display:
            st.info(NO_FLASHCARDS_GENERATED)
        # Implicitly handles invalid JSON case by not showing flashcards

    with col2: # Raw JSON Output
         st.markdown(f"**Raw Output**")
         with st.expander(VIEW_GENERATED_JSON_EXPANDER, expanded=False):
            if is_valid_json_display and parsed_data:
                st.json(parsed_data, expanded=False)
            else:
                st.warning(f"{INFO_ICON} Output was not valid JSON.")
                st.text_area("Generated_Raw_Output", content_json, height=300, key="generated_raw_display", disabled=True)
    st.divider()

    # --- Evaluation Metrics Display ---
    st.subheader(EVALUATION_HEADER)
    st.metric(GENERATION_TIME_LABEL, f"{generation_time:.2f} s")
    st.divider()

    # Automatic Checks Display
    st.markdown(f"**{METRICS_AUTO_HEADER}**")
    col_auto1, col_auto2 = st.columns(2)
    with col_auto1: # JSON Validation
        json_status = f"{PASS_ICON} {JSON_VALID_LABEL}" if is_valid_json_display else f"{FAIL_ICON} {JSON_INVALID_LABEL}"
        st.metric(JSON_VALIDATION_LABEL, json_status)
    with col_auto2: # BERT Score (Color-coded)
        bert_score_val = metrics.get("summary_bert_score", 0.0)
        if bert_score_val >= BERT_SCORE_GREEN_THRESHOLD: color = "green"
        elif bert_score_val >= BERT_SCORE_YELLOW_THRESHOLD: color = "orange"
        else: color = "red"
        st.markdown(f"**{BERT_LABEL}**: <span style='color:{color};'>{bert_score_val:.3f}</span>", unsafe_allow_html=True)
        st.caption(BERT_HELP_TEXT)
    st.divider()

    # AI Judge Display
    st.markdown(f"**{METRICS_AI_HEADER}**")
    ai_results = metrics.get("ai_judge_results")
    ai_judge_skipped = metrics.get("ai_judge_skipped", True)

    if ai_judge_skipped:
        if not is_valid_json_display: st.info(f"{INFO_ICON} AI Judge skipped (JSON invalid).")
        elif is_valid_json_display and not summary_text: st.info(f"{INFO_ICON} AI Judge skipped (Summary text empty in valid JSON).")
        else: st.info(f"{INFO_ICON} {INFO_AI_JUDGE_SKIPPED}") # Fallback
    elif ai_results:
        if ai_results.get("error"): # Check if AI Judge call itself failed
            st.error(f"{FAIL_ICON} AI Judge evaluation failed: {ai_results['error']}")
            if "raw_response" in ai_results:
                # CORRECT INDENTATION HERE
                with st.expander("View Raw AI Judge Error Response"):
                     st.code(ai_results["raw_response"], language="json")
        else: # AI Judge ran successfully, display results
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            threshold = AI_JUDGE_THRESHOLD # Local threshold for display comparison
            with col_ai1:
                ov_score = ai_results.get('overall_score', 0)
                ov_pass_icon = f"{PASS_ICON}" if ai_results.get('overall_pass') else f"{WARN_ICON}"
                st.metric(AI_OVERALL_LABEL, f"{ov_score}{SCORE_SUFFIX}", ov_pass_icon)
            with col_ai2:
                rel_score = ai_results.get('relevance_score', 0)
                rel_pass_icon = f"{PASS_ICON}" if ai_results.get('relevance_pass') else f"{WARN_ICON}"
                st.metric(AI_RELEVANCE_LABEL, f"{rel_score}{SCORE_SUFFIX}", rel_pass_icon)
            with col_ai3:
                flu_score = ai_results.get('fluency_score', 0)
                flu_pass_icon = f"{PASS_ICON}" if ai_results.get('fluency_pass') else f"{WARN_ICON}"
                st.metric(AI_FLUENCY_LABEL, f"{flu_score}{SCORE_SUFFIX}", flu_pass_icon)

            critique = ai_results.get("critique", "No critique provided.")
            # CORRECT INDENTATION HERE
            with st.expander(VIEW_AI_JUDGE_CRITIQUE_EXPANDER):
                st.markdown(critique)

            # Overall Status Indicator Logic
            ai_judge_ran_successfully = not ai_judge_skipped and ai_results and not ai_results.get("error")
            ai_judge_passed_all_metrics = ai_judge_ran_successfully and ( ai_results.get('overall_pass', False) and ai_results.get('relevance_pass', False) and ai_results.get('fluency_pass', False) )
            if is_valid_json_display and ai_judge_passed_all_metrics:
                st.success(f"{PASS_ICON} {SUCCESS_MESSAGE}")
            elif is_valid_json_display and ai_judge_ran_successfully and not ai_judge_passed_all_metrics:
                st.warning(f"{WARN_ICON} JSON valid, but AI Judge metrics for summary below threshold.")
            else: # Covers JSON invalid OR AI Judge skipped/error
                st.warning(f"{WARN_ICON} {WARNING_MESSAGE}")
    else:
        st.error("Internal error: AI judge results missing unexpectedly.") # Should not be reached

    st.divider()
    # CORRECT INDENTATION HERE
    with st.expander(VIEW_DETAILS_EXPANDER):
        st.json(metrics)


# --- Async Evaluation Helper ---
async def _run_async_evaluations(transcript: str, summary_text_for_eval: str) -> Tuple[float, Optional[Dict[str, Any]], bool, bool]:
    """Runs BERT & AI judge. Returns BERT score, AI Judge results, AI Judge skipped status, BERT pass indicator."""
    bert_score_val = 0.0; ai_judge_results = None; ai_judge_skipped = True
    tasks_to_run = []
    bert_task_index = -1
    judge_task_index = -1

    # Schedule BERT Score calculation if summary text exists
    if summary_text_for_eval:
        tasks_to_run.append(evaluate_bert_score(transcript, summary_text_for_eval))
        bert_task_index = len(tasks_to_run) - 1
        tasks_to_run.append(evaluate_with_ai_judge(summary_text_for_eval))
        judge_task_index = len(tasks_to_run) - 1
        ai_judge_skipped = False # Mark as attempted
        logger.info("Summary text exists, scheduling BERT and AI Judge.")
    else:
        logger.info(f"BERT score and AI Judge skipped: summary text empty."); ai_judge_skipped = True

    # Run tasks concurrently if any were scheduled
    if tasks_to_run:
        eval_start_time = time.time()
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        logger.info(f"Async evaluations gather took {time.time() - eval_start_time:.2f}s")

        # Process BERT results if it ran
        if bert_task_index != -1:
            if isinstance(results[bert_task_index], Exception):
                logger.error(f"BERT Score evaluation failed: {results[bert_task_index]}", exc_info=results[bert_task_index])
                bert_score_val = 0.0
            else:
                bert_score_val = results[bert_task_index]

        # Process AI Judge results if it ran
        if judge_task_index != -1:
            if isinstance(results[judge_task_index], Exception):
                logger.error(f"AI Judge evaluation failed in gather: {results[judge_task_index]}", exc_info=results[judge_task_index])
                # Ensure error is captured in the results dict format expected by display logic
                ai_judge_results = {"error": f"Async gather failed: {results[judge_task_index]}", "overall_pass": False}
            else:
                ai_judge_results = results[judge_task_index]
                # If judge returns an error dict internally, keep it
                if not ai_judge_results or isinstance(ai_judge_results.get("error"), str):
                     logger.warning(f"AI Judge ran but returned an error state: {ai_judge_results.get('error')}")

    # Calculate the pass/fail indicator based on the FAIL threshold (< 0.5)
    passed_bert_indicator = bert_score_val >= BERT_SCORE_FAIL_THRESHOLD
    logger.info(f"BERT Score: {bert_score_val:.3f}, Pass Threshold: {BERT_SCORE_FAIL_THRESHOLD:.2f}, Passed Indicator: {passed_bert_indicator}")

    return bert_score_val, ai_judge_results, ai_judge_skipped, passed_bert_indicator

# --- Main Synchronous Pipeline Function ---
def run_pipeline(url: str, status) -> Optional[Tuple[str, str, Dict, float]]:
    """Runs the full pipeline synchronously, calling async helper for evaluations."""
    # Initialize variables
    transcript_raw = ""; transcript_cleaned = ""; content_json = ""; generation_time = 0.0
    final_metrics = {}; video_id = None; parsed_content_data = None
    is_valid_json = False; summary_text_for_eval = ""
    bert_score_val = 0.0; ai_judge_results = None; ai_judge_skipped = True; passed_bert_indicator = False

    try:
        # Step 1: Fetch & Clean Transcript
        status.update(label=STATUS_STEP_1); start_time = time.time(); video_id = extract_video_id(url)
        if not video_id: raise ValueError(ERROR_INVALID_URL)
        transcript_raw = fetch_transcript(video_id); transcript_cleaned = clean_transcript(transcript_raw)
        step_1_time = time.time() - start_time; tran_len = len(transcript_cleaned); tran_tok = count_tokens(transcript_cleaned)
        status.update(label=STATUS_STEP_1_COMPLETE.format(tran_len=tran_len, tran_tok=tran_tok)); logger.info(f"Step 1 took {step_1_time:.2f}s")

        # Step 2: Generate Content
        content_json, generation_time = generate_content(transcript_cleaned, status.update)
        status.update(label=STATUS_STEP_2_COMPLETE.format(gen_time=generation_time)); logger.info(f"Step 2 took {generation_time:.2f}s")

        # Steps 3 & 4: Evaluation Prep
        status.update(label=STATUS_STEP_3); eval_start_time = time.time()
        is_valid_json, parsed_content_data = validate_json_structure(content_json)
        if is_valid_json and parsed_content_data: summary_text_for_eval = parsed_content_data.get("summary", "")
        elif is_valid_json: logger.warning("JSON valid but summary text missing/empty.")
        else: logger.warning("JSON invalid, cannot extract summary for eval.")

        # Run Async Evals
        try:
            bert_score_val, ai_judge_results, ai_judge_skipped, passed_bert_indicator = asyncio.run(
                 _run_async_evaluations(transcript_cleaned, summary_text_for_eval)
            )
        except RuntimeError as e:
             if "cannot run loop while another loop is running" in str(e): logger.error("Asyncio loop conflict.", exc_info=True); raise RuntimeError(ERROR_ASYNC_EVAL_LOOP) from e
             else: raise

        eval_time_total = time.time() - eval_start_time
        status.update(label=STATUS_STEP_3_COMPLETE.format(bert_score=bert_score_val)) # Auto eval complete
        if not ai_judge_skipped: status.update(label=STATUS_STEP_4_COMPLETE) # AI judge status
        else: status.update(label=STATUS_STEP_4_SKIPPED)
        logger.info(f"Steps 3 & 4 (Evaluation) took {eval_time_total:.2f}s")

        # Combine Final Metrics
        final_metrics = { "video_id": video_id, "timestamp": datetime.now().isoformat(), "generation_model": GENERATION_MODEL, "generation_time_sec": generation_time, "transcript_chars": tran_len, "transcript_tokens": tran_tok, "parsed_content": parsed_content_data if is_valid_json else None, "content_json_valid": is_valid_json, "summary_bert_score": bert_score_val, "summary_bert_passed": passed_bert_indicator, "ai_judge_model": AI_JUDGE_MODEL, "ai_judge_skipped": ai_judge_skipped, "ai_judge_results": ai_judge_results }
        return transcript_raw, content_json, final_metrics, generation_time

    # Consolidate error handling
    except (ValueError, RuntimeError, OpenAIError) as e: # Handle specific known errors first
        error_message = str(e); logger.exception(f"Pipeline error: {error_message}")
        status.update(label=STATUS_ERROR.format(error_message), state="error", expanded=True); st.error(f"{FAIL_ICON} Processing failed: {error_message}")
        return None
    except Exception as e: # Catch-all for unexpected errors
        error_message = ERROR_UNKNOWN.format(e); logger.exception(f"Unexpected pipeline error")
        status.update(label=STATUS_ERROR.format(error_message), state="error", expanded=True); st.error(f"{FAIL_ICON} An unexpected error occurred: {e}")
        return None

# --- Main Streamlit App Function ---
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(APP_TITLE); st.caption(APP_SUBHEADER); st.info(DEMO_SCOPE_INFO, icon=INFO_ICON)
    url = st.text_input(URL_INPUT_LABEL, placeholder=URL_PLACEHOLDER, key="youtube_url_input")
    if url:
        results = None
        with st.status(STATUS_IN_PROGRESS, expanded=True) as status: results = run_pipeline(url, status)
        if results:
            transcript_raw, content_json, final_metrics, generation_time = results
            status.update(label=STATUS_ALL_COMPLETE, state="complete", expanded=False)
            display_final_results(transcript_raw, content_json, final_metrics, generation_time)

if __name__ == "__main__":
    main()