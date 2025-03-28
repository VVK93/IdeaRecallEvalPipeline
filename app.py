# app.py
import streamlit as st
import logging # Import standard logging
import asyncio
from typing import Tuple, Optional, Dict, Any # Import necessary types

# Import pipeline and constants
from pipeline import run_pipeline
from constants import * # Import all constants

# Import shared logger setup from utils (to ensure configuration is applied)
# Also check if client/tokenizer initialization failed
import utils
if utils.client is None: st.error(ERROR_OPENAI_INIT.format("Initialization failed. Check logs.")); st.stop()
if utils.tokenizer is None: st.error(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, "Initialization failed. Check logs.")); st.stop()

# Get a logger specific to this module
logger = logging.getLogger(__name__)

# Import and apply nest_asyncio (needs to be done early)
import nest_asyncio
nest_asyncio.apply()

# --- UI Display Function ---
def display_final_results( transcript: str, content_json: str, metrics: Dict[str, Any], generation_time: float ) -> None:
    """Displays the final results with updated BERT interpretation and overall status logic."""
    # --- Header ---
    st.header(RESULTS_HEADER)
    st.divider()

    # --- Parse Content for Display ---
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
        # If JSON is invalid, show the raw JSON string as the 'summary' for context
        summary_text = content_json
        logger.info("Displaying raw content because JSON was invalid.")


    # --- Transcript Display ---
    st.subheader(TRANSCRIPT_HEADER)
    # Displaying length/tokens of the *original* transcript passed in
    tran_len = len(transcript); tran_tok = utils.count_tokens(transcript) # Use count_tokens from utils
    st.caption(f"{LENGTH_LABEL.format(tran_len)} | {TOKENS_LABEL.format(tran_tok) if tran_tok != -1 else 'Token error'}")
    with st.expander(VIEW_TRANSCRIPT_EXPANDER, expanded=False):
        st.text_area("Transcript_Text", transcript, height=300, key="transcript_display", disabled=True)
    st.divider()

    # --- Generated Content Display ---
    st.subheader(GENERATED_CONTENT_HEADER)
    col1, col2 = st.columns([2, 1]) # Main content | Raw output
    with col1: # Summary and Flashcards Display
        st.markdown(f"**{SUMMARY_SUBHEADER}**")
        sum_len = len(summary_text); sum_tok = utils.count_tokens(summary_text) # Use count_tokens
        st.caption(f"{LENGTH_LABEL.format(sum_len)} | {TOKENS_LABEL.format(sum_tok) if sum_tok != -1 else 'Token error'}")
        # Display summary text (or raw JSON if invalid)
        st.markdown(summary_text if summary_text else "*Summary empty or generation failed.*")

        # Only show flashcard section if JSON was valid
        if is_valid_json_display:
            st.markdown(f"**{FLASHCARDS_SUBHEADER}**")
            st.caption(FLASHCARD_COUNT_LABEL.format(count=len(flashcards)))
            if flashcards:
                for i, card in enumerate(flashcards):
                    q = card.get("question", f"Card {i+1}: Q missing"); a = card.get("answer", f"Card {i+1}: A missing")
                    with st.expander(f"Flashcard {i+1}: {q[:50]}..."):
                        st.markdown(f"**Q:** {q}")
                        st.markdown(f"**A:** {a}")
            else: # JSON valid, but no flashcards found
                st.info(NO_FLASHCARDS_GENERATED)

    with col2: # Raw JSON Output Display
         st.markdown(f"**Raw Output**")
         with st.expander(VIEW_GENERATED_JSON_EXPANDER, expanded=False):
            # Show parsed JSON if valid, otherwise show raw content string
            if is_valid_json_display and parsed_data:
                st.json(parsed_data, expanded=False)
            else:
                st.warning(f"{INFO_ICON} Output was not valid JSON.")
                # Display the raw content_json string received from generation
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
        # Determine color based on constants
        if bert_score_val >= BERT_SCORE_GREEN_THRESHOLD: color = "green"
        elif bert_score_val >= BERT_SCORE_YELLOW_THRESHOLD: color = "orange" # Use constant
        else: color = "red"
        # Display using markdown for color
        st.markdown(f"**{BERT_LABEL}**: <span style='color:{color};'>{bert_score_val:.3f}</span>", unsafe_allow_html=True)
        st.caption(BERT_HELP_TEXT) # Use help text from constants
    st.divider()

    # AI Judge Display
    st.markdown(f"**{METRICS_AI_HEADER}**")
    ai_results = metrics.get("ai_judge_results")
    ai_judge_skipped = metrics.get("ai_judge_skipped", True)

    if ai_judge_skipped:
        # Provide specific context for skipping if possible
        if not is_valid_json_display:
             st.info(f"{INFO_ICON} AI Judge skipped (JSON invalid).")
        elif is_valid_json_display and not summary_text:
             # Check if summary was empty *within* a valid JSON structure
             st.info(f"{INFO_ICON} AI Judge skipped (Summary text empty in valid JSON).")
        else:
             # Fallback, might occur if summary_text_for_eval was empty for other reasons
             st.info(f"{INFO_ICON} {INFO_AI_JUDGE_SKIPPED}")
    elif ai_results:
        # Check if the AI Judge evaluation itself resulted in an error
        if ai_results.get("error"):
            st.error(f"{FAIL_ICON} AI Judge evaluation failed: {ai_results['error']}")
            # Display raw response if available in the error dict
            if "raw_response" in ai_results:
                with st.expander("View Raw AI Judge Error Response"):
                     st.code(ai_results["raw_response"], language="json")
        else: # AI Judge ran successfully, display scores and critique
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            # Use local threshold constant for display logic
            threshold = AI_JUDGE_THRESHOLD
            with col_ai1:
                ov_score = ai_results.get('overall_score', 0)
                ov_pass = ai_results.get('overall_pass', False)
                ov_pass_icon = f"{PASS_ICON}" if ov_pass else f"{WARN_ICON}"
                st.metric(AI_OVERALL_LABEL, f"{ov_score}{SCORE_SUFFIX}", ov_pass_icon)
            with col_ai2:
                rel_score = ai_results.get('relevance_score', 0)
                rel_pass = ai_results.get('relevance_pass', False)
                rel_pass_icon = f"{PASS_ICON}" if rel_pass else f"{WARN_ICON}"
                st.metric(AI_RELEVANCE_LABEL, f"{rel_score}{SCORE_SUFFIX}", rel_pass_icon)
            with col_ai3:
                flu_score = ai_results.get('fluency_score', 0)
                flu_pass = ai_results.get('fluency_pass', False)
                flu_pass_icon = f"{PASS_ICON}" if flu_pass else f"{WARN_ICON}"
                st.metric(AI_FLUENCY_LABEL, f"{flu_score}{SCORE_SUFFIX}", flu_pass_icon)

            critique = ai_results.get("critique", "No critique provided.")
            with st.expander(VIEW_AI_JUDGE_CRITIQUE_EXPANDER):
                st.markdown(critique)

            # --- Determine Overall Status based on JSON validity and AI Judge success/scores ---
            ai_judge_ran_successfully = not ai_judge_skipped and ai_results and not ai_results.get("error")
            ai_judge_passed_all_metrics = ai_judge_ran_successfully and (
                ai_results.get('overall_pass', False) and
                ai_results.get('relevance_pass', False) and
                ai_results.get('fluency_pass', False)
            )

            if is_valid_json_display and ai_judge_passed_all_metrics:
                # Success: JSON is valid AND AI Judge ran AND AI Judge passed all metrics
                st.success(f"{PASS_ICON} {SUCCESS_MESSAGE}")
            elif is_valid_json_display and ai_judge_ran_successfully and not ai_judge_passed_all_metrics:
                # Warning: JSON valid, AI Judge ran but metrics were low
                st.warning(f"{WARN_ICON} JSON valid, but AI Judge metrics for summary below threshold.")
            else:
                # Warning: Covers JSON invalid OR AI Judge skipped/failed error
                st.warning(f"{WARN_ICON} {WARNING_MESSAGE}")
    else:
        # This state should ideally not be reached if ai_judge_skipped is handled correctly
        st.error("Internal error: AI judge results missing when not marked as skipped.")

    st.divider()
    # Expander for Raw Metrics
    with st.expander(VIEW_DETAILS_EXPANDER):
        st.json(metrics) # Display the final_metrics dictionary

# --- Main Streamlit App Function ---
def main():
    """Sets up and runs the Streamlit application."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBHEADER)
    # Display informational box about the demo scope
    st.info(DEMO_SCOPE_INFO, icon=INFO_ICON)

    # Get user input
    url = st.text_input(URL_INPUT_LABEL, placeholder=URL_PLACEHOLDER, key="youtube_url_input")

    # Process if URL is provided
    if url:
        results = None
        # Use st.status for progress indication during pipeline execution
        with st.status(STATUS_IN_PROGRESS, expanded=True) as status_ui:
            # Call the synchronous pipeline function, passing the status object's update method
            results = run_pipeline(url, status_ui.update)

        # Display results if pipeline execution was successful (didn't return None)
        if results:
            transcript_raw, content_json, final_metrics, generation_time = results
            # Update status to complete and collapse the box
            status_ui.update(label=STATUS_ALL_COMPLETE, state="complete", expanded=False)
            # Call the display function
            display_final_results(transcript_raw, content_json, final_metrics, generation_time)
        else:
            # Error occurred, message already shown via status_ui.update(state="error") and st.error() in run_pipeline
            logger.warning(f"Pipeline execution failed for URL: {url}")
            # No further action needed here, error is already displayed

if __name__ == "__main__":
    main()