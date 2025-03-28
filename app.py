# app.py
import streamlit as st
import logging
import asyncio
import json # Added for saving feedback
from pathlib import Path # Added for feedback file path
from datetime import datetime # Added for feedback timestamp
from typing import Dict, Any, Tuple, Optional # Ensure all types are imported

# Import pipeline function and constants
from pipeline import run_pipeline
from constants import * # Import all constants

# Import utils to access logger and check client/tokenizer init
import utils
if utils.client is None: st.error(ERROR_OPENAI_INIT.format("Initialization failed. Check logs.")); st.stop()
if utils.tokenizer is None: st.error(ERROR_TOKENIZER_INIT.format(TOKENIZER_MODEL, "Initialization failed. Check logs.")); st.stop()

# Get a logger specific to this module
logger = logging.getLogger(__name__)

# Import and apply nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# --- Feedback Saving Function ---
FEEDBACK_FILE = Path("feedback_log.jsonl")

def save_feedback(feedback_data: Dict[str, Any]):
    """Appends feedback data as a new line in the JSON Lines file."""
    try:
        # Ensure data has a timestamp
        feedback_data["feedback_timestamp"] = datetime.now().isoformat()
        json_string = json.dumps(feedback_data)
        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json_string + "\n")
        logger.info(f"Feedback saved successfully for video_id: {feedback_data.get('video_id')}")
        return True
    except IOError as e:
        logger.error(f"IOError saving feedback: {e}", exc_info=True)
        st.error(f"Error saving feedback: Could not write to file ({FEEDBACK_FILE}).")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving feedback: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while saving feedback.")
        return False

# --- UI Display Function (No changes needed from previous version) ---
def display_final_results( transcript: str, content_json: str, metrics: Dict[str, Any], generation_time: float ) -> None:
    """Displays the final results with updated BERT interpretation and overall status logic."""
    # --- Header ---
    st.header(RESULTS_HEADER); st.divider()
    # --- Parse Content ---
    parsed_data = metrics.get("parsed_content"); summary_text = ""; flashcards = []
    is_valid_json_display = metrics.get("content_json_valid", False)
    if is_valid_json_display and parsed_data:
        summary_text = parsed_data.get("summary", "*Summary text not found in JSON*"); flashcards = parsed_data.get("flashcards", [])
        if not isinstance(flashcards, list): logger.warning("Parsed 'flashcards' not list."); flashcards = []
    elif not is_valid_json_display: summary_text = content_json; logger.info("Displaying raw content because JSON was invalid.")
    # --- Transcript Display ---
    st.subheader(TRANSCRIPT_HEADER); tran_len = len(transcript); tran_tok = utils.count_tokens(transcript)
    st.caption(f"{LENGTH_LABEL.format(tran_len)} | {TOKENS_LABEL.format(tran_tok) if tran_tok != -1 else 'Token error'}")
    with st.expander(VIEW_TRANSCRIPT_EXPANDER, expanded=False): st.text_area("Transcript_Text", transcript, height=300, key="transcript_display", disabled=True)
    st.divider()
    # --- Generated Content Display ---
    st.subheader(GENERATED_CONTENT_HEADER); col1, col2 = st.columns([2, 1])
    with col1: # Summary and Flashcards Display
        st.markdown(f"**{SUMMARY_SUBHEADER}**"); sum_len = len(summary_text); sum_tok = utils.count_tokens(summary_text)
        st.caption(f"{LENGTH_LABEL.format(sum_len)} | {TOKENS_LABEL.format(sum_tok) if sum_tok != -1 else 'Token error'}")
        st.markdown(summary_text if summary_text else "*Summary empty or generation failed.*")
        if is_valid_json_display:
            st.markdown(f"**{FLASHCARDS_SUBHEADER}**"); st.caption(FLASHCARD_COUNT_LABEL.format(count=len(flashcards)))
            if flashcards:
                for i, card in enumerate(flashcards):
                    q = card.get("question", f"Card {i+1}: Q missing"); a = card.get("answer", f"Card {i+1}: A missing")
                    with st.expander(f"Flashcard {i+1}: {q[:50]}..."): st.markdown(f"**Q:** {q}"); st.markdown(f"**A:** {a}")
            else: st.info(NO_FLASHCARDS_GENERATED)
    with col2: # Raw JSON Output Display
         st.markdown(f"**Raw Output**");
         with st.expander(VIEW_GENERATED_JSON_EXPANDER, expanded=False):
            if is_valid_json_display and parsed_data: st.json(parsed_data, expanded=False)
            else: st.warning(f"{INFO_ICON} Output not valid JSON."); st.text_area("Generated_Raw_Output", content_json, height=300, key="generated_raw_display", disabled=True)
    st.divider()
    # --- Evaluation Metrics Display ---
    st.subheader(EVALUATION_HEADER); st.metric(GENERATION_TIME_LABEL, f"{generation_time:.2f} s"); st.divider()
    st.markdown(f"**{METRICS_AUTO_HEADER}**"); col_auto1, col_auto2 = st.columns(2)
    with col_auto1: json_status = f"{PASS_ICON} {JSON_VALID_LABEL}" if is_valid_json_display else f"{FAIL_ICON} {JSON_INVALID_LABEL}"; st.metric(JSON_VALIDATION_LABEL, json_status)
    with col_auto2:
        bert_score_val = metrics.get("summary_bert_score", 0.0)
        if bert_score_val >= BERT_SCORE_GREEN_THRESHOLD: color = "green"
        elif bert_score_val >= BERT_SCORE_YELLOW_THRESHOLD: color = "orange"
        else: color = "red"
        st.markdown(f"**{BERT_LABEL}**: <span style='color:{color};'>{bert_score_val:.3f}</span>", unsafe_allow_html=True); st.caption(BERT_HELP_TEXT)
    st.divider()
    st.markdown(f"**{METRICS_AI_HEADER}**"); ai_results = metrics.get("ai_judge_results"); ai_judge_skipped = metrics.get("ai_judge_skipped", True)
    if ai_judge_skipped:
        if not is_valid_json_display: st.info(f"{INFO_ICON} AI Judge skipped (JSON invalid).")
        elif is_valid_json_display and not summary_text: st.info(f"{INFO_ICON} AI Judge skipped (Summary text empty).") # Corrected check
        else: st.info(f"{INFO_ICON} {INFO_AI_JUDGE_SKIPPED}")
    elif ai_results:
        if ai_results.get("error"): st.error(f"{FAIL_ICON} AI Judge evaluation failed: {ai_results['error']}")
        if "raw_response" in ai_results: 
            with st.expander("View Raw AI Judge Error Response"): st.code(ai_results["raw_response"], language="json")
        else:
            col_ai1, col_ai2, col_ai3 = st.columns(3); threshold = AI_JUDGE_THRESHOLD
            with col_ai1: ov_score = ai_results.get('overall_score', 0); ov_pass = ai_results.get('overall_pass', False); ov_pass_icon = f"{PASS_ICON}" if ov_pass else f"{WARN_ICON}"; st.metric(AI_OVERALL_LABEL, f"{ov_score}{SCORE_SUFFIX}", ov_pass_icon)
            with col_ai2: rel_score = ai_results.get('relevance_score', 0); rel_pass = ai_results.get('relevance_pass', False); rel_pass_icon = f"{PASS_ICON}" if rel_pass else f"{WARN_ICON}"; st.metric(AI_RELEVANCE_LABEL, f"{rel_score}{SCORE_SUFFIX}", rel_pass_icon)
            with col_ai3: flu_score = ai_results.get('fluency_score', 0); flu_pass = ai_results.get('fluency_pass', False); flu_pass_icon = f"{PASS_ICON}" if flu_pass else f"{WARN_ICON}"; st.metric(AI_FLUENCY_LABEL, f"{flu_score}{SCORE_SUFFIX}", flu_pass_icon)
            critique = ai_results.get("critique", "No critique provided.")
            with st.expander(VIEW_AI_JUDGE_CRITIQUE_EXPANDER): st.markdown(critique)
            # Overall Status Indicator Logic
            ai_judge_ran_successfully = not ai_judge_skipped and ai_results and not ai_results.get("error")
            ai_judge_passed_all_metrics = ai_judge_ran_successfully and ( ai_results.get('overall_pass', False) and ai_results.get('relevance_pass', False) and ai_results.get('fluency_pass', False) )
            if is_valid_json_display and ai_judge_passed_all_metrics: st.success(f"{PASS_ICON} {SUCCESS_MESSAGE}")
            elif is_valid_json_display and ai_judge_ran_successfully and not ai_judge_passed_all_metrics: st.warning(f"{WARN_ICON} JSON valid, but AI Judge metrics below threshold.")
            else: st.warning(f"{WARN_ICON} {WARNING_MESSAGE}")
    else: st.error("Internal error: AI judge results missing.")
    st.divider()
    with st.expander(VIEW_DETAILS_EXPANDER): st.json(metrics)

# --- Main Streamlit App Function ---
def main():
    """Sets up and runs the Streamlit application."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(APP_TITLE); st.caption(APP_SUBHEADER); st.info(DEMO_SCOPE_INFO, icon=INFO_ICON)

    # Initialize session state for results and feedback submission status
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'current_url' not in st.session_state:
         st.session_state.current_url = ""

    # Get user input
    url = st.text_input(URL_INPUT_LABEL, placeholder=URL_PLACEHOLDER, key="youtube_url_input")

    # --- Logic to run pipeline ---
    # Run only if URL is entered AND it's different from the last processed URL OR no results yet
    if url and (url != st.session_state.current_url or st.session_state.pipeline_results is None):
        logger.info(f"Processing new URL: {url}")
        st.session_state.current_url = url
        st.session_state.pipeline_results = None # Clear previous results
        st.session_state.feedback_submitted = False # Reset feedback status

        with st.status(STATUS_IN_PROGRESS, expanded=True) as status_ui:
            # Call the synchronous pipeline function
            pipeline_output: Optional[Tuple[str, str, Dict, float]] = run_pipeline(url, status_ui.update)

            if pipeline_output:
                # Store results in session state on success
                st.session_state.pipeline_results = {
                    "transcript_raw": pipeline_output[0],
                    "content_json": pipeline_output[1],
                    "final_metrics": pipeline_output[2],
                    "generation_time": pipeline_output[3],
                    "input_url": url # Store the URL associated with these results
                }
                status_ui.update(label=STATUS_ALL_COMPLETE, state="complete", expanded=False)
                logger.info(f"Pipeline execution successful for URL: {url}")
                # Force rerun to display results and feedback form
                st.rerun()
            else:
                # Error occurred, message already shown via status_ui.update(state="error") and st.error() in run_pipeline
                logger.warning(f"Pipeline execution failed for URL: {url}")
                # Keep pipeline_results as None

    # --- Display Results and Feedback Form (if results exist in session state) ---
    if st.session_state.pipeline_results:
        # Retrieve results from session state
        results_data = st.session_state.pipeline_results
        # Check if the results correspond to the currently entered URL
        if results_data["input_url"] == url:
            # Call the display function with data from session state
            display_final_results(
                results_data["transcript_raw"],
                results_data["content_json"],
                results_data["final_metrics"],
                results_data["generation_time"]
            )

            # --- Add HITL Feedback Section ---
            st.divider()
            st.subheader("📊 Rate This Result")

            # Only show rating form if feedback hasn't been submitted for this result
            if not st.session_state.feedback_submitted:
                rating = st.radio(
                    "Overall Quality (Summary & Flashcards):",
                    options=["1", "2", "3", "4", "5"],
                    index=None, # Default to no selection
                    horizontal=True,
                    key="hitl_rating_radio"
                )
                feedback_text = st.text_area(
                    "Optional Feedback (e.g., What was good/bad?):",
                    key="hitl_feedback_text"
                )
                submit_button = st.button("Submit Feedback", key="hitl_submit_button")

                if submit_button:
                    if rating is None:
                        st.warning("Please select a rating (1-5) before submitting.")
                    else:
                        # Prepare data to save
                        feedback_data_to_save = {
                            "input_url": results_data["input_url"],
                            "video_id": results_data["final_metrics"].get("video_id"),
                            "run_timestamp": results_data["final_metrics"].get("timestamp"),
                            "generation_model": results_data["final_metrics"].get("generation_model"),
                            "content_json_valid": results_data["final_metrics"].get("content_json_valid"),
                            "summary_bert_score": results_data["final_metrics"].get("summary_bert_score"),
                            "ai_judge_results": results_data["final_metrics"].get("ai_judge_results"),
                            "user_rating": int(rating), # Convert rating to int
                            "user_feedback_text": feedback_text.strip() if feedback_text else None,
                            # Optionally include raw content/transcript if needed for analysis later
                            # "raw_content_json": results_data["content_json"],
                            # "raw_transcript": results_data["transcript_raw"]
                        }

                        # Save the feedback
                        if save_feedback(feedback_data_to_save):
                            st.success("Thank you for your feedback!")
                            st.session_state.feedback_submitted = True # Mark as submitted
                            # Rerun to potentially hide the form or show thank you message clearly
                            st.rerun()
                        # else: Error message handled by save_feedback
            else:
                # Show message if feedback was already submitted for this run
                st.success("Feedback submitted for this result. Thank you!")
        else:
            # This case happens if user inputs a new URL but old results are still in state
            # before the pipeline runs for the new URL. Clear old state.
            st.session_state.pipeline_results = None
            st.session_state.feedback_submitted = False
            st.session_state.current_url = url # Ensure current URL is updated
            # Rerun might be needed if you want to immediately clear display
            # st.rerun()


if __name__ == "__main__":
    main()