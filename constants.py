# constants.py

# --- UI Text: General App ---
PAGE_TITLE = "Idea Recall"
PAGE_ICON = "🎥"
APP_TITLE = "🎥 Idea Recall: YouTube Summarizer & Evaluator"
APP_SUBHEADER = "Enter a YouTube URL to fetch, summarize, and evaluate its transcript."
URL_INPUT_LABEL = "YouTube Video URL"
URL_PLACEHOLDER = "e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# --- UI Text: Results Section ---
RESULTS_HEADER = "📊 Results"
TRANSCRIPT_HEADER = "📜 Original Transcript"
SUMMARY_HEADER = "📝 Generated Summary"
EVALUATION_HEADER = "⚖️ Evaluation Metrics"
VIEW_TRANSCRIPT_EXPANDER = "View Full Transcript"
VIEW_SUMMARY_EXPANDER = "View Generated JSON Summary"
VIEW_CRITIQUE_EXPANDER = "View AI Judge Critique"
VIEW_DETAILS_EXPANDER = "View Raw Evaluation Data"
LENGTH_LABEL = "Length: {:,} chars"
TOKENS_LABEL = "Tokens: {:,}"
GENERATION_TIME_LABEL = "Summary Generation Time"

# --- UI Text: Evaluation Metrics ---
METRICS_AUTO_HEADER = "🤖 Automatic Checks"
METRICS_AI_HEADER = "🧑‍⚖️ AI Judge Evaluation"
JSON_VALIDATION_LABEL = "JSON Structure"
JSON_VALID_LABEL = "Valid"
JSON_INVALID_LABEL = "Invalid"
BERT_LABEL = "BERT Score (F1)"
BERT_THRESHOLD_NOTE = "(Threshold ≥ 0.8)" # Ensure this matches BERT_SCORE_THRESHOLD
AI_OVERALL_LABEL = "Overall Score"
AI_RELEVANCE_LABEL = "Relevance Score"
AI_FLUENCY_LABEL = "Fluency Score"
SCORE_SUFFIX = "/ 100"

# --- UI Text: Icons & Status Messages ---
PASS_ICON = "✅"
FAIL_ICON = "❌"
WARN_ICON = "⚠️"
INFO_ICON = "ℹ️"
SUCCESS_MESSAGE = "All evaluation checks passed successfully!"
WARNING_MESSAGE = "Some evaluation checks did not pass. Review the metrics."
INFO_AI_JUDGE_SKIPPED = "AI Judge evaluation skipped due to failed automatic checks."
INFO_AI_JUDGE_NO_RESULTS = "AI Judge ran, but no results were returned or parsed correctly." # More descriptive

# --- UI Text: Status Updates (for st.status) ---
STATUS_IN_PROGRESS = "Processing video... buckle up!"
STATUS_STEP_1 = "Step 1/4: Fetching transcript..."
STATUS_STEP_1_COMPLETE = "Step 1/4: Transcript fetched ({:,} chars, {:,} tokens)."
STATUS_STEP_2 = "Step 2/4: Generating summary..."
STATUS_STEP_2_COMPLETE = "Step 2/4: Summary generated ({:.2f}s, {:,} chars, {:,} tokens)."
STATUS_STEP_3 = "Step 3/4: Running automatic evaluation (JSON & BERT)..."
STATUS_STEP_3_COMPLETE = "Step 3/4: Automatic evaluation complete."
STATUS_STEP_4 = "Step 4/4: Running AI Judge evaluation..."
STATUS_STEP_4_COMPLETE = "Step 4/4: AI Judge evaluation complete."
STATUS_STEP_4_SKIPPED = "Step 4/4: AI Judge evaluation skipped (Auto checks failed)."
STATUS_ALL_COMPLETE = "Processing complete!"
STATUS_ERROR = "An error occurred: {}"

# --- Error Messages ---
ERROR_API_KEY_MISSING = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
ERROR_OPENAI_API = "OpenAI API error: {}"
ERROR_INVALID_URL = "Invalid YouTube URL provided."
ERROR_TRANSCRIPT_DISABLED = "Transcripts are disabled for this video."
ERROR_NO_TRANSCRIPT = "No transcript found for this video. It might be unavailable or in an unsupported language."
ERROR_FETCH_TRANSCRIPT = "Could not fetch transcript. Error: {}"
ERROR_GENERATE_SUMMARY = "Could not generate summary. Error: {}"
ERROR_EVALUATE_BERT = "BERT score evaluation failed. Error: {}" # Used by standalone BERT func if kept
ERROR_UNKNOWN = "An unexpected error occurred: {}"

# --- Prompts for OpenAI ---
# Make sure these prompts align with your desired output format and model capabilities.
# Using JSON mode where applicable simplifies things.

SYSTEM_PROMPT_SUMMARY = """You are an expert assistant tasked with summarizing video transcripts.
Generate a concise, informative summary focusing on the key topics, arguments, and conclusions presented in the transcript.
The output MUST be a JSON object containing a single key "summary" with the summary text as its value.
Example: {"summary": "The video discusses..."}"""

USER_PROMPT_SUMMARY = """Please summarize the following transcript and provide the output strictly in the specified JSON format:

Transcript:
\"\"\"
{}
\"\"\""""

SYSTEM_PROMPT_COMBINE = """You will receive multiple summary sections, potentially from different parts of a video transcript.
Combine these sections into a single, coherent, and concise summary.
The final output MUST be a JSON object containing a single key "summary" with the combined summary text as its value.
Example: {"summary": "This video covers topic A, explaining... It then moves on to topic B..."}"""

USER_PROMPT_COMBINE = """Please combine the following summary sections into one cohesive JSON summary, following the specified format:

Sections:
\"\"\"
{}
\"\"\""""

SYSTEM_PROMPT_AI_JUDGE = """You are an expert evaluator assessing the quality of a video transcript summary.
Analyze the provided summary and evaluate it based on Overall Quality, Relevance (to an assumed source transcript), and Fluency/Readability.
Provide scores from 0 to 100 for each category and a brief critique.
Your response MUST be a JSON object with the keys "overall_score", "relevance_score", "fluency_score", and "critique".
Example: {"overall_score": 85, "relevance_score": 90, "fluency_score": 80, "critique": "Good summary, covers main points well. Could be slightly more concise."}"""

USER_PROMPT_AI_JUDGE = """Please evaluate the following summary based on the criteria (Overall Quality, Relevance, Fluency). Provide your evaluation strictly in the specified JSON format.

Summary to Evaluate:
```json
{}
```"""


# --- Numerical Thresholds ---
BERT_SCORE_THRESHOLD = 0.8
AI_JUDGE_THRESHOLD = 80 # Threshold for individual AI judge scores to be considered "passing"