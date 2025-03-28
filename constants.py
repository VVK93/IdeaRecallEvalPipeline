# constants.py
import os

# --- Core Configuration ---
# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

# Model Selection (Allows easy changes)
GENERATION_MODEL = "gpt-4-turbo"
AI_JUDGE_MODEL = "gpt-4-turbo" # Note: Article may mention Gemini; this demo uses GPT-4 Turbo.

# Tokenizer Model (Should align with generation model)
TOKENIZER_MODEL = "gpt-4"

# Token Limits & Chunking Strategy
# Adjust based on the specific variant of GENERATION_MODEL if needed
MAX_MODEL_TOKENS_SUMMARY = 120000 # Max context window for the generation model
SUMMARY_PROMPT_OVERHEAD = 1000 # Estimated tokens used by system/user prompts & instructions
# Target size for each chunk in Map-Reduce (leaves buffer for overhead)
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500 # Tokens to overlap between chunks for better context flow

# Evaluation Thresholds
BERT_SCORE_THRESHOLD = 0.80 # Minimum acceptable BERT F1 score
AI_JUDGE_THRESHOLD = 80 # Minimum acceptable score (out of 100) for AI Judge metrics

# --- UI Text: General App ---
PAGE_TITLE = "Idea Recall Demo"
PAGE_ICON = "🛠️" # Changed icon to reflect demo tool nature
APP_TITLE = "🛠️ Idea Recall: YouTube Summarizer & Evaluator Demo"
APP_SUBHEADER = "A tool demonstrating core AI steps from the Idea Recall project."
URL_INPUT_LABEL = "YouTube Video URL"
URL_PLACEHOLDER = "e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# --- UI Text: Demo Scope Info ---
DEMO_SCOPE_INFO = """
**About This Demo Tool**

This Streamlit application demonstrates key components of the AI pipeline for **Idea Recall**, the YouTube learning assistant described in my portfolio article.

**Functionality Shown:**
- Fetching YouTube transcripts.
- Generating summaries using an LLM (with token-based chunking via Map-Reduce for long videos).
- **Evaluation Pipeline Stages:**
    - Automatic Check: JSON structure validation.
    - Automatic Check: BERTScore calculation (semantic similarity).
    - AI Judge: Using an LLM (`{ai_judge_model}`) to assess summary quality.

**Relation to Portfolio Article & Full Product Vision:**
- This tool showcases the *technical implementation* of core steps.
- The full **Idea Recall** product is envisioned as a **Telegram bot**.
- **Evaluation Differences:** The complete pipeline includes ROUGE/SBERT scores (evaluated offline) and crucial **Human-in-the-Loop feedback** (via Telegram), which are **not** part of this interactive demo.
- **Model Note:** While the article discusses using Gemini Flash 1.5 for cost-efficiency, this demo uses `{ai_judge_model}` for evaluation.
""".format(ai_judge_model=AI_JUDGE_MODEL) # Dynamically insert model name

# --- UI Text: Results Section ---
RESULTS_HEADER = "📊 Results"
TRANSCRIPT_HEADER = "📜 Original Transcript"
SUMMARY_HEADER = "📝 Generated Summary (JSON)" # Clarified format
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
METRICS_AI_HEADER = "🧑‍⚖️ AI Judge Evaluation (`{ai_judge_model}`)".format(ai_judge_model=AI_JUDGE_MODEL) # Show judge model
JSON_VALIDATION_LABEL = "JSON Structure"
JSON_VALID_LABEL = "Valid"
JSON_INVALID_LABEL = "Invalid"
BERT_LABEL = "BERT Score (F1)"
BERT_THRESHOLD_NOTE = f"(Threshold ≥ {BERT_SCORE_THRESHOLD:.2f})" # Use constant
AI_OVERALL_LABEL = "Overall Score"
AI_RELEVANCE_LABEL = "Relevance Score"
AI_FLUENCY_LABEL = "Fluency Score"
SCORE_SUFFIX = "/ 100"

# --- UI Text: Icons & Status Messages ---
PASS_ICON = "✅"
FAIL_ICON = "❌"
WARN_ICON = "⚠️"
INFO_ICON = "ℹ️"
SUCCESS_MESSAGE = "All primary evaluation checks passed successfully!"
WARNING_MESSAGE = "Some evaluation checks did not meet thresholds. Review metrics."
INFO_AI_JUDGE_SKIPPED = f"AI Judge evaluation skipped (BERT score < {BERT_SCORE_THRESHOLD:.2f} or JSON invalid)."
INFO_AI_JUDGE_NO_RESULTS = "AI Judge ran, but no valid results were parsed."

# --- UI Text: Status Updates (for st.status) ---
STATUS_IN_PROGRESS = "Processing video..."
STATUS_STEP_1 = "Step 1/4: Fetching transcript..."
STATUS_STEP_1_COMPLETE = "Step 1/4: Transcript fetched ({tran_len:,} chars, {tran_tok:,} tokens)."
STATUS_STEP_2_SINGLE = "Step 2/4: Generating summary (single call)..."
STATUS_STEP_2_CHUNK = "Step 2/4: Generating summary (Map-Reduce: {num_chunks} chunks)..."
STATUS_STEP_2_MAP = "Step 2/4: Generating summary (Map: Chunk {i}/{n})..." # For Map phase
STATUS_STEP_2_REDUCE = "Step 2/4: Generating summary (Reduce phase)..." # For Reduce phase
STATUS_STEP_2_COMPLETE = "Step 2/4: Summary generated ({gen_time:.2f}s, {sum_len:,} chars, {sum_tok:,} tokens)."
STATUS_STEP_3 = "Step 3/4: Running automatic evaluation (JSON & BERT)..."
STATUS_STEP_3_COMPLETE = "Step 3/4: Automatic evaluation complete (BERT: {bert_score:.3f})."
STATUS_STEP_4 = "Step 4/4: Running AI Judge evaluation..."
STATUS_STEP_4_COMPLETE = "Step 4/4: AI Judge evaluation complete."
STATUS_STEP_4_SKIPPED = "Step 4/4: AI Judge evaluation skipped."
STATUS_ALL_COMPLETE = "Processing complete!"
STATUS_ERROR = "An error occurred: {}"

# --- Error Messages ---
ERROR_API_KEY_MISSING = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
ERROR_OPENAI_API = "OpenAI API error: {}"
ERROR_OPENAI_INIT = "Failed to initialize OpenAI client. Check API key and network. Error: {}"
ERROR_TOKENIZER_INIT = "Failed to initialize tokenizer for model '{}'. Error: {}"
ERROR_INVALID_URL = "Invalid YouTube URL provided."
ERROR_TRANSCRIPT_DISABLED = "Transcripts are disabled for this video."
ERROR_NO_TRANSCRIPT = "No transcript found for this video. Might be unavailable/unsupported language."
ERROR_FETCH_TRANSCRIPT = "Could not fetch transcript. Error: {}"
ERROR_TOKEN_COUNT = "Could not count tokens for text. Error: {}"
ERROR_GENERATE_SUMMARY_CHUNK = "Error processing chunk {i}/{n} for summary. Error: {}"
ERROR_GENERATE_SUMMARY_COMBINE = "Error combining chunk summaries. Error: {}"
ERROR_GENERATE_SUMMARY_GENERAL = "Could not generate summary. Error: {}"
ERROR_EVALUATE_BERT = "BERT score evaluation failed. Error: {}"
ERROR_EVALUATE_AI_JUDGE = "AI Judge evaluation call failed. Error: {}"
ERROR_AI_JUDGE_PARSE = "Failed to parse AI judge response or structure incorrect (Attempt {attempt}/{max_retries}). Error: {}"
ERROR_AI_JUDGE_RETRY_FAIL = "AI Judge evaluation failed after {max_retries} retries."
ERROR_UNKNOWN = "An unexpected error occurred: {}"

# --- Prompts for OpenAI ---

# Note: Ensuring the LLM strictly adheres to JSON output is crucial.
# Using response_format={"type": "json_object"} helps enforce this.

SYSTEM_PROMPT_SUMMARY = """You are an expert assistant tasked with summarizing video transcripts.
Generate a concise, informative summary focusing on the key topics, arguments, and conclusions presented in the transcript.
The output MUST be a JSON object containing ONLY a single key "summary" with the summary text as its value.
Example: {"summary": "The video discusses the importance of..."}"""

# Used for single calls or the Map step in Map-Reduce
USER_PROMPT_SUMMARY = """Please summarize the following transcript and provide the output strictly in the specified JSON format:

Transcript:
\"\"\"
{}
\"\"\""""

# Used for the Reduce step in Map-Reduce
SYSTEM_PROMPT_COMBINE = """You will receive multiple summary sections generated from different chunks of a longer video transcript.
Combine these sections into a single, coherent, and concise summary that flows well. Capture the overall narrative and key takeaways of the entire video.
The final output MUST be a JSON object containing ONLY a single key "summary" with the combined summary text as its value.
Example: {"summary": "This video initially covers topic A... It then transitions to topic B detailing... Finally, it concludes by emphasizing..."}"""

USER_PROMPT_COMBINE = """Please combine the following summary sections (from video chunks) into one cohesive JSON summary, adhering strictly to the specified format:

Summary Sections to Combine:
\"\"\"
{}
\"\"\""""

SYSTEM_PROMPT_AI_JUDGE = """You are an expert evaluator assessing the quality of a video transcript summary.
Analyze the provided summary (extracted from a JSON object) and evaluate it based on Overall Quality, Relevance (to the assumed original video content), and Fluency/Readability.
Provide scores from 0 to 100 for each category and a brief critique explaining your scores.
Your response MUST be a JSON object with the exact keys: "overall_score" (int), "relevance_score" (int), "fluency_score" (int), and "critique" (string).
Example: {"overall_score": 85, "relevance_score": 90, "fluency_score": 80, "critique": "Good summary, covers main points well. Could be slightly more concise."}"""

USER_PROMPT_AI_JUDGE = """Please evaluate the following summary based on the criteria (Overall Quality, Relevance, Fluency). Provide your evaluation strictly in the specified JSON format.

Summary to Evaluate:
\"\"\"
{}
\"\"\"""" # Pass the actual summary string here, not the JSON wrapper