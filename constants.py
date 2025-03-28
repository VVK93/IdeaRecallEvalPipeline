# constants.py
import os

# --- Core Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4-turbo" # Or your preferred model
AI_JUDGE_MODEL = "gpt-4-turbo"   # Can be same or different
TOKENIZER_MODEL = "gpt-4"        # Should align with generation model

# Token Limits & Chunking Strategy
MAX_MODEL_TOKENS_SUMMARY = 120000 # Max context for generation model
SUMMARY_PROMPT_OVERHEAD = 1600    # Estimated tokens for prompts/instructions
CHUNK_SIZE = 8000                 # Target token size for Map-Reduce chunks
CHUNK_OVERLAP = 500               # Token overlap between chunks

# --- Evaluation Thresholds & Indicators ---
# BERT Score thresholds for UI color-coding
BERT_SCORE_GREEN_THRESHOLD = 0.70
BERT_SCORE_YELLOW_THRESHOLD = 0.50 # Threshold for Yellow/Red split

# BERT Score internal pass/fail threshold (Fail if score < 0.50)
BERT_SCORE_FAIL_THRESHOLD = BERT_SCORE_YELLOW_THRESHOLD

# AI Judge threshold for individual metric scores (e.g., overall, relevance, fluency)
AI_JUDGE_THRESHOLD = 80 # Score out of 100

# --- UI Text: General App ---
PAGE_TITLE = "Idea Recall Demo"
PAGE_ICON = "💡"
APP_TITLE = "💡 Idea Recall: YouTube Summarizer & Flashcard Demo"
APP_SUBHEADER = "Generates a summary and flashcards from YouTube videos, with evaluation."
URL_INPUT_LABEL = "YouTube Video URL"
URL_PLACEHOLDER = "e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# --- UI Text: Demo Scope Info ---
# It's important this text accurately reflects the demo's scope and limitations
DEMO_SCOPE_INFO = """
**About This Demo Tool**

This Streamlit application demonstrates key components of the **Idea Recall** AI pipeline, focusing on generating both a summary and flashcards from YouTube transcripts, followed by evaluation of the summary.

**Functionality Shown:**
- Fetching and basic cleaning of YouTube transcripts.
- Generating **summary & flashcards** using an LLM (`{generation_model}`). Handles long videos via Map-Reduce.
- **Evaluation Pipeline Stages (Applied Primarily to Summary Text):**
    - Automatic Check: JSON structure validation of the entire output.
    - Automatic Check: **BERT Score** calculation (comparing summary to cleaned transcript), displayed as a semantic overlap indicator with color-coding.
    - AI Judge: Using an LLM (`{ai_judge_model}`) to assess **summary quality** (Overall, Relevance, Fluency), run if JSON is valid and summary text exists.

**Relation to Portfolio Article & Full Product Vision:**
- This tool showcases the *technical implementation* of core generation and evaluation steps.
- The full **Idea Recall** product is envisioned as a **Telegram bot** with user interaction for recall practice.
- **Evaluation Differences:** This demo evaluates only the *summary text* with BERT/AI Judge. The full pipeline described in the article includes ROUGE/SBERT (offline/separate tooling) and crucial **Human-in-the-Loop feedback** (via Telegram) for both summary and flashcards. Flashcard-specific automated evaluation is not implemented here.
- **BERT Score Interpretation:** BERT score is presented as an indicator of semantic overlap. Due to the abstractive nature of summaries, lower scores (e.g., in the Yellow range) may still correspond to high-quality summaries if AI Judge scores and/or human feedback are positive.
- **Model Note:** Demo uses `{generation_model}`/`{ai_judge_model}`. Production might use different models (e.g., Gemini) for cost/performance optimization.
""".format(generation_model=GENERATION_MODEL, ai_judge_model=AI_JUDGE_MODEL)

# --- UI Text: Results Section ---
RESULTS_HEADER = "📊 Results"
TRANSCRIPT_HEADER = "📜 Original Transcript"
GENERATED_CONTENT_HEADER = "📝 Generated Content"
SUMMARY_SUBHEADER = "Summary"
FLASHCARDS_SUBHEADER = "Flashcards"
EVALUATION_HEADER = "⚖️ Evaluation Metrics (Summary)"
VIEW_TRANSCRIPT_EXPANDER = "View Full Transcript"
VIEW_GENERATED_JSON_EXPANDER = "View Raw Generated JSON"
VIEW_AI_JUDGE_CRITIQUE_EXPANDER = "View AI Judge Critique (Summary)"
VIEW_DETAILS_EXPANDER = "View Raw Evaluation Data"
LENGTH_LABEL = "Length: {:,} chars"
TOKENS_LABEL = "Tokens: {:,}"
GENERATION_TIME_LABEL = "Content Generation Time"
FLASHCARD_COUNT_LABEL = "{count} Flashcards Generated"
NO_FLASHCARDS_GENERATED = "No flashcards were generated or found in the output."

# --- UI Text: Evaluation Metrics ---
METRICS_AUTO_HEADER = "🤖 Automatic Checks"
METRICS_AI_HEADER = "🧑‍⚖️ AI Judge Evaluation (`{ai_judge_model}`) - Summary".format(ai_judge_model=AI_JUDGE_MODEL)
JSON_VALIDATION_LABEL = "JSON Structure"
JSON_VALID_LABEL = "Valid"
JSON_INVALID_LABEL = "Invalid"
BERT_LABEL = "BERT Score (Semantic Overlap)"
BERT_HELP_TEXT = f"""Measures semantic similarity between summary and transcript. Higher scores mean more overlap.
- Green (> {BERT_SCORE_GREEN_THRESHOLD:.2f}): High overlap
- Yellow ({BERT_SCORE_YELLOW_THRESHOLD:.2f} - {BERT_SCORE_GREEN_THRESHOLD:.2f}): Moderate overlap
- Red (< {BERT_SCORE_YELLOW_THRESHOLD:.2f}): Low overlap (Indicates potential issues)
*Note: Abstractive summaries can be high quality even with lower scores.*"""
AI_OVERALL_LABEL = "Overall Score"; AI_RELEVANCE_LABEL = "Relevance Score"; AI_FLUENCY_LABEL = "Fluency Score"; SCORE_SUFFIX = "/ 100"

# --- UI Text: Icons & Status Messages ---
PASS_ICON = "✅"; FAIL_ICON = "❌"; WARN_ICON = "⚠️"; INFO_ICON = "ℹ️"
SUCCESS_MESSAGE = "JSON valid & AI Judge metrics met thresholds!"
WARNING_MESSAGE = "Check failed: Review JSON validity or AI Judge metrics/critique."
INFO_AI_JUDGE_SKIPPED = "AI Judge evaluation skipped (JSON invalid or Summary text empty)."
INFO_AI_JUDGE_NO_RESULTS = "AI Judge ran for summary, but no valid results were parsed."

# --- UI Text: Status Updates ---
STATUS_IN_PROGRESS = "Processing video..."
STATUS_STEP_1 = "Step 1/4: Fetching & Cleaning Transcript..."
STATUS_STEP_1_COMPLETE = "Step 1/4: Transcript Processed ({tran_len:,} chars, {tran_tok:,} tokens)."
STATUS_STEP_2_SINGLE = "Step 2/4: Generating content (single call)..."
STATUS_STEP_2_CHUNK = "Step 2/4: Generating content (Map-Reduce: {num_chunks} chunks)..."
STATUS_STEP_2_MAP = "Step 2/4: Generating content (Map: Chunk {i}/{n})..."
STATUS_STEP_2_REDUCE = "Step 2/4: Generating content (Reduce phase)..."
STATUS_STEP_2_COMPLETE = "Step 2/4: Content generated ({gen_time:.2f}s)."
STATUS_STEP_3 = "Step 3/4: Running automatic evaluation (JSON & BERT)..."
STATUS_STEP_3_COMPLETE = "Step 3/4: Automatic evaluation complete (BERT: {bert_score:.3f})."
STATUS_STEP_4 = "Step 4/4: Running AI Judge evaluation (Summary)..."
STATUS_STEP_4_COMPLETE = "Step 4/4: AI Judge evaluation complete."
STATUS_STEP_4_SKIPPED = "Step 4/4: AI Judge evaluation skipped."
STATUS_ALL_COMPLETE = "Processing complete!"
STATUS_ERROR = "An error occurred: {}"

# --- Error Messages ---
ERROR_API_KEY_MISSING = "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
ERROR_OPENAI_API = "OpenAI API error: {}"
ERROR_OPENAI_INIT = "Failed to initialize OpenAI client. Check API key/network. Error: {}"
ERROR_TOKENIZER_INIT = "Failed to initialize tokenizer for model '{}'. Error: {}"
ERROR_INVALID_URL = "Invalid YouTube URL provided."
ERROR_TRANSCRIPT_DISABLED = "Transcripts are disabled for this video."
ERROR_NO_TRANSCRIPT = "No transcript found (unavailable/unsupported language)."
ERROR_FETCH_TRANSCRIPT = "Could not fetch transcript. Error: {}"
ERROR_TOKEN_COUNT = "Could not count tokens. Error: {}"
ERROR_GENERATE_CONTENT_CHUNK = "Error processing chunk {i}/{n}. Error: {}"
ERROR_GENERATE_CONTENT_COMBINE = "Error combining chunk content. Error: {}"
ERROR_GENERATE_CONTENT_GENERAL = "Could not generate content. Error: {}"
ERROR_EVALUATE_BERT = "BERT score evaluation failed. Error: {}"
ERROR_EVALUATE_AI_JUDGE = "AI Judge evaluation call failed. Error: {}"
ERROR_AI_JUDGE_PARSE = "Failed to parse AI judge response (Attempt {attempt}/{max_retries}). Error: {}"
ERROR_AI_JUDGE_RETRY_FAIL = "AI Judge evaluation failed after {max_retries} retries."
ERROR_EXTRACT_CONTENT = "Failed to extract 'summary' or 'flashcards' field from generated JSON."
ERROR_ASYNC_EVAL_LOOP = "Failed to run async evaluations due to loop conflict. Check nest_asyncio."
ERROR_UNKNOWN = "An unexpected error occurred: {}"

# --- Prompts for OpenAI ---
OUTPUT_JSON_STRUCTURE_DESCRIPTION = """
The output MUST be a valid JSON object containing exactly two keys: "summary" and "flashcards".
- "summary": A string containing the comprehensive summary text.
- "flashcards": A JSON array of objects. Each object MUST have two keys: "question" (string) and "answer" (string).

Example Format:
{
  "summary": "The video explains concept X, detailing A, B, and C. It compares X to Y and concludes Z.",
  "flashcards": [
    {"question": "What is the main definition of X?", "answer": "X is defined as..."},
    {"question": "How does X compare to Y according to the video?", "answer": "X is faster/cheaper/different than Y because..."},
    {"question": "What was the main conclusion regarding Z?", "answer": "The main conclusion was that Z leads to..."}
  ]
}
"""

SYSTEM_PROMPT_GENERATION = f"""You are an expert content analyzer that extracts ALL distinct ideas from content.
For videos with numbered tips or steps, you MUST create a separate idea for each tip/step.
ALWAYS respond with valid JSON matching the exact structure specified.
Each idea MUST have:
- Catchy, memorable title (max 60 chars)
- overview: Concise explanation (100-200 words)
- quote: 1-2 relevant quotes from the content
- why_it_matters: Brief explanation of importance
- how_to_apply: Practical steps for application

2.  A set of **flashcards** (5-10) covering distinct main points from the transcript. Questions should test understanding of core concepts.

{OUTPUT_JSON_STRUCTURE_DESCRIPTION}
Adhere STRICTLY to the JSON format. Focus the summary on transcript fidelity.
"""

USER_PROMPT_GENERATION = """Please analyze this content and provide:
1. A summary of what the video is about that includes all key ideas and concepts mentioned in the video with actionable insights and quotes.
2. 5-10 meaningful flashcards 

For each insight, include:
- A clear, specific title for the insight
- The content of the insight explained in details with quotes and examples
- adhere strictly to the JSON format:
Transcript:
\"\"\"
{}
\"\"\""""

SYSTEM_PROMPT_COMBINE = f"""You will receive multiple JSON objects (from video chunks), each with "summary" and "flashcards". Synthesize these into a single, final JSON object:
1. **Combine Summaries:** Create **one cohesive, detailed, comprehensive final summary** integrating information smoothly. Cover **all major topics, arguments, comparisons, limitations, specific examples (e.g., model names '01', cost), evaluation insights, and future outlooks** discussed across chunks. Reflect key terminology accurately. Maintain logical flow.
2. **Consolidate Flashcards:** Review all incoming flashcards. Create a **final set** (typically 5-15) covering the *entire* video's main concepts. **Prioritize distinct key points.** Avoid redundancy - choose the clearest version or merge insights. Ensure final questions test understanding, are balanced, and relate to the overall content.

{OUTPUT_JSON_STRUCTURE_DESCRIPTION}
Analyze the input JSON objects below and generate the final, consolidated JSON output STRICTLY in the specified format.
"""

USER_PROMPT_COMBINE = """Synthesize the content from the following JSON objects into one final JSON object (combined summary, consolidated flashcards). Follow the specified output format strictly.
Input JSON Objects:
```json
{}
```"""

SYSTEM_PROMPT_AI_JUDGE = """You are an expert evaluator assessing video transcript **summary text** quality.
Analyze the provided summary **text** for Overall Quality, Relevance (to assumed source), and Fluency/Readability.
Provide scores (0-100) and a brief critique.
Response MUST be a JSON object with keys: "overall_score"(int), "relevance_score"(int), "fluency_score"(int), "critique"(string).
Example: {"overall_score": 85, "relevance_score": 90, "fluency_score": 80, "critique": "Good summary, covers main points. Could be slightly more concise."}"""

USER_PROMPT_AI_JUDGE = """Evaluate the following **summary text** based on Overall Quality, Relevance, Fluency. Provide evaluation strictly in the specified JSON format.
Summary Text to Evaluate:
\"\"\"
{}
\"\"\""""