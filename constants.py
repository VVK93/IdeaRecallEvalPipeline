# App title and description
TITLE = "YouTube Video Summarizer"
DESCRIPTION = "Enter a YouTube video URL to get a summary of its content."
INPUT_LABEL = "YouTube Video URL"

# Status messages
STATUS_FETCHING_TRANSCRIPT = "Step 1/4: Retrieving video transcript..."
STATUS_TRANSCRIPT_SUCCESS = "✅ Transcript retrieved successfully!"
STATUS_GENERATING_SUMMARY = "Step 2/4: Generating summary..."
STATUS_SUMMARY_SUCCESS = "✅ Summary generated successfully!"
STATUS_EVALUATING = "Step 3/4: Running evaluation metrics..."
STATUS_ALL_DONE = "✅ All automatic checks passed!"
STATUS_WARNING = "⚠️ Some automatic checks did not pass. See metrics above for details."

# Transcript info
TRANSCRIPT_INFO = "Transcript length: {} characters"

# Summary header
SUMMARY_HEADER = "### Summary (Generated in {}s)"

# Metrics
METRICS_HEADER = "Evaluation Metrics"
DETAILED_METRICS_HEADER = "📊 Detailed Metrics"
BERT_LABEL = "BERT Score"
BERT_THRESHOLD = "(Threshold: 0.8)"
GENERATION_TIME_LABEL = "Generation Time"

# Error messages
ERROR_INVALID_URL = "Invalid YouTube URL. Please enter a valid YouTube video URL."
ERROR_TRANSCRIPT_DISABLED = "This video has disabled transcripts."
ERROR_NO_TRANSCRIPT = "No transcript found for this video."
ERROR_FETCH_TRANSCRIPT = "Failed to get transcript: {}"
ERROR_GENERATE_SUMMARY = "Failed to generate summary: {}"
ERROR_EVALUATE_BERT = "BERT score evaluation failed: {}"
ERROR_EVALUATION = "Evaluation failed: {}"

# OpenAI Prompts
SYSTEM_PROMPT_SUMMARY = "You are a helpful assistant that creates concise summaries of YouTube video transcripts in JSON format."
USER_PROMPT_SUMMARY = """You are an expert at extracting actionable insights from educational content.

The transcript of the video is below:
'''
{}
'''

Please analyze this content and provide:
1. A summary of what the video is about, including most significant key points and themes and numbers or dates mentioned in the all video.
2. Most relevant and useful insights from the video

Each insight MUST haveand be nicely formmated for telegram using emojies, bullet points, and bold text for the summary and insights:
- name: Catchy, memorable title (max 60 chars)
- subject: 1-2 word topic area (e.g., "Productivity", "Psychology", "Health")
- overview: Concise explanation (100-200 words)
- quote: 1-2 relevant quotes from the content
- why it matters: Brief explanation of importance
- how to apply: Practical steps for application

Format your response as a valid JSON with the following structure:
{{
  "summary": "Concise summary of the video",
  "ideas": [
    {{
      "title": "Clear title for insight 1",
      "content": "Detailed explanation of insight 1",
      "timestamp": "MM:SS",
      "topic": "Category"
    }},
    ...
  ]
}}

Be sure your response is ONLY valid JSON with no additional text or explanations before or after."""
SYSTEM_PROMPT_COMBINE = "You are a helpful assistant that combines summaries into one coherent JSON summary."
USER_PROMPT_COMBINE = "Please combine these summaries into one coherent JSON summary:\n\n{}"

# Evaluation thresholds
BERT_THRESHOLD_VALUE = 0.8