# 🛠️ Idea Recall - YouTube Summarizer & Evaluator Demo

This project is a Streamlit web application demonstrating core components of the **Idea Recall** AI pipeline, designed to summarize YouTube video transcripts and evaluate the quality of the generated summaries. It serves as a technical showcase for the concepts discussed in the associated portfolio article about building human-centered AI products.

**Note:** This application is a **demonstration tool** and differs in scope and interface from the final envisioned "Idea Recall" product (a Telegram bot).

## Overview

The application allows a user to:

1.  Enter a YouTube video URL.
2.  Automatically fetch the video's transcript.
3.  Generate a concise summary using an OpenAI LLM (currently configured for `gpt-4-turbo`). It employs a Map-Reduce strategy with token-based chunking to handle transcripts longer than the model's context window.
4.  Evaluate the generated summary using a multi-step pipeline implemented within this demo:
    *   **JSON Structure Validation:** Checks if the LLM output adheres to the requested JSON format.
    *   **BERT Score:** Calculates semantic similarity (F1 score) between the summary text and the original transcript.
    *   **AI Judge:** Conditionally uses a separate LLM call (also configured for `gpt-4-turbo` in this demo) to assess the summary's Overall Quality, Relevance, and Fluency, providing scores and a critique.
5.  Display the transcript, generated summary, and detailed evaluation metrics in a user-friendly interface.

## Features

*   Fetches transcripts from YouTube videos (where available).
*   Generates summaries using OpenAI's `gpt-4-turbo` (configurable in `constants.py`).
*   Handles long transcripts via token-based Map-Reduce chunking (`tiktoken`).
*   Implements an automated evaluation pipeline:
    *   JSON validation.
    *   BERT Score (F1) calculation.
    *   AI-as-Judge evaluation (Overall, Relevance, Fluency scores + critique).
*   Displays results clearly using Streamlit components (metrics, expanders, JSON/text areas).
*   Provides informative status updates during processing.
*   Includes basic logging to `logs/app.log`.

## Technology Stack

*   **Language:** Python 3.8+
*   **Framework:** Streamlit (for the web UI)
*   **AI/LLM:** OpenAI Python SDK (`openai`)
*   **Transcript Fetching:** `youtube-transcript-api`
*   **Tokenization:** `tiktoken` (for OpenAI models)
*   **Evaluation Metrics:** `bert-score` (requires `torch` and `transformers`)
*   **Environment Management:** `python-dotenv`

## Setup

1.  **Prerequisites:**
    *   Python 3.8 or higher installed.
    *   `git` installed (for cloning).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

3.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the root directory of the project.
    *   Add your OpenAI API key to this file:
        ```dotenv
        # .env
        OPENAI_API_KEY="sk-YOUR_API_KEY_HERE"
        ```
    *   Ensure this file is listed in your `.gitignore` if you plan to commit the code.

4.  **Install Dependencies:**
    *   It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install the required packages:
        ```bash
        pip install streamlit openai python-dotenv youtube-transcript-api tiktoken bert-score torch torchvision torchaudio transformers
        ```
        *(Note: Explicitly installing `torch`, `torchvision`, `torchaudio`, and `transformers` helps ensure `bert-score` dependencies are met correctly.)*

## Running the Application

Once the setup is complete, run the Streamlit application from your terminal:

```bash
streamlit run app.py