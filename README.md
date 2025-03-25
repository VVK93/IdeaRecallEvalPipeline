# Idea Recall

A web application that generates summaries and flashcards from YouTube video transcripts using OpenAI's GPT-4 model.

## Features

- YouTube video transcript extraction
- AI-powered summary generation
- Automatic evaluation of summaries using:
  - BERT Score for semantic similarity
  - JSON structure validation
- Clean and intuitive user interface
- Real-time progress tracking

## Prerequisites

- Python 3.8+
- OpenAI API key
- YouTube video with available transcripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/idea-recall.git
cd idea-recall
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter a YouTube video URL and click "Generate Summary"

## Project Structure

```
idea-recall/
├── app.py              # Main Streamlit application
├── constants.py        # String constants and configuration
├── evaluation/         # Evaluation pipeline
│   └── evaluator.py    # Evaluation logic
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
└── README.md          # Project documentation
```

## Evaluation Metrics

The application evaluates summaries using:
- BERT Score: Measures semantic similarity between the summary and original transcript
- JSON Structure: Validates that the output is properly formatted JSON

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 