from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import json
from bert_score import score
from rouge_score import rouge_scorer
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class SummaryOutput(BaseModel):
    summary: str
    flashcards: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

class EvaluationMetrics(BaseModel):
    generation_id: str
    timestamp: datetime
    bert_score: float
    rouge_l: float
    passed: bool

class Evaluator:
    def __init__(self, 
                 bert_threshold: float = 0.8,
                 rouge_threshold: float = 0.6,
                 ai_judge_threshold: float = 75.0,
                 human_rating_threshold: float = 3.5):
        self.bert_threshold = bert_threshold
        self.rouge_threshold = rouge_threshold
        self.ai_judge_threshold = ai_judge_threshold
        self.human_rating_threshold = human_rating_threshold
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_dir / "evaluation.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def validate_json(self, output: str) -> bool:
        """Validate if the output is valid JSON"""
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            logging.error("Invalid JSON output")
            return False
            
    def compute_bert_score(self, reference: str, candidate: str) -> float:
        """Compute BERTScore between reference and candidate"""
        try:
            P, R, F1 = score([candidate], [reference], lang='en', verbose=False)
            return float(F1.mean())
        except Exception as e:
            logging.error(f"Error computing BERTScore: {str(e)}")
            return 0.0
            
    def compute_rouge_score(self, reference: str, candidate: str) -> float:
        """Compute ROUGE-L score between reference and candidate"""
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logging.error(f"Error computing ROUGE score: {str(e)}")
            return 0.0
            
    async def get_ai_judge_score(self, 
                               transcript: str, 
                               summary: str, 
                               flashcards: List[Dict[str, str]]) -> Tuple[float, str]:
        """Get AI judge score and reasoning"""
        prompt = f"""
        Evaluate the following YouTube video summary and flashcards:
        
        Transcript excerpt:
        {transcript}
        
        Generated Summary:
        {summary}
        
        Generated Flashcards:
        {json.dumps(flashcards, indent=2)}
        
        Please evaluate on the following criteria:
        1. Correctness (0-100): How accurate is the information?
        2. Coverage (0-100): How well does it cover the key points?
        3. Relevance (0-100): How relevant are the flashcards to the content?
        
        Provide a JSON response with:
        {{
            "overall_score": <average of the three scores>,
            "correctness": <score>,
            "coverage": <score>,
            "relevance": <score>,
            "reasoning": "<detailed explanation>"
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["overall_score"], result["reasoning"]
            
        except Exception as e:
            logging.error(f"Error getting AI judge score: {str(e)}")
            return 0.0, "Error in AI evaluation"
            
    async def evaluate_output(self, 
                       transcript: str, 
                       output: SummaryOutput,
                       human_rating: Optional[float] = None) -> Tuple[bool, EvaluationMetrics]:
        """
        Evaluate the generated output using all metrics
        Returns (passed_evaluation, metrics)
        """
        # Validate JSON structure
        if not self.validate_json(json.dumps(output.dict())):
            return False, None
            
        # Compute automatic metrics
        bert_score = self.compute_bert_score(transcript, output.summary)
        rouge_score = self.compute_rouge_score(transcript, output.summary)
        
        # Get AI judge score
        ai_score, reasoning = await self.get_ai_judge_score(
            transcript, output.summary, output.flashcards
        )
        
        # Create metrics object
        metrics = EvaluationMetrics(
            bert_score=bert_score,
            rouge_l=rouge_score,
            passed=bert_score >= self.bert_threshold and
                  rouge_score >= self.rouge_threshold and
                  ai_score >= self.ai_judge_threshold and
                  (human_rating is None or human_rating >= self.human_rating_threshold),
            generation_id=output.metadata.get("generation_id", "unknown"),
            timestamp=datetime.now()
        )
        
        # Log evaluation results
        self.log_evaluation(metrics, metrics.passed)
        
        return metrics.passed, metrics
        
    def log_evaluation(self, metrics: EvaluationMetrics, passed: bool):
        """Log evaluation results to CSV and logging file"""
        # Log to CSV
        log_file = Path("logs/evaluation_results.csv")
        df = pd.DataFrame([metrics.dict()])
        df.to_csv(log_file, mode='a', header=not log_file.exists(), index=False)
        
        # Log to logging file
        logging.info(f"""
        Evaluation Results:
        Generation ID: {metrics.generation_id}
        BERT Score: {metrics.bert_score:.3f}
        ROUGE-L Score: {metrics.rouge_l:.3f}
        Passed Evaluation: {passed}
        """)
        
    def analyze_performance_trends(self) -> Dict[str, any]:
        """Analyze performance trends from logged data"""
        log_file = Path("logs/evaluation_results.csv")
        if not log_file.exists():
            return {}
            
        df = pd.read_csv(log_file)
        
        # Calculate trends over time
        trends = {
            "bert_score_trend": self._calculate_trend(df["bert_score"]),
            "rouge_l_trend": self._calculate_trend(df["rouge_l"]),
            "pass_rate": (df["passed"].mean() * 100),
            "average_scores": {
                "bert": df["bert_score"].mean(),
                "rouge_l": df["rouge_l"].mean()
            }
        }
        
        return trends
        
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate the trend (slope) of a metric over time"""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope 