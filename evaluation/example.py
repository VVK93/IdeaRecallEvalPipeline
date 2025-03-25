import asyncio
from evaluator import Evaluator, SummaryOutput

async def main():
    # Initialize evaluator with custom thresholds
    evaluator = Evaluator(
        bert_threshold=0.8,
        rouge_threshold=0.6,
        ai_judge_threshold=75.0,
        human_rating_threshold=3.5
    )
    
    # Example transcript and generated output
    transcript = """
    In this video, we'll explore the fundamentals of machine learning.
    Machine learning is a subset of artificial intelligence that focuses on building
    systems that can learn from and make decisions based on data. There are three
    main types of machine learning: supervised learning, unsupervised learning,
    and reinforcement learning.
    """
    
    output = SummaryOutput(
        summary="This video covers machine learning basics, explaining it as a subset of AI that enables systems to learn from data. It discusses three main types: supervised, unsupervised, and reinforcement learning.",
        flashcards=[
            {
                "question": "What is machine learning?",
                "answer": "A subset of artificial intelligence that focuses on building systems that can learn from and make decisions based on data."
            },
            {
                "question": "What are the three main types of machine learning?",
                "answer": "Supervised learning, unsupervised learning, and reinforcement learning."
            }
        ],
        metadata={
            "generation_id": "example_001",
            "model": "gpt-4",
            "timestamp": "2024-01-01T00:00:00"
        }
    )
    
    # Evaluate the output
    passed, metrics = await evaluator.evaluate_output(
        transcript=transcript,
        output=output,
        human_rating=4.5  # Optional human rating
    )
    
    # Print results
    print(f"Evaluation Results:")
    print(f"Passed: {passed}")
    print(f"BERT Score: {metrics.bert_score:.3f}")
    print(f"ROUGE-L Score: {metrics.rouge_l:.3f}")
    print(f"AI Judge Score: {metrics.ai_judge_score:.3f}")
    print(f"Human Rating: {metrics.human_rating}")
    
    # Analyze performance trends
    trends = evaluator.analyze_performance_trends()
    print("\nPerformance Trends:")
    print(f"BERT Score Trend: {trends['bert_score_trend']:.3f}")
    print(f"ROUGE-L Score Trend: {trends['rouge_l_trend']:.3f}")
    print(f"AI Judge Score Trend: {trends['ai_judge_score_trend']:.3f}")
    print(f"Pass Rate: {trends['pass_rate']:.1f}%")

if __name__ == "__main__":
    asyncio.run(main()) 