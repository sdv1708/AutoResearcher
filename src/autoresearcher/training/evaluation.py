"""
Evaluation Metrics for AutoResearcher Training Pipeline

This module provides comprehensive evaluation metrics for medical AI models
including QA accuracy, summarization quality, and clinical relevance scoring.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""

    metric_name: str
    score: float
    details: Dict[str, Any]
    sample_count: int
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = None


class BaseMetric:
    """Abstract base class for evaluation metrics"""

    def __init__(self, name: str):
        self.name = name

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """
        Compute metric score

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            **kwargs: Additional metric-specific arguments

        Returns:
            EvaluationResult with score and details
        """
        raise NotImplementedError

    def _bootstrap_confidence_interval(
        self, scores: List[float], confidence: float = 0.95, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval

        Args:
            scores: List of individual scores
            confidence: Confidence level (0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # TODO: Implement bootstrap confidence interval calculation
        # Example: Use numpy.random.choice for resampling, compute percentiles
        # Use: np.random.choice, np.percentile
        # Input: scores (List[float]), confidence (float), n_bootstrap (int)
        # Output: Tuple[float, float] with confidence interval bounds
        # Note: Handle edge cases with small sample sizes

        if len(scores) < 2:
            return (0.0, 0.0)

        np.random.seed(42)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_scores.append(np.mean(resampled))

        # Calculate percentiles for confidence interval
        alpha = (1 - confidence) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100

        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)

        return (float(lower_bound), float(upper_bound))


class ExactMatchMetric(BaseMetric):
    """Exact match accuracy for QA tasks"""

    def __init__(self):
        super().__init__("exact_match")

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """Compute exact match accuracy"""
        # TODO: Implement exact match calculation with normalization
        # Example: Normalize text (lowercase, strip), compare strings
        # Use: String normalization, case-insensitive comparison
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with exact match score
        # Note: Handle multiple reference answers and partial matches

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        normalize = kwargs.get("normalize", True)

        exact_matches = []
        correct_count = 0

        for pred, ref in zip(predictions, references):
            if normalize:
                pred_norm = self._normalize_text(pred)
                ref_norm = self._normalize_text(ref)
                match = pred_norm == ref_norm
            else:
                match = pred == ref

            exact_matches.append(1.0 if match else 0.0)
            if match:
                correct_count += 1

        accuracy = correct_count / len(predictions) if predictions else 0.0
        ci = self._bootstrap_confidence_interval(exact_matches)

        return EvaluationResult(
            metric_name=self.name,
            score=accuracy,
            details={
                "correct": correct_count,
                "total": len(predictions),
                "accuracy": accuracy,
                "individual_scores": exact_matches,
            },
            sample_count=len(predictions),
            confidence_interval=ci,
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # TODO: Implement comprehensive text normalization
        # Example: Remove punctuation, lowercase, strip whitespace
        # Use: Regular expressions, string methods
        # Input: text (str)
        # Output: normalized text (str)
        # Note: Handle medical abbreviations and special cases

        # Basic normalization
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common punctuation
        text = re.sub(r"[^\w\s]", "", text)

        return text


class F1ScoreMetric(BaseMetric):
    """F1 score for QA tasks (token-level)"""

    def __init__(self):
        super().__init__("f1_score")

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """Compute F1 score"""
        # TODO: Implement token-level F1 score calculation
        # Example: Tokenize, compute precision/recall, calculate F1
        # Use: Token overlap, set operations
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with F1 score
        # Note: Handle multiple reference answers and partial matches

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        f1_scores = []
        total_precision = 0.0
        total_recall = 0.0

        for pred, ref in zip(predictions, references):
            f1, precision, recall = self._compute_f1_single(pred, ref)
            f1_scores.append(f1)
            total_precision += precision
            total_recall += recall

        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_precision = total_precision / len(predictions) if predictions else 0.0
        avg_recall = total_recall / len(predictions) if predictions else 0.0

        ci = self._bootstrap_confidence_interval(f1_scores)

        return EvaluationResult(
            metric_name=self.name,
            score=avg_f1,
            details={
                "f1": avg_f1,
                "precision": avg_precision,
                "recall": avg_recall,
                "individual_scores": f1_scores,
            },
            sample_count=len(predictions),
            confidence_interval=ci,
        )

    def _compute_f1_single(
        self, prediction: str, reference: str
    ) -> Tuple[float, float, float]:
        """Compute F1 score for a single prediction-reference pair"""
        # TODO: Implement single-sample F1 calculation
        # Example: Tokenize both texts, compute token overlap
        # Use: Set intersection, precision/recall formulas
        # Input: prediction (str), reference (str)
        # Output: Tuple[float, float, float] (f1, precision, recall)
        # Note: Handle empty predictions and references

        # Tokenize
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        if not pred_tokens and not ref_tokens:
            return 1.0, 1.0, 1.0

        if not pred_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0

        # Compute overlap
        common_tokens = pred_tokens.intersection(ref_tokens)

        # Calculate precision and recall
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)

        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1, precision, recall


class ROUGEMetric(BaseMetric):
    """ROUGE metrics for summarization evaluation"""

    def __init__(self, rouge_types: List[str] = None):
        super().__init__("rouge")
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """Compute ROUGE scores"""
        # TODO: Implement ROUGE score calculation
        # Example: Use rouge-score library or implement n-gram overlap
        # Use: rouge_score.rouge_scorer, n-gram extraction
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with ROUGE scores
        # Note: Handle multiple references and different ROUGE variants

        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)

            all_scores = {rouge_type: [] for rouge_type in self.rouge_types}

            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                for rouge_type in self.rouge_types:
                    all_scores[rouge_type].append(scores[rouge_type].fmeasure)

            # Calculate average scores
            avg_scores = {}
            for rouge_type in self.rouge_types:
                avg_scores[rouge_type] = np.mean(all_scores[rouge_type])

            # Use ROUGE-L as the main score
            main_score = avg_scores.get("rougeL", avg_scores[self.rouge_types[0]])

            return EvaluationResult(
                metric_name=self.name,
                score=main_score,
                details=avg_scores,
                sample_count=len(predictions),
            )

        except ImportError:
            logger.warning(
                "rouge-score package not available, using basic implementation"
            )
            return self._compute_rouge_basic(predictions, references)

    def _compute_rouge_basic(
        self, predictions: List[str], references: List[str]
    ) -> EvaluationResult:
        """Basic ROUGE-like implementation"""
        # TODO: Implement basic ROUGE calculation without external library
        # Example: Compute n-gram overlap, LCS for ROUGE-L
        # Use: N-gram extraction, longest common subsequence
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with basic ROUGE scores
        # Note: Simplified version of ROUGE for fallback

        rouge1_scores = []

        for pred, ref in zip(predictions, references):
            # Simple ROUGE-1 (unigram overlap)
            pred_unigrams = set(pred.lower().split())
            ref_unigrams = set(ref.lower().split())

            if not ref_unigrams:
                rouge1_scores.append(0.0)
                continue

            overlap = len(pred_unigrams.intersection(ref_unigrams))
            rouge1 = overlap / len(ref_unigrams)
            rouge1_scores.append(rouge1)

        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0

        return EvaluationResult(
            metric_name=self.name,
            score=avg_rouge1,
            details={"rouge1": avg_rouge1},
            sample_count=len(predictions),
        )


class BERTScoreMetric(BaseMetric):
    """BERTScore for semantic similarity evaluation"""

    def __init__(self, model_type: str = "distilbert-base-uncased"):
        super().__init__("bertscore")
        self.model_type = model_type

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """Compute BERTScore"""
        # TODO: Implement BERTScore calculation
        # Example: Use bert-score library with pre-trained models
        # Use: bert_score.score, model loading
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with BERTScore
        # Note: Handle GPU/CPU execution and model caching

        try:
            from bert_score import score

            # Compute BERTScore
            P, R, F1 = score(
                predictions, references, model_type=self.model_type, verbose=False
            )

            # Convert to numpy arrays and compute means
            precision = P.mean().item()
            recall = R.mean().item()
            f1 = F1.mean().item()

            return EvaluationResult(
                metric_name=self.name,
                score=f1,
                details={
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "model_type": self.model_type,
                },
                sample_count=len(predictions),
            )

        except ImportError:
            logger.warning("bert-score package not available, using fallback")
            return self._compute_bertscore_fallback(predictions, references)

    def _compute_bertscore_fallback(
        self, predictions: List[str], references: List[str]
    ) -> EvaluationResult:
        """Fallback BERTScore implementation using simple similarity"""
        # TODO: Implement fallback similarity measure
        # Example: Use word overlap or simple embedding similarity
        # Use: Basic similarity measures, word vectors if available
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with similarity score
        # Note: Simplified alternative when BERTScore is unavailable

        similarities = []

        for pred, ref in zip(predictions, references):
            # Simple Jaccard similarity as fallback
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())

            if not pred_words and not ref_words:
                similarity = 1.0
            elif not pred_words or not ref_words:
                similarity = 0.0
            else:
                intersection = len(pred_words.intersection(ref_words))
                union = len(pred_words.union(ref_words))
                similarity = intersection / union

            similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 0.0

        return EvaluationResult(
            metric_name=self.name,
            score=avg_similarity,
            details={"jaccard_similarity": avg_similarity},
            sample_count=len(predictions),
        )


class MedicalRelevanceMetric(BaseMetric):
    """Medical domain-specific relevance scoring"""

    def __init__(self):
        super().__init__("medical_relevance")

        # Medical keywords and concepts
        self.medical_keywords = {
            "diagnoses": ["diagnosis", "disease", "condition", "syndrome", "disorder"],
            "treatments": [
                "treatment",
                "therapy",
                "medication",
                "drug",
                "intervention",
            ],
            "symptoms": ["symptom", "sign", "manifestation", "presentation"],
            "anatomy": ["heart", "lung", "brain", "liver", "kidney", "blood"],
            "procedures": ["surgery", "procedure", "operation", "test", "examination"],
        }

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """Compute medical relevance score"""
        # TODO: Implement medical relevance scoring
        # Example: Check for medical terminology, clinical accuracy
        # Use: Medical vocabularies, domain-specific keywords
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with relevance score
        # Note: Consider using medical NLP tools like scispaCy

        relevance_scores = []
        category_scores = {category: [] for category in self.medical_keywords}

        for pred, ref in zip(predictions, references):
            pred_score = self._score_medical_content(pred)
            ref_score = self._score_medical_content(ref)

            # Compare medical content coverage
            relevance = min(pred_score / max(ref_score, 0.1), 1.0)
            relevance_scores.append(relevance)

            # Category-specific scoring
            for category, keywords in self.medical_keywords.items():
                pred_cat_score = self._score_category(pred, keywords)
                category_scores[category].append(pred_cat_score)

        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

        # Calculate average category scores
        avg_category_scores = {}
        for category, scores in category_scores.items():
            avg_category_scores[category] = np.mean(scores) if scores else 0.0

        return EvaluationResult(
            metric_name=self.name,
            score=avg_relevance,
            details={
                "overall_relevance": avg_relevance,
                "category_scores": avg_category_scores,
                "individual_scores": relevance_scores,
            },
            sample_count=len(predictions),
        )

    def _score_medical_content(self, text: str) -> float:
        """Score text for medical content"""
        # TODO: Implement comprehensive medical content scoring
        # Example: Use medical vocabularies, named entity recognition
        # Use: Medical word lists, NLP libraries
        # Input: text (str)
        # Output: float score (0-1)
        # Note: Consider clinical accuracy and appropriateness

        text_lower = text.lower()
        total_keywords = sum(
            len(keywords) for keywords in self.medical_keywords.values()
        )
        found_keywords = 0

        for keywords in self.medical_keywords.values():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords += 1

        return found_keywords / total_keywords if total_keywords > 0 else 0.0

    def _score_category(self, text: str, keywords: List[str]) -> float:
        """Score text for specific medical category"""
        text_lower = text.lower()
        found = sum(1 for keyword in keywords if keyword in text_lower)
        return found / len(keywords) if keywords else 0.0


class PerplexityMetric(BaseMetric):
    """Perplexity metric for language modeling evaluation"""

    def __init__(self, model_name: str = None):
        super().__init__("perplexity")
        self.model_name = model_name

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> EvaluationResult:
        """Compute perplexity score"""
        # TODO: Implement perplexity calculation
        # Example: Use transformers models to compute log-likelihood
        # Use: transformers.AutoModelForCausalLM, tokenizer
        # Input: predictions (List[str]), references (List[str])
        # Output: EvaluationResult with perplexity score
        # Note: Handle model loading and GPU memory management

        try:
            # Placeholder implementation - would use actual model
            perplexities = []

            for text in predictions:
                # Simplified perplexity estimation based on text statistics
                words = text.split()
                if len(words) == 0:
                    perplexity = float("inf")
                else:
                    # Simple heuristic: longer sentences tend to have higher perplexity
                    avg_word_length = np.mean([len(word) for word in words])
                    perplexity = np.exp(avg_word_length / 10)  # Arbitrary formula

                perplexities.append(perplexity)

            avg_perplexity = np.mean(perplexities) if perplexities else float("inf")

            return EvaluationResult(
                metric_name=self.name,
                score=avg_perplexity,
                details={
                    "perplexity": avg_perplexity,
                    "individual_scores": perplexities,
                    "note": "Simplified implementation",
                },
                sample_count=len(predictions),
            )

        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=float("inf"),
                details={"error": str(e)},
                sample_count=len(predictions),
            )


class EvaluationSuite:
    """Complete evaluation suite for medical AI models"""

    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or [
            "exact_match",
            "f1_score",
            "rouge",
            "bertscore",
            "medical_relevance",
        ]
        self.metric_instances = self._initialize_metrics()

    def _initialize_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize metric instances"""
        # TODO: Implement metric initialization with configuration
        # Example: Create metric instances based on requested metrics
        # Use: Factory pattern, metric registry
        # Input: self.metrics (List[str])
        # Output: Dict[str, BaseMetric] with initialized metrics
        # Note: Handle metric-specific configuration and dependencies

        metric_classes = {
            "exact_match": ExactMatchMetric,
            "f1_score": F1ScoreMetric,
            "rouge": ROUGEMetric,
            "bertscore": BERTScoreMetric,
            "medical_relevance": MedicalRelevanceMetric,
            "perplexity": PerplexityMetric,
        }

        instances = {}
        for metric_name in self.metrics:
            if metric_name in metric_classes:
                instances[metric_name] = metric_classes[metric_name]()
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return instances

    def evaluate(
        self, predictions: List[str], references: List[str], task_type: str = "qa"
    ) -> Dict[str, EvaluationResult]:
        """
        Run complete evaluation suite

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            task_type: Type of task ("qa", "summarization", "synthesis")

        Returns:
            Dictionary of evaluation results
        """
        # TODO: Implement complete evaluation with task-specific metrics
        # Example: Select appropriate metrics based on task type
        # Use: Task-specific metric filtering, parallel evaluation
        # Input: predictions (List[str]), references (List[str]), task_type (str)
        # Output: Dict[str, EvaluationResult] with all metric results
        # Note: Handle metric failures gracefully and provide summary statistics

        logger.info(f"Running evaluation suite for {task_type} task")

        results = {}

        # Filter metrics based on task type
        task_metrics = self._get_task_metrics(task_type)

        for metric_name in task_metrics:
            if metric_name in self.metric_instances:
                try:
                    logger.info(f"Computing {metric_name}")
                    result = self.metric_instances[metric_name].compute(
                        predictions, references
                    )
                    results[metric_name] = result
                    logger.info(f"{metric_name}: {result.score:.3f}")
                except Exception as e:
                    logger.error(f"Failed to compute {metric_name}: {e}")
                    results[metric_name] = EvaluationResult(
                        metric_name=metric_name,
                        score=0.0,
                        details={"error": str(e)},
                        sample_count=len(predictions),
                    )

        return results

    def _get_task_metrics(self, task_type: str) -> List[str]:
        """Get appropriate metrics for task type"""
        # TODO: Implement task-specific metric selection
        # Example: Return relevant metrics based on task characteristics
        # Use: Task-metric mapping, metric compatibility
        # Input: task_type (str)
        # Output: List[str] with appropriate metrics
        # Note: Consider both primary and secondary metrics for each task

        task_metric_map = {
            "qa": ["exact_match", "f1_score", "medical_relevance"],
            "summarization": ["rouge", "bertscore", "medical_relevance"],
            "synthesis": ["rouge", "bertscore", "medical_relevance"],
            "general": ["f1_score", "bertscore", "perplexity"],
        }

        return task_metric_map.get(task_type, self.metrics)

    def generate_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Generate human-readable evaluation report"""
        # TODO: Implement comprehensive evaluation report generation
        # Example: Create formatted report with scores, confidence intervals
        # Use: String formatting, statistical summaries
        # Input: results (Dict[str, EvaluationResult])
        # Output: str with formatted evaluation report
        # Note: Include recommendations and comparisons to baselines

        report_lines = []
        report_lines.append("# Evaluation Report")
        report_lines.append("=" * 50)

        for metric_name, result in results.items():
            report_lines.append(f"\n## {metric_name.upper()}")
            report_lines.append(f"Score: {result.score:.3f}")
            report_lines.append(f"Samples: {result.sample_count}")

            if result.confidence_interval:
                ci_lower, ci_upper = result.confidence_interval
                report_lines.append(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

            # Add metric-specific details
            if "error" in result.details:
                report_lines.append(f"Error: {result.details['error']}")
            elif metric_name == "rouge" and "rouge1" in result.details:
                report_lines.append(f"ROUGE-1: {result.details['rouge1']:.3f}")
                if "rouge2" in result.details:
                    report_lines.append(f"ROUGE-2: {result.details['rouge2']:.3f}")
                if "rougeL" in result.details:
                    report_lines.append(f"ROUGE-L: {result.details['rougeL']:.3f}")
            elif metric_name == "bertscore":
                if "precision" in result.details:
                    report_lines.append(f"Precision: {result.details['precision']:.3f}")
                    report_lines.append(f"Recall: {result.details['recall']:.3f}")

        # Overall summary
        report_lines.append("\n## Summary")
        valid_scores = [
            result.score
            for result in results.values()
            if not ("error" in result.details)
        ]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            report_lines.append(f"Average Score: {avg_score:.3f}")
            report_lines.append(f"Metrics Computed: {len(valid_scores)}/{len(results)}")

        return "\n".join(report_lines)


def run_comprehensive_evaluation(
    predictions: List[str],
    references: List[str],
    task_type: str = "qa",
    metrics: List[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation and return detailed results

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        task_type: Type of task being evaluated
        metrics: List of metrics to compute (None for defaults)

    Returns:
        Dictionary with evaluation results and report
    """
    # TODO: Implement comprehensive evaluation function
    # Example: Run full evaluation suite, generate report, save results
    # Use: EvaluationSuite, result aggregation
    # Input: predictions, references, task_type, metrics
    # Output: Dict with results, report, and metadata
    # Note: Include timing information and system details

    suite = EvaluationSuite(metrics=metrics)

    start_time = pd.Timestamp.now()
    results = suite.evaluate(predictions, references, task_type)
    end_time = pd.Timestamp.now()

    report = suite.generate_report(results)

    return {
        "results": results,
        "report": report,
        "task_type": task_type,
        "sample_count": len(predictions),
        "evaluation_time_seconds": (end_time - start_time).total_seconds(),
        "timestamp": end_time.isoformat(),
    }


# Test function
def test_evaluation_metrics():
    """Test evaluation metrics with sample data"""
    print("Testing evaluation metrics...")

    # Sample data
    predictions = [
        "Metformin is effective for treating type 2 diabetes",
        "The patient shows symptoms of hypertension",
        "Surgery is recommended for this cardiac condition",
    ]

    references = [
        "Metformin shows good efficacy in type 2 diabetes treatment",
        "Hypertension symptoms are present in the patient",
        "Cardiac surgery is the recommended treatment approach",
    ]

    # Test individual metrics
    metrics = [
        ExactMatchMetric(),
        F1ScoreMetric(),
        ROUGEMetric(),
        MedicalRelevanceMetric(),
    ]

    for metric in metrics:
        try:
            result = metric.compute(predictions, references)
            print(f"{metric.name}: {result.score:.3f}")
        except Exception as e:
            print(f"{metric.name}: Error - {e}")

    # Test evaluation suite
    try:
        evaluation_results = run_comprehensive_evaluation(
            predictions=predictions, references=references, task_type="qa"
        )
        print("\nEvaluation Suite Results:")
        print(f"Sample count: {evaluation_results['sample_count']}")
        print(f"Evaluation time: {evaluation_results['evaluation_time_seconds']:.2f}s")

    except Exception as e:
        print(f"Evaluation suite error: {e}")

    print("Evaluation testing completed!")


if __name__ == "__main__":
    test_evaluation_metrics()
