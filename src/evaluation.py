import re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import pandas as pd
from datetime import datetime

class ModelEvaluator:
    """Evaluates model performance and response quality"""
    
    def __init__(self):
        self.medical_keywords = [
            "symptom", "symptoms", "pain", "medication", "treatment", "diagnosis",
            "doctor", "hospital", "health", "medical", "condition", "therapy",
            "prescription", "medicine", "dose", "side effects", "recovery",
            "examination", "test", "results", "appointment", "follow-up"
        ]
        
        self.inappropriate_content = [
            "lawsuit", "sue", "malpractice", "lawyer", "attorney",
            "kill", "die", "death", "suicide", "harm",
            "stupid", "idiot", "moron", "incompetent"
        ]
        
        self.persona_indicators = {
            "calm": ["thank", "understand", "okay", "appreciate", "helpful"],
            "anxious": ["worried", "scared", "nervous", "concerned", "afraid"],
            "rude": ["waste", "time", "ridiculous", "whatever", "hurry"],
            "patient": ["wait", "time", "grateful", "kind", "thorough"],
            "confused": ["understand", "confused", "explain", "unclear", "lost"],
            "aggressive": ["angry", "unacceptable", "demand", "furious", "outrageous"]
        }
    
    def evaluate_model(
        self, 
        model_name: str, 
        test_data: List[Dict[str, Any]], 
        model_client: Any
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        results = {
            "model_name": model_name,
            "evaluation_date": datetime.now().isoformat(),
            "test_data_size": len(test_data),
            "detailed_results": [],
            "overall_metrics": {}
        }
        
        quality_scores = []
        persona_consistency_scores = []
        medical_appropriateness_scores = []
        response_times = []
        
        for i, test_case in enumerate(test_data):
            try:
                # Generate response
                start_time = datetime.now()
                
                if hasattr(model_client, 'generate_response'):
                    # Fine-tuned model
                    response = model_client.generate_response(
                        test_case['doctor'],
                        test_case['persona'],
                        model_name
                    )
                else:
                    # Ollama model
                    response = model_client.generate_response(
                        test_case['doctor'],
                        test_case['persona'],
                        model_name
                    )
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Evaluate response
                evaluation = self._evaluate_single_response(
                    test_case['doctor'],
                    response,
                    test_case['patient'],
                    test_case['persona']
                )
                
                # Store detailed results
                detailed_result = {
                    "test_case_id": i,
                    "doctor_input": test_case['doctor'],
                    "expected_persona": test_case['persona'],
                    "generated_response": response,
                    "expected_response": test_case['patient'],
                    "response_time": response_time,
                    **evaluation
                }
                
                results["detailed_results"].append(detailed_result)
                
                # Collect metrics
                quality_scores.append(evaluation['quality_score'])
                persona_consistency_scores.append(evaluation['persona_consistency'])
                medical_appropriateness_scores.append(evaluation['medical_appropriateness'])
                response_times.append(response_time)
                
            except Exception as e:
                print(f"Error evaluating test case {i}: {e}")
                continue
        
        # Calculate overall metrics
        if quality_scores:
            results["overall_metrics"] = {
                "avg_quality": np.mean(quality_scores),
                "avg_persona_consistency": np.mean(persona_consistency_scores),
                "avg_medical_appropriateness": np.mean(medical_appropriateness_scores),
                "avg_response_time": np.mean(response_times),
                "quality_std": np.std(quality_scores),
                "persona_consistency_std": np.std(persona_consistency_scores),
                "successful_evaluations": len(quality_scores),
                "failed_evaluations": len(test_data) - len(quality_scores)
            }
        
        return results
    
    def _evaluate_single_response(
        self,
        doctor_input: str,
        generated_response: str,
        expected_response: str,
        expected_persona: str
    ) -> Dict[str, Any]:
        """Evaluate a single model response"""
        
        evaluation = {}
        
        # 1. Quality Score (0-1)
        evaluation['quality_score'] = self._calculate_quality_score(
            generated_response, expected_response
        )
        
        # 2. Persona Consistency (0-1)
        evaluation['persona_consistency'] = self._calculate_persona_consistency(
            generated_response, expected_persona
        )
        
        # 3. Medical Appropriateness (0-1)
        evaluation['medical_appropriateness'] = self._calculate_medical_appropriateness(
            generated_response, doctor_input
        )
        
        # 4. Response Length Check
        evaluation['length_appropriate'] = self._check_response_length(generated_response)
        
        # 5. Coherence Check
        evaluation['coherence_score'] = self._calculate_coherence(generated_response)
        
        # 6. Safety Check
        evaluation['safety_score'] = self._calculate_safety_score(generated_response)
        
        return evaluation
    
    def _calculate_quality_score(self, generated: str, expected: str) -> float:
        """Calculate response quality compared to expected response"""
        
        # Simple similarity based on word overlap
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.5  # Neutral score if no expected response
        
        intersection = generated_words.intersection(expected_words)
        union = generated_words.union(expected_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # Adjust score based on length appropriateness
        length_ratio = len(generated.split()) / max(len(expected.split()), 1)
        length_penalty = 1.0 if 0.5 <= length_ratio <= 2.0 else 0.8
        
        return min(jaccard_similarity * length_penalty, 1.0)
    
    def _calculate_persona_consistency(self, response: str, expected_persona: str) -> float:
        """Calculate how well the response matches the expected persona"""
        
        response_lower = response.lower()
        persona_keywords = self.persona_indicators.get(expected_persona, [])
        
        if not persona_keywords:
            return 0.5  # Neutral score for unknown personas
        
        # Count persona indicators
        indicator_count = sum(1 for keyword in persona_keywords if keyword in response_lower)
        
        # Check for conflicting persona indicators
        other_personas = [p for p in self.persona_indicators.keys() if p != expected_persona]
        conflicting_count = 0
        
        for other_persona in other_personas:
            other_keywords = self.persona_indicators[other_persona]
            conflicting_count += sum(1 for keyword in other_keywords if keyword in response_lower)
        
        # Calculate score
        positive_score = min(indicator_count / len(persona_keywords), 1.0)
        negative_penalty = min(conflicting_count * 0.2, 0.5)
        
        return max(positive_score - negative_penalty, 0.0)
    
    def _calculate_medical_appropriateness(self, response: str, doctor_input: str) -> float:
        """Calculate medical appropriateness of the response"""
        
        response_lower = response.lower()
        
        # Check for medical context relevance
        medical_relevance = sum(1 for keyword in self.medical_keywords if keyword in response_lower)
        medical_relevance_score = min(medical_relevance / 3, 1.0)  # Normalize to max 3 keywords
        
        # Check for inappropriate content
        inappropriate_count = sum(1 for word in self.inappropriate_content if word in response_lower)
        safety_penalty = min(inappropriate_count * 0.3, 0.9)
        
        # Check if response is contextually appropriate
        context_score = 0.5  # Base score
        
        # Simple context check - if doctor asks about symptoms, patient should respond about symptoms
        if any(word in doctor_input.lower() for word in ["feel", "symptom", "pain", "hurt"]):
            if any(word in response_lower for word in ["feel", "pain", "hurt", "better", "worse", "fine"]):
                context_score = 1.0
        
        return max((medical_relevance_score + context_score) / 2 - safety_penalty, 0.0)
    
    def _check_response_length(self, response: str) -> bool:
        """Check if response length is appropriate"""
        
        word_count = len(response.split())
        return 3 <= word_count <= 50  # Reasonable range for patient responses
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence"""
        
        # Basic coherence checks
        score = 1.0
        
        # Check for complete sentences
        if not response.strip().endswith(('.', '!', '?')):
            score -= 0.2
        
        # Check for repeated words (may indicate generation issues)
        words = response.lower().split()
        if len(words) != len(set(words)) and len(words) > 5:
            repetition_ratio = (len(words) - len(set(words))) / len(words)
            score -= repetition_ratio * 0.5
        
        # Check for very short or very long responses
        word_count = len(words)
        if word_count < 3:
            score -= 0.3
        elif word_count > 100:
            score -= 0.2
        
        return max(score, 0.0)
    
    def _calculate_safety_score(self, response: str) -> float:
        """Calculate safety score (absence of harmful content)"""
        
        response_lower = response.lower()
        
        # Check for inappropriate medical advice
        harmful_medical = ["don't take medication", "ignore doctor", "medical advice is wrong"]
        medical_violations = sum(1 for phrase in harmful_medical if phrase in response_lower)
        
        # Check for inappropriate content
        inappropriate_violations = sum(1 for word in self.inappropriate_content if word in response_lower)
        
        # Calculate penalty
        total_violations = medical_violations + inappropriate_violations
        penalty = min(total_violations * 0.3, 1.0)
        
        return max(1.0 - penalty, 0.0)
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        
        report = f"""
# Model Evaluation Report

**Model:** {evaluation_results['model_name']}
**Evaluation Date:** {evaluation_results['evaluation_date']}
**Test Cases:** {evaluation_results['test_data_size']}

## Overall Performance

"""
        
        metrics = evaluation_results.get('overall_metrics', {})
        
        if metrics:
            report += f"""
- **Average Quality Score:** {metrics['avg_quality']:.3f} ± {metrics['quality_std']:.3f}
- **Persona Consistency:** {metrics['avg_persona_consistency']:.3f} ± {metrics['persona_consistency_std']:.3f}
- **Medical Appropriateness:** {metrics['avg_medical_appropriateness']:.3f} ± {metrics.get('medical_appropriateness_std', 0):.3f}
- **Average Response Time:** {metrics['avg_response_time']:.3f} seconds
- **Success Rate:** {metrics['successful_evaluations']}/{metrics['successful_evaluations'] + metrics['failed_evaluations']} ({metrics['successful_evaluations']/(metrics['successful_evaluations'] + metrics['failed_evaluations'])*100:.1f}%)

## Performance Analysis

"""
            
            # Performance categories
            quality_rating = "Excellent" if metrics['avg_quality'] > 0.8 else "Good" if metrics['avg_quality'] > 0.6 else "Needs Improvement"
            persona_rating = "Excellent" if metrics['avg_persona_consistency'] > 0.8 else "Good" if metrics['avg_persona_consistency'] > 0.6 else "Needs Improvement"
            medical_rating = "Excellent" if metrics['avg_medical_appropriateness'] > 0.8 else "Good" if metrics['avg_medical_appropriateness'] > 0.6 else "Needs Improvement"
            
            report += f"""
- **Quality Rating:** {quality_rating}
- **Persona Consistency Rating:** {persona_rating}
- **Medical Appropriateness Rating:** {medical_rating}

## Recommendations

"""
            
            # Generate recommendations
            recommendations = []
            
            if metrics['avg_quality'] < 0.7:
                recommendations.append("- Consider additional training data or fine-tuning iterations to improve response quality")
            
            if metrics['avg_persona_consistency'] < 0.7:
                recommendations.append("- Focus on persona-specific training examples to improve character consistency")
            
            if metrics['avg_medical_appropriateness'] < 0.7:
                recommendations.append("- Include more medical context in training data to improve domain relevance")
            
            if metrics['avg_response_time'] > 2.0:
                recommendations.append("- Consider model optimization or quantization to improve response speed")
            
            if not recommendations:
                recommendations.append("- Model performance is satisfactory across all metrics")
            
            report += "\n".join(recommendations)
        
        else:
            report += "No metrics available - evaluation may have failed."
        
        return report
    
    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model evaluation results"""
        
        if len(evaluation_results) < 2:
            return {"error": "Need at least 2 models to compare"}
        
        comparison = {
            "models": [result['model_name'] for result in evaluation_results],
            "metrics_comparison": {},
            "best_model": {},
            "detailed_comparison": []
        }
        
        metrics_to_compare = ['avg_quality', 'avg_persona_consistency', 'avg_medical_appropriateness', 'avg_response_time']
        
        for metric in metrics_to_compare:
            values = []
            for result in evaluation_results:
                if 'overall_metrics' in result and metric in result['overall_metrics']:
                    values.append(result['overall_metrics'][metric])
                else:
                    values.append(0.0)
            
            comparison['metrics_comparison'][metric] = values
            
            # Find best model for this metric
            if metric != 'avg_response_time':  # Higher is better
                best_idx = np.argmax(values)
            else:  # Lower is better for response time
                best_idx = np.argmin(values)
            
            comparison['best_model'][metric] = evaluation_results[best_idx]['model_name']
        
        return comparison
