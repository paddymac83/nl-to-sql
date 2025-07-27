"""
Comprehensive Performance Evaluation Framework for Enhanced NL to SQL Agent
Evaluates query accuracy, result accuracy, table selection, success rates, and latency
"""

import json
import time
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib
import sqlparse
from sqlalchemy import create_engine, text
import statistics
import re
from pathlib import Path
import yaml
import logging
from multi_table_nlsql import NLToSQLAgent, InputPayload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GoldenTestCase:
    """Structure for golden test case data"""
    test_id: str
    natural_query: str
    user_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    
    # Golden standards
    expected_sql: str = ""
    expected_tables: List[str] = None
    expected_primary_table: str = ""
    expected_result: List[Dict[str, Any]] = None
    expected_row_count: int = 0
    expected_numerical_values: Dict[str, float] = None
    
    # Metadata
    difficulty: str = "medium"  # easy, medium, hard, expert
    category: str = "general"   # payments, accounts, categories, etc.
    description: str = ""
    tags: List[str] = None

@dataclass
class EvaluationResult:
    """Results from evaluating a single test case"""
    test_id: str
    success: bool
    
    # Query accuracy metrics
    sql_similarity_score: float = 0.0
    sql_edit_distance: int = 0
    sql_structure_match: bool = False
    
    # Table selection accuracy
    table_precision: float = 0.0
    table_recall: float = 0.0
    table_f1: float = 0.0
    primary_table_correct: bool = False
    
    # Result accuracy metrics
    result_exact_match: bool = False
    numerical_accuracy: Dict[str, float] = None
    row_count_match: bool = False
    result_similarity_score: float = 0.0
    
    # Execution metrics
    execution_successful: bool = False
    error_message: str = ""
    error_category: str = ""
    
    # Latency metrics (in milliseconds)
    total_latency: float = 0.0
    stage_latencies: Dict[str, float] = None
    
    # Generated outputs
    generated_sql: str = ""
    generated_tables: List[str] = None
    generated_primary_table: str = ""
    actual_result: List[Dict[str, Any]] = None

class SQLSimilarityAnalyzer:
    """Analyzes similarity between generated and golden SQL queries"""
    
    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison"""
        # Parse and format SQL
        try:
            parsed = sqlparse.parse(sql)[0]
            formatted = sqlparse.format(
                str(parsed), 
                reindent=True, 
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=True
            )
            
            # Remove extra whitespace
            formatted = re.sub(r'\s+', ' ', formatted).strip()
            return formatted
        except:
            # Fallback to basic normalization
            normalized = re.sub(r'\s+', ' ', sql.strip().upper())
            return normalized
    
    @staticmethod
    def calculate_similarity(sql1: str, sql2: str) -> Tuple[float, int]:
        """Calculate similarity score and edit distance"""
        norm1 = SQLSimilarityAnalyzer.normalize_sql(sql1)
        norm2 = SQLSimilarityAnalyzer.normalize_sql(sql2)
        
        # Calculate sequence similarity
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # Calculate edit distance
        edit_distance = len(list(difflib.unified_diff(norm1.split(), norm2.split())))
        
        return similarity, edit_distance
    
    @staticmethod
    def check_structure_match(sql1: str, sql2: str) -> bool:
        """Check if SQL queries have similar structure"""
        try:
            parsed1 = sqlparse.parse(sql1)[0]
            parsed2 = sqlparse.parse(sql2)[0]
            
            # Extract key components
            def extract_components(parsed):
                components = {
                    'select': [],
                    'from': [],
                    'where': [],
                    'join': [],
                    'group_by': [],
                    'order_by': []
                }
                
                for token in parsed.flatten():
                    if token.ttype is sqlparse.tokens.Keyword:
                        key = token.value.lower()
                        if key in components:
                            components[key].append(key)
                
                return components
            
            comp1 = extract_components(parsed1)
            comp2 = extract_components(parsed2)
            
            # Check if major components match
            major_components = ['select', 'from', 'where', 'join']
            matches = sum(1 for comp in major_components 
                         if bool(comp1[comp]) == bool(comp2[comp]))
            
            return matches >= 3  # At least 3 major components match
            
        except:
            return False

class TableSelectionEvaluator:
    """Evaluates table selection accuracy"""
    
    @staticmethod
    def calculate_metrics(predicted: List[str], actual: List[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for table selection"""
        if not actual:
            return 1.0 if not predicted else 0.0, 1.0, 1.0 if not predicted else 0.0
        
        predicted_set = set(predicted)
        actual_set = set(actual)
        
        if not predicted_set:
            return 0.0, 0.0, 0.0
        
        intersection = predicted_set.intersection(actual_set)
        
        precision = len(intersection) / len(predicted_set)
        recall = len(intersection) / len(actual_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

class ResultAccuracyEvaluator:
    """Evaluates accuracy of query results"""
    
    @staticmethod
    def compare_results(actual: List[Dict], expected: List[Dict]) -> Tuple[bool, float]:
        """Compare actual vs expected results"""
        if not expected:
            return True, 1.0
        
        if len(actual) != len(expected):
            return False, 0.0
        
        # Sort both for comparison
        actual_sorted = sorted(actual, key=lambda x: str(sorted(x.items())))
        expected_sorted = sorted(expected, key=lambda x: str(sorted(x.items())))
        
        exact_match = actual_sorted == expected_sorted
        
        # Calculate similarity score
        total_fields = 0
        matching_fields = 0
        
        for act_row, exp_row in zip(actual_sorted, expected_sorted):
            for key in exp_row:
                total_fields += 1
                if key in act_row:
                    if ResultAccuracyEvaluator._values_close(act_row[key], exp_row[key]):
                        matching_fields += 1
        
        similarity = matching_fields / total_fields if total_fields > 0 else 0.0
        
        return exact_match, similarity
    
    @staticmethod
    def _values_close(val1, val2, tolerance=0.01) -> bool:
        """Check if two values are close within tolerance"""
        if type(val1) != type(val2):
            return str(val1) == str(val2)
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) <= tolerance
        
        return val1 == val2
    
    @staticmethod
    def compare_numerical_values(actual: List[Dict], expected_values: Dict[str, float]) -> Dict[str, float]:
        """Compare specific numerical values"""
        accuracy_scores = {}
        
        for field, expected_val in expected_values.items():
            # Extract values for this field from actual results
            actual_vals = []
            for row in actual:
                if field in row and isinstance(row[field], (int, float)):
                    actual_vals.append(row[field])
            
            if actual_vals:
                # For aggregated values, compare the sum/avg/etc.
                if len(actual_vals) == 1:
                    actual_val = actual_vals[0]
                else:
                    actual_val = sum(actual_vals)  # Assume sum for now
                
                # Calculate accuracy
                if expected_val != 0:
                    accuracy = 1 - abs(actual_val - expected_val) / abs(expected_val)
                    accuracy = max(0, accuracy)  # Ensure non-negative
                else:
                    accuracy = 1.0 if actual_val == 0 else 0.0
                
                accuracy_scores[field] = accuracy
            else:
                accuracy_scores[field] = 0.0
        
        return accuracy_scores

class LatencyProfiler:
    """Profiles latency across different stages of the pipeline"""
    
    def __init__(self):
        self.stage_start_times = {}
        self.stage_durations = {}
    
    def start_stage(self, stage_name: str):
        """Mark the start of a stage"""
        self.stage_start_times[stage_name] = time.time()
    
    def end_stage(self, stage_name: str):
        """Mark the end of a stage and record duration"""
        if stage_name in self.stage_start_times:
            duration = (time.time() - self.stage_start_times[stage_name]) * 1000  # Convert to ms
            self.stage_durations[stage_name] = duration
            return duration
        return 0.0
    
    def get_total_latency(self) -> float:
        """Get total latency across all stages"""
        return sum(self.stage_durations.values())
    
    def get_stage_latencies(self) -> Dict[str, float]:
        """Get latencies for all stages"""
        return self.stage_durations.copy()

class ErrorAnalyzer:
    """Analyzes and categorizes errors"""
    
    ERROR_CATEGORIES = {
        'sql_generation': ['generation error', 'llm error', 'openai error'],
        'sql_validation': ['validation error', 'dangerous keyword', 'invalid query'],
        'sql_execution': ['execution error', 'database error', 'connection error'],
        'table_detection': ['table detection', 'schema error'],
        'payload_parsing': ['payload error', 'validation error'],
        'timeout': ['timeout', 'time limit'],
        'unknown': []
    }
    
    @staticmethod
    def categorize_error(error_message: str) -> str:
        """Categorize error based on error message"""
        error_lower = error_message.lower()
        
        for category, keywords in ErrorAnalyzer.ERROR_CATEGORIES.items():
            if any(keyword in error_lower for keyword in keywords):
                return category
        
        return 'unknown'

class PerformanceEvaluator:
    """Main performance evaluation engine"""
    
    def __init__(self, agent: NLToSQLAgent, golden_dataset_path: str):
        self.agent = agent
        self.golden_dataset_path = golden_dataset_path
        self.golden_cases = self._load_golden_dataset()
        
        # Initialize analyzers
        self.sql_analyzer = SQLSimilarityAnalyzer()
        self.table_evaluator = TableSelectionEvaluator()
        self.result_evaluator = ResultAccuracyEvaluator()
        self.error_analyzer = ErrorAnalyzer()
    
    def _load_golden_dataset(self) -> List[GoldenTestCase]:
        """Load golden dataset from file"""
        golden_cases = []
        
        try:
            with open(self.golden_dataset_path, 'r') as f:
                if self.golden_dataset_path.endswith('.json'):
                    data = json.load(f)
                elif self.golden_dataset_path.endswith(('.yml', '.yaml')):
                    data = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported file format. Use JSON or YAML.")
            
            for case_data in data.get('test_cases', []):
                case = GoldenTestCase(**case_data)
                golden_cases.append(case)
            
            logger.info(f"Loaded {len(golden_cases)} golden test cases")
            
        except Exception as e:
            logger.error(f"Failed to load golden dataset: {e}")
            
        return golden_cases
    
    async def evaluate_single_case(self, test_case: GoldenTestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        logger.info(f"Evaluating test case: {test_case.test_id}")
        
        profiler = LatencyProfiler()
        result = EvaluationResult(test_id=test_case.test_id, success=False)
        
        try:
            # Prepare input payload
            payload = None
            if test_case.payload:
                payload = InputPayload(**test_case.payload)
            
            # Start total timing
            profiler.start_stage('total')
            
            # Process query through agent
            agent_result = await self.agent.process_query(
                test_case.natural_query,
                test_case.user_id,
                payload,
                store_result=False  # Don't store evaluation queries
            )
            
            # End total timing
            profiler.end_stage('total')
            
            # Extract results
            result.execution_successful = agent_result['success']
            result.generated_sql = agent_result.get('sql_query', '')
            result.generated_tables = agent_result.get('relevant_tables', [])
            result.generated_primary_table = agent_result.get('primary_table', '')
            
            if not agent_result['success']:
                result.error_message = agent_result.get('error_message', '')
                result.error_category = self.error_analyzer.categorize_error(result.error_message)
            else:
                # Get actual results if execution was successful
                if agent_result.get('formatted_data', {}).get('data'):
                    result.actual_result = agent_result['formatted_data']['data']
            
            # Evaluate SQL similarity
            if test_case.expected_sql and result.generated_sql:
                similarity, edit_dist = self.sql_analyzer.calculate_similarity(
                    result.generated_sql, test_case.expected_sql
                )
                result.sql_similarity_score = similarity
                result.sql_edit_distance = edit_dist
                result.sql_structure_match = self.sql_analyzer.check_structure_match(
                    result.generated_sql, test_case.expected_sql
                )
            
            # Evaluate table selection
            if test_case.expected_tables:
                precision, recall, f1 = self.table_evaluator.calculate_metrics(
                    result.generated_tables, test_case.expected_tables
                )
                result.table_precision = precision
                result.table_recall = recall
                result.table_f1 = f1
            
            if test_case.expected_primary_table:
                result.primary_table_correct = (
                    result.generated_primary_table == test_case.expected_primary_table
                )
            
            # Evaluate result accuracy
            if test_case.expected_result and result.actual_result:
                exact_match, similarity = self.result_evaluator.compare_results(
                    result.actual_result, test_case.expected_result
                )
                result.result_exact_match = exact_match
                result.result_similarity_score = similarity
            
            if test_case.expected_row_count > 0:
                actual_count = len(result.actual_result) if result.actual_result else 0
                result.row_count_match = (actual_count == test_case.expected_row_count)
            
            # Evaluate numerical accuracy
            if test_case.expected_numerical_values and result.actual_result:
                result.numerical_accuracy = self.result_evaluator.compare_numerical_values(
                    result.actual_result, test_case.expected_numerical_values
                )
            
            # Record latencies
            result.total_latency = profiler.get_total_latency()
            result.stage_latencies = profiler.get_stage_latencies()
            
            # Determine overall success
            result.success = (
                result.execution_successful and
                result.sql_similarity_score > 0.7 and  # Configurable threshold
                result.table_f1 > 0.8 and  # Configurable threshold
                (not test_case.expected_result or result.result_similarity_score > 0.9)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating test case {test_case.test_id}: {e}")
            result.error_message = str(e)
            result.error_category = 'evaluation_error'
        
        return result
    
    async def evaluate_all(self, test_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate all test cases with optional filtering"""
        logger.info("Starting comprehensive evaluation")
        
        # Filter test cases if needed
        test_cases = self.golden_cases
        if test_filter:
            test_cases = self._filter_test_cases(test_cases, test_filter)
        
        logger.info(f"Evaluating {len(test_cases)} test cases")
        
        # Run evaluations
        results = []
        for test_case in test_cases:
            result = await self.evaluate_single_case(test_case)
            results.append(result)
        
        # Generate comprehensive report
        report = self._generate_report(results)
        
        return report
    
    def _filter_test_cases(self, test_cases: List[GoldenTestCase], 
                          filter_criteria: Dict[str, Any]) -> List[GoldenTestCase]:
        """Filter test cases based on criteria"""
        filtered = []
        
        for case in test_cases:
            include = True
            
            if 'difficulty' in filter_criteria:
                if case.difficulty not in filter_criteria['difficulty']:
                    include = False
            
            if 'category' in filter_criteria:
                if case.category not in filter_criteria['category']:
                    include = False
            
            if 'tags' in filter_criteria:
                if not any(tag in (case.tags or []) for tag in filter_criteria['tags']):
                    include = False
            
            if include:
                filtered.append(case)
        
        return filtered
    
    def _generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Overall metrics
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r.success)
        executable_cases = sum(1 for r in results if r.execution_successful)
        
        # SQL quality metrics
        sql_similarities = [r.sql_similarity_score for r in results if r.sql_similarity_score > 0]
        sql_structure_matches = sum(1 for r in results if r.sql_structure_match)
        
        # Table selection metrics
        table_precisions = [r.table_precision for r in results if r.table_precision > 0]
        table_recalls = [r.table_recall for r in results if r.table_recall > 0]
        table_f1s = [r.table_f1 for r in results if r.table_f1 > 0]
        primary_table_correct = sum(1 for r in results if r.primary_table_correct)
        
        # Result accuracy metrics
        exact_matches = sum(1 for r in results if r.result_exact_match)
        result_similarities = [r.result_similarity_score for r in results 
                              if r.result_similarity_score > 0]
        
        # Latency metrics
        total_latencies = [r.total_latency for r in results if r.total_latency > 0]
        
        # Error analysis
        error_categories = {}
        for result in results:
            if result.error_category:
                error_categories[result.error_category] = error_categories.get(result.error_category, 0) + 1
        
        # Compile report
        report = {
            "evaluation_summary": {
                "total_test_cases": total_cases,
                "successful_cases": successful_cases,
                "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
                "executable_cases": executable_cases,
                "execution_rate": executable_cases / total_cases if total_cases > 0 else 0,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            
            "sql_quality_metrics": {
                "average_similarity": statistics.mean(sql_similarities) if sql_similarities else 0,
                "median_similarity": statistics.median(sql_similarities) if sql_similarities else 0,
                "structure_match_rate": sql_structure_matches / total_cases if total_cases > 0 else 0,
                "similarity_distribution": {
                    "excellent (>0.9)": sum(1 for s in sql_similarities if s > 0.9),
                    "good (0.7-0.9)": sum(1 for s in sql_similarities if 0.7 <= s <= 0.9),
                    "poor (<0.7)": sum(1 for s in sql_similarities if s < 0.7)
                }
            },
            
            "table_selection_metrics": {
                "average_precision": statistics.mean(table_precisions) if table_precisions else 0,
                "average_recall": statistics.mean(table_recalls) if table_recalls else 0,
                "average_f1": statistics.mean(table_f1s) if table_f1s else 0,
                "primary_table_accuracy": primary_table_correct / total_cases if total_cases > 0 else 0
            },
            
            "result_accuracy_metrics": {
                "exact_match_rate": exact_matches / total_cases if total_cases > 0 else 0,
                "average_result_similarity": statistics.mean(result_similarities) if result_similarities else 0,
                "median_result_similarity": statistics.median(result_similarities) if result_similarities else 0
            },
            
            "latency_metrics": {
                "average_latency_ms": statistics.mean(total_latencies) if total_latencies else 0,
                "median_latency_ms": statistics.median(total_latencies) if total_latencies else 0,
                "p95_latency_ms": statistics.quantiles(total_latencies, n=20)[18] if len(total_latencies) > 10 else 0,
                "p99_latency_ms": statistics.quantiles(total_latencies, n=100)[98] if len(total_latencies) > 50 else 0
            },
            
            "error_analysis": {
                "error_categories": error_categories,
                "most_common_errors": sorted(error_categories.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            
            "detailed_results": [asdict(result) for result in results]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save evaluation report to file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")

class GoldenDatasetBuilder:
    """Helper class to build golden datasets"""
    
    @staticmethod
    def create_test_case(test_id: str, natural_query: str, **kwargs) -> GoldenTestCase:
        """Create a test case with validation"""
        return GoldenTestCase(
            test_id=test_id,
            natural_query=natural_query,
            **kwargs
        )
    
    @staticmethod
    def save_dataset(test_cases: List[GoldenTestCase], output_path: str):
        """Save test cases to golden dataset file"""
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_cases": len(test_cases),
                "version": "1.0"
            },
            "test_cases": [asdict(case) for case in test_cases]
        }
        
        with open(output_path, 'w') as f:
            if output_path.endswith('.json'):
                json.dump(dataset, f, indent=2, default=str)
            elif output_path.endswith(('.yml', '.yaml')):
                yaml.dump(dataset, f, default_flow_style=False)
        
        logger.info(f"Golden dataset saved to {output_path}")

# Example usage and testing
async def main():
    """Example usage of the evaluation framework"""
    
    # Initialize agent
    agent = NLToSQLAgent("postgresql://user:pass@localhost/db", "api-key")
    
    # Create sample golden dataset
    sample_cases = [
        GoldenDatasetBuilder.create_test_case(
            test_id="basic_payment_query",
            natural_query="Show me all payments over $100",
            expected_sql="SELECT * FROM payments WHERE amount > 100",
            expected_tables=["payments"],
            expected_primary_table="payments",
            expected_row_count=5,
            expected_numerical_values={"amount": 650.00},
            difficulty="easy",
            category="payments",
            description="Basic payment filtering query"
        ),
        
        GoldenDatasetBuilder.create_test_case(
            test_id="multi_table_category_query",
            natural_query="Show spending by category",
            payload={"CIN": 22},
            expected_sql="SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id WHERE p.customer_id = 22 GROUP BY c.category_name",
            expected_tables=["payments", "categories"],
            expected_primary_table="payments",
            expected_result=[
                {"category_name": "Food", "sum": 150.50},
                {"category_name": "Transport", "sum": 89.25}
            ],
            expected_numerical_values={"sum": 239.75},
            difficulty="medium",
            category="categories",
            description="Multi-table query with payload filtering"
        )
    ]
    
    # Save sample dataset
    GoldenDatasetBuilder.save_dataset(sample_cases, "golden_dataset.json")
    
    # Run evaluation
    evaluator = PerformanceEvaluator(agent, "golden_dataset.json")
    
    # Evaluate specific categories
    report = await evaluator.evaluate_all(test_filter={"category": ["payments", "categories"]})
    
    # Save report
    evaluator.save_report(report, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Print summary
    print("Evaluation Summary:")
    print(f"Success Rate: {report['evaluation_summary']['success_rate']:.2%}")
    print(f"Execution Rate: {report['evaluation_summary']['execution_rate']:.2%}")
    print(f"Average SQL Similarity: {report['sql_quality_metrics']['average_similarity']:.3f}")
    print(f"Average Latency: {report['latency_metrics']['average_latency_ms']:.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())