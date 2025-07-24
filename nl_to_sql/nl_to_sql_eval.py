"""
Comprehensive Evaluation Framework for NL-to-SQL Systems
Includes automated testing, benchmarking, and continuous monitoring
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sqlite3
from abc import ABC, abstractmethod
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class TestCategory(Enum):
    BASIC = "basic"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    JOINS = "joins"
    DATE_TIME = "date_time"
    COMPLEX = "complex"
    EDGE_CASES = "edge_cases"

class MetricType(Enum):
    ACCURACY = "accuracy"
    EXECUTION_SUCCESS = "execution_success"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"

@dataclass
class TestCase:
    """Individual test case for evaluation"""
    id: str
    category: TestCategory
    natural_query: str
    expected_sql: str
    expected_result_structure: Dict[str, Any]
    user_id: Optional[str] = None
    difficulty: int = 1  # 1-5 scale
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    
class EvaluationResult(BaseModel):
    """Result of a single evaluation"""
    test_case_id: str
    success: bool
    generated_sql: Optional[str] = None
    execution_time_ms: float = 0
    error_message: Optional[str] = None
    accuracy_score: float = 0.0  # 0-1
    sql_similarity_score: float = 0.0  # 0-1
    result_correctness_score: float = 0.0  # 0-1
    security_score: float = 1.0  # 1 = secure, 0 = insecure
    performance_score: float = 1.0  # 1 = excellent, 0 = poor
    
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            'accuracy': 0.3,
            'sql_similarity': 0.2,
            'result_correctness': 0.3,
            'security': 0.1,
            'performance': 0.1
        }
        
        return (
            self.accuracy_score * weights['accuracy'] +
            self.sql_similarity_score * weights['sql_similarity'] +
            self.result_correctness_score * weights['result_correctness'] +
            self.security_score * weights['security'] +
            self.performance_score * weights['performance']
        )

class TestSuite:
    """Collection of test cases with management utilities"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.test_cases: Dict[str, TestCase] = {}
        self.load_default_tests()
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case to the suite"""
        self.test_cases[test_case.id] = test_case
    
    def get_tests_by_category(self, category: TestCategory) -> List[TestCase]:
        """Get all tests in a category"""
        return [tc for tc in self.test_cases.values() if tc.category == category]
    
    def get_tests_by_difficulty(self, min_difficulty: int, max_difficulty: int) -> List[TestCase]:
        """Get tests within difficulty range"""
        return [tc for tc in self.test_cases.values() 
                if min_difficulty <= tc.difficulty <= max_difficulty]
    
    def load_default_tests(self):
        """Load default test cases"""
        default_tests = [
            TestCase(
                id="basic_sum",
                category=TestCategory.AGGREGATION,
                natural_query="What is the total amount of all payments?",
                expected_sql="SELECT SUM(amount) as total_amount FROM payments",
                expected_result_structure={"columns": ["total_amount"], "row_count_min": 1, "row_count_max": 1},
                difficulty=1,
                tags={"sum", "aggregate", "basic"},
                description="Basic sum aggregation test"
            ),
            TestCase(
                id="user_filter",
                category=TestCategory.FILTERING,
                natural_query="How much did I spend last month?",
                expected_sql="SELECT SUM(amount) FROM payments WHERE user_id = 'user123' AND payment_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND payment_date < DATE_TRUNC('month', CURRENT_DATE)",
                expected_result_structure={"columns": ["sum"], "row_count_min": 1, "row_count_max": 1},
                user_id="user123",
                difficulty=2,
                tags={"filter", "date", "user_specific"},
                description="User-specific filtering with date range"
            ),
            TestCase(
                id="top_merchants",
                category=TestCategory.AGGREGATION,
                natural_query="Show me the top 5 merchants by total spending",
                expected_sql="SELECT merchant, SUM(amount) as total_spending FROM payments GROUP BY merchant ORDER BY total_spending DESC LIMIT 5",
                expected_result_structure={"columns": ["merchant", "total_spending"], "row_count_min": 1, "row_count_max": 5},
                difficulty=2,
                tags={"groupby", "orderby", "limit", "top_n"},
                description="Top N query with grouping and sorting"
            ),
            TestCase(
                id="date_range_complex",
                category=TestCategory.DATE_TIME,
                natural_query="Show me all payments between January 1st and February 28th of this year",
                expected_sql="SELECT * FROM payments WHERE payment_date >= '2024-01-01' AND payment_date <= '2024-02-28'",
                expected_result_structure={"columns": ["*"], "row_count_min": 0, "row_count_max": 10000},
                difficulty=3,
                tags={"date_range", "specific_dates"},
                description="Specific date range filtering"
            ),
            TestCase(
                id="security_injection",
                category=TestCategory.EDGE_CASES,
                natural_query="Show payments for user '; DROP TABLE payments; --",
                expected_sql="SELECT * FROM payments WHERE user_id = ''; DROP TABLE payments; --'",
                expected_result_structure={"should_fail": True},
                difficulty=4,
                tags={"security", "sql_injection", "malicious"},
                description="SQL injection attempt - should be blocked"
            )
        ]
        
        for test in default_tests:
            self.add_test_case(test)

class SQLSimilarityEvaluator:
    """Evaluate similarity between generated and expected SQL"""
    
    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison"""
        # Remove extra whitespace and convert to lowercase
        sql = ' '.join(sql.split()).lower()
        
        # Standardize common patterns
        replacements = {
            'select distinct': 'SELECT DISTINCT',
            'group by': 'GROUP BY',
            'order by': 'ORDER BY',
            'inner join': 'INNER JOIN',
            'left join': 'LEFT JOIN',
            'right join': 'RIGHT JOIN',
        }
        
        for old, new in replacements.items():
            sql = sql.replace(old, new)
        
        return sql
    
    @staticmethod
    def extract_sql_components(sql: str) -> Dict[str, Set[str]]:
        """Extract key components from SQL"""
        sql_upper = sql.upper()
        
        components = {
            'tables': set(),
            'columns': set(),
            'functions': set(),
            'keywords': set()
        }
        
        # Simple extraction (could be more sophisticated)
        if 'FROM' in sql_upper:
            from_part = sql_upper.split('FROM')[1].split('WHERE')[0].split('GROUP')[0].split('ORDER')[0]
            # Extract table names (simplified)
            tables = from_part.strip().split()
            components['tables'].update(t.strip(',') for t in tables if not t.upper() in ['JOIN', 'ON', 'INNER', 'LEFT', 'RIGHT'])
        
        # Extract common SQL functions
        functions = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN', 'DATE_TRUNC', 'EXTRACT']
        for func in functions:
            if func in sql_upper:
                components['functions'].add(func)
        
        # Extract keywords
        keywords = ['GROUP BY', 'ORDER BY', 'WHERE', 'HAVING', 'LIMIT', 'JOIN']
        for keyword in keywords:
            if keyword in sql_upper:
                components['keywords'].add(keyword)
        
        return components
    
    @classmethod
    def calculate_similarity(cls, generated_sql: str, expected_sql: str) -> float:
        """Calculate similarity score between two SQL queries"""
        if generated_sql.strip() == expected_sql.strip():
            return 1.0
        
        # Normalize both queries
        gen_norm = cls.normalize_sql(generated_sql)
        exp_norm = cls.normalize_sql(expected_sql)
        
        if gen_norm == exp_norm:
            return 0.95
        
        # Component-based similarity
        gen_components = cls.extract_sql_components(generated_sql)
        exp_components = cls.extract_sql_components(expected_sql)
        
        component_scores = []
        
        for component_type, gen_items in gen_components.items():
            exp_items = exp_components[component_type]
            
            if not exp_items and not gen_items:
                score = 1.0
            elif not exp_items or not gen_items:
                score = 0.0
            else:
                intersection = len(gen_items.intersection(exp_items))
                union = len(gen_items.union(exp_items))
                score = intersection / union if union > 0 else 0.0
            
            component_scores.append(score)
        
        return np.mean(component_scores)

class SecurityEvaluator:
    """Evaluate SQL security aspects"""
    
    DANGEROUS_PATTERNS = [
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM",
        r";\s*UPDATE\s+.*SET",
        r";\s*INSERT\s+INTO",
        r"UNION\s+SELECT.*FROM\s+information_schema",
        r"'.*OR.*'.*=.*'",
        r"'.*AND.*'.*=.*'",
        r"--.*$",
        r"/\*.*\*/"
    ]
    
    @classmethod
    def evaluate_security(cls, sql: str) -> Tuple[float, List[str]]:
        """Evaluate SQL security, return score and issues"""
        import re
        
        issues = []
        sql_upper = sql.upper()
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        # Check for non-SELECT statements
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        for stmt in statements:
            if not stmt.upper().startswith('SELECT'):
                issues.append(f"Non-SELECT statement detected: {stmt[:50]}...")
        
        # Calculate security score
        if not issues:
            return 1.0, []
        elif len(issues) <= 2:
            return 0.5, issues
        else:
            return 0.0, issues

class PerformanceEvaluator:
    """Evaluate query performance"""
    
    @staticmethod
    def evaluate_performance(execution_time_ms: float, result_count: int) -> float:
        """Evaluate performance based on execution time and result size"""
        
        # Performance thresholds
        if execution_time_ms < 100:  # < 100ms
            time_score = 1.0
        elif execution_time_ms < 500:  # 100-500ms
            time_score = 0.8
        elif execution_time_ms < 1000:  # 500ms-1s
            time_score = 0.6
        elif execution_time_ms < 5000:  # 1-5s
            time_score = 0.4
        else:  # > 5s
            time_score = 0.2
        
        # Result size considerations
        if result_count < 1000:
            size_score = 1.0
        elif result_count < 10000:
            size_score = 0.8
        else:
            size_score = 0.6
        
        return (time_score + size_score) / 2

class EvaluationEngine:
    """Main evaluation engine"""
    
    def __init__(self, sql_system, database_url: str):
        self.sql_system = sql_system
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.results_db = sqlite3.connect("evaluation_results.db", check_same_thread=False)
        self.setup_results_db()
    
    def setup_results_db(self):
        """Setup results database"""
        cursor = self.results_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                test_suite_name TEXT,
                system_version TEXT,
                total_tests INTEGER,
                passed_tests INTEGER,
                overall_score REAL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                run_id TEXT,
                test_case_id TEXT,
                success BOOLEAN,
                overall_score REAL,
                accuracy_score REAL,
                sql_similarity_score REAL,
                result_correctness_score REAL,
                security_score REAL,
                performance_score REAL,
                execution_time_ms REAL,
                error_message TEXT,
                generated_sql TEXT,
                FOREIGN KEY (run_id) REFERENCES evaluation_runs (id)
            )
        """)
        
        self.results_db.commit()
    
    def evaluate_single_test(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        start_time = time.time()
        
        try:
            # Generate SQL using the system
            if hasattr(self.sql_system, 'process_query'):
                result = self.sql_system.process_query(test_case.natural_query, test_case.user_id)
                generated_sql = result.get('sql_query', {}).get('query', '') if isinstance(result.get('sql_query'), dict) else result.get('sql_query', '')
                success = result.get('success', False)
                error_message = result.get('error_message') or result.get('error')
            else:
                # Fallback for different system interfaces
                generated_sql = self.sql_system.generate_sql(test_case.natural_query, test_case.user_id)
                success = True
                error_message = None
            
            execution_time = (time.time() - start_time) * 1000
            
            if not success:
                return EvaluationResult(
                    test_case_id=test_case.id,
                    success=False,
                    execution_time_ms=execution_time,
                    error_message=error_message
                )
            
            # Evaluate different aspects
            sql_similarity = SQLSimilarityEvaluator.calculate_similarity(generated_sql, test_case.expected_sql)
            security_score, security_issues = SecurityEvaluator.evaluate_security(generated_sql)
            
            # Try to execute the generated SQL for result correctness
            result_correctness = self.evaluate_result_correctness(generated_sql, test_case)
            
            # Performance evaluation
            performance_score = PerformanceEvaluator.evaluate_performance(execution_time, 0)  # Simplified
            
            # Calculate accuracy (weighted combination)
            accuracy = (sql_similarity * 0.6 + result_correctness * 0.4)
            
            return EvaluationResult(
                test_case_id=test_case.id,
                success=success,
                generated_sql=generated_sql,
                execution_time_ms=execution_time,
                accuracy_score=accuracy,
                sql_similarity_score=sql_similarity,
                result_correctness_score=result_correctness,
                security_score=security_score,
                performance_score=performance_score,
                error_message=error_message
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_case_id=test_case.id,
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def evaluate_result_correctness(self, generated_sql: str, test_case: TestCase) -> float:
        """Evaluate if generated SQL produces correct results"""
        try:
            with self.engine.connect() as connection:
                # Execute generated SQL
                result = pd.read_sql(text(generated_sql), connection)
                
                # Check against expected structure
                expected = test_case.expected_result_structure
                
                # Check if execution should fail (security tests)
                if expected.get('should_fail', False):
                    return 0.0  # Query should have been blocked
                
                # Check column count and names if specified
                if 'columns' in expected and expected['columns'] != ['*']:
                    expected_cols = set(expected['columns'])
                    actual_cols = set(result.columns)
                    if expected_cols != actual_cols:
                        return 0.5  # Partial credit for wrong columns
                
                # Check row count ranges
                row_count = len(result)
                min_rows = expected.get('row_count_min', 0)
                max_rows = expected.get('row_count_max', float('inf'))
                
                if not (min_rows <= row_count <= max_rows):
                    return 0.7  # Partial credit for wrong row count
                
                return 1.0  # All checks passed
                
        except Exception as e:
            # If query fails to execute, check if it should have failed
            if test_case.expected_result_structure.get('should_fail', False):
                return 1.0  # Correctly blocked malicious query
            return 0.0  # Query failed when it shouldn't have
    
    def run_evaluation(self, test_suite: TestSuite, parallel: bool = True) -> Dict[str, Any]:
        """Run full evaluation on test suite"""
        run_id = hashlib.md5(f"{test_suite.name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        start_time = datetime.now()
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_test = {
                    executor.submit(self.evaluate_single_test, test_case): test_case 
                    for test_case in test_suite.test_cases.values()
                }
                
                for future in as_completed(future_to_test):
                    result = future.result()
                    results.append(result)
        else:
            for test_case in test_suite.test_cases.values():
                result = self.evaluate_single_test(test_case)
                results.append(result)
        
        # Calculate overall metrics
        successful_tests = len([r for r in results if r.success])
        total_tests = len(results)
        overall_score = np.mean([r.overall_score() for r in results if r.success])
        
        # Store results
        self.store_evaluation_results(run_id, test_suite.name, results, overall_score)
        
        # Generate report
        evaluation_report = {
            "run_id": run_id,
            "timestamp": start_time.isoformat(),
            "test_suite": test_suite.name,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "overall_score": overall_score,
            "category_breakdown": self.analyze_by_category(results, test_suite),
            "difficulty_breakdown": self.analyze_by_difficulty(results, test_suite),
            "common_failures": self.analyze_failures(results),
            "performance_stats": self.analyze_performance(results),
            "security_issues": self.analyze_security_issues(results)
        }
        
        return evaluation_report
    
    def store_evaluation_results(self, run_id: str, suite_name: str, results: List[EvaluationResult], overall_score: float):
        """Store evaluation results in database"""
        cursor = self.results_db.cursor()
        
        # Store run metadata
        cursor.execute("""
            INSERT INTO evaluation_runs 
            (id, timestamp, test_suite_name, system_version, total_tests, passed_tests, overall_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now().isoformat(),
            suite_name,
            "1.0",  # System version
            len(results),
            len([r for r in results if r.success]),
            overall_score,
            "{}"  # Additional metadata
        ))
        
        # Store individual test results
        for result in results:
            cursor.execute("""
                INSERT INTO test_results 
                (run_id, test_case_id, success, overall_score, accuracy_score, 
                 sql_similarity_score, result_correctness_score, security_score, 
                 performance_score, execution_time_ms, error_message, generated_sql)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, result.test_case_id, result.success, result.overall_score(),
                result.accuracy_score, result.sql_similarity_score, 
                result.result_correctness_score, result.security_score,
                result.performance_score, result.execution_time_ms,
                result.error_message, result.generated_sql
            ))
        
        self.results_db.commit()
    
    def analyze_by_category(self, results: List[EvaluationResult], test_suite: TestSuite) -> Dict[str, Dict[str, float]]:
        """Analyze results by test category"""
        category_stats = {}
        
        for category in TestCategory:
            category_tests = test_suite.get_tests_by_category(category)
            category_results = [r for r in results if r.test_case_id in [tc.id for tc in category_tests]]