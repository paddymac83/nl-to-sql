#!/usr/bin/env python3
"""
Command Line Interface for Performance Evaluation Framework
Provides easy-to-use commands for evaluating NL-SQL performance
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional

# Import evaluation framework
from evaluate import PerformanceEvaluator, GoldenDatasetBuilder
from multi_table_nlsql import NLToSQLAgent, create_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationCLI:
    """Command line interface for evaluation framework"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Performance Evaluation Framework for Enhanced NL-SQL Agent",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run full evaluation
  python evaluate.py run --dataset golden_dataset.yaml --output report.json
  
  # Run specific categories
  python evaluate.py run --dataset golden_dataset.yaml --filter-category payments,accounts
  
  # Run with custom thresholds
  python evaluate.py run --dataset golden_dataset.yaml --sql-threshold 0.8 --table-threshold 0.9
  
  # Generate sample dataset
  python evaluate.py generate-sample --output sample_dataset.yaml --size 50
  
  # Create test case interactively
  python evaluate.py create-case --dataset my_dataset.yaml
  
  # Compare two reports
  python evaluate.py compare --baseline report1.json --experimental report2.json
  
  # Run continuous monitoring
  python evaluate.py monitor --dataset golden_dataset.yaml --interval 3600
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Run evaluation command
        run_parser = subparsers.add_parser('run', help='Run performance evaluation')
        run_parser.add_argument('--dataset', required=True, help='Path to golden dataset file')
        run_parser.add_argument('--output', help='Output path for evaluation report')
        run_parser.add_argument('--database-url', help='Database URL (overrides env var)')
        run_parser.add_argument('--openai-key', help='OpenAI API key (overrides env var)')
        
        # Filtering options
        run_parser.add_argument('--filter-category', help='Comma-separated categories to evaluate')
        run_parser.add_argument('--filter-difficulty', help='Comma-separated difficulties (easy,medium,hard,expert)')
        run_parser.add_argument('--filter-tags', help='Comma-separated tags to filter by')
        run_parser.add_argument('--filter-ids', help='Comma-separated test IDs to evaluate')
        
        # Threshold options
        run_parser.add_argument('--sql-threshold', type=float, default=0.7, help='SQL similarity threshold')
        run_parser.add_argument('--table-threshold', type=float, default=0.8, help='Table F1 threshold') 
        run_parser.add_argument('--result-threshold', type=float, default=0.9, help='Result similarity threshold')
        run_parser.add_argument('--latency-threshold', type=int, default=5000, help='Max latency in ms')
        
        # Execution options
        run_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel evaluations')
        run_parser.add_argument('--timeout', type=int, default=30, help='Timeout per test case in seconds')
        run_parser.add_argument('--store-results', action='store_true', help='Store evaluation queries in DB')
        run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # Report options
        run_parser.add_argument('--format', choices=['json', 'yaml', 'html'], default='json', help='Report format')
        run_parser.add_argument('--include-details', action='store_true', help='Include detailed results')
        
        # Generate sample dataset command
        generate_parser = subparsers.add_parser('generate-sample', help='Generate sample golden dataset')
        generate_parser.add_argument('--output', required=True, help='Output path for dataset file')
        generate_parser.add_argument('--size', type=int, default=20, help='Number of test cases to generate')
        generate_parser.add_argument('--categories', help='Comma-separated categories to include')
        generate_parser.add_argument('--difficulties', help='Comma-separated difficulties to include')
        
        # Create test case command
        create_parser = subparsers.add_parser('create-case', help='Create test case interactively')
        create_parser.add_argument('--dataset', required=True, help='Dataset file to add case to')
        create_parser.add_argument('--test-id', help='Test case ID')
        create_parser.add_argument('--query', help='Natural language query')
        create_parser.add_argument('--auto-generate', action='store_true', help='Auto-generate expected values')
        
        # Compare reports command
        compare_parser = subparsers.add_parser('compare', help='Compare two evaluation reports')
        compare_parser.add_argument('--baseline', required=True, help='Baseline report file')
        compare_parser.add_argument('--experimental', required=True, help='Experimental report file')
        compare_parser.add_argument('--output', help='Output path for comparison report')
        compare_parser.add_argument('--metrics', help='Comma-separated metrics to compare')
        
        # Monitor command
        monitor_parser = subparsers.add_parser('monitor', help='Continuous performance monitoring')
        monitor_parser.add_argument('--dataset', required=True, help='Path to golden dataset file')
        monitor_parser.add_argument('--interval', type=int, default=3600, help='Monitoring interval in seconds')
        monitor_parser.add_argument('--output-dir', default='monitoring_reports', help='Output directory for reports')
        monitor_parser.add_argument('--alert-threshold', type=float, default=0.8, help='Alert threshold for success rate')
        monitor_parser.add_argument('--max-runs', type=int, help='Maximum number of monitoring runs')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze evaluation reports')
        analyze_parser.add_argument('--report', required=True, help='Report file to analyze')
        analyze_parser.add_argument('--output', help='Output path for analysis')
        analyze_parser.add_argument('--focus', choices=['errors', 'latency', 'accuracy'], help='Analysis focus')
        
        # Validate dataset command
        validate_parser = subparsers.add_parser('validate', help='Validate golden dataset')
        validate_parser.add_argument('--dataset', required=True, help='Dataset file to validate')
        validate_parser.add_argument('--fix', action='store_true', help='Attempt to fix validation errors')
        
        return parser
    
    async def run_evaluation(self, args) -> Dict[str, Any]:
        """Run performance evaluation"""
        logger.info("Starting performance evaluation")
        
        # Initialize agent
        database_url = args.database_url or os.getenv('DATABASE_URL')
        openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
        
        if not database_url or not openai_key:
            logger.error("Database URL and OpenAI API key are required")
            sys.exit(1)
        
        try:
            agent = create_agent(database_url, openai_key)
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            sys.exit(1)
        
        # Initialize evaluator
        evaluator = PerformanceEvaluator(agent, args.dataset)
        
        # Set custom thresholds
        evaluator.sql_similarity_threshold = args.sql_threshold
        evaluator.table_f1_threshold = args.table_threshold
        evaluator.result_similarity_threshold = args.result_threshold
        evaluator.latency_threshold = args.latency_threshold
        
        # Build filter criteria
        test_filter = {}
        if args.filter_category:
            test_filter['category'] = args.filter_category.split(',')
        if args.filter_difficulty:
            test_filter['difficulty'] = args.filter_difficulty.split(',')
        if args.filter_tags:
            test_filter['tags'] = args.filter_tags.split(',')
        if args.filter_ids:
            test_filter['test_ids'] = args.filter_ids.split(',')
        
        # Run evaluation
        logger.info(f"Running evaluation with filter: {test_filter}")
        report = await evaluator.evaluate_all(test_filter if test_filter else None)
        
        # Save report
        output_path = args.output or f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if args.format == 'json':
            evaluator.save_report(report, output_path)
        elif args.format == 'yaml':
            import yaml
            output_path = output_path.replace('.json', '.yaml')
            with open(output_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
        elif args.format == 'html':
            output_path = output_path.replace('.json', '.html')
            self._generate_html_report(report, output_path)
        
        # Print summary
        self._print_evaluation_summary(report, args.verbose)
        
        logger.info(f"Evaluation complete. Report saved to: {output_path}")
        return report
    
    def generate_sample_dataset(self, args):
        """Generate sample golden dataset"""
        logger.info(f"Generating sample dataset with {args.size} test cases")
        
        # Define sample test cases
        sample_cases = []
        
        # Basic queries
        basic_queries = [
            ("Show me all payments", "SELECT * FROM payments WHERE user_id = 'user_001'"),
            ("What's my account balance?", "SELECT balance FROM accounts WHERE user_id = 'user_001'"),
            ("Show recent transactions", "SELECT * FROM payments WHERE user_id = 'user_001' ORDER BY transaction_date DESC LIMIT 10"),
            ("Show payments over $100", "SELECT * FROM payments WHERE amount > 100"),
            ("What's my total spending?", "SELECT SUM(amount) FROM payments WHERE user_id = 'user_001'"),
        ]
        
        # Multi-table queries
        multi_table_queries = [
            ("Show spending by category", "SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id WHERE p.user_id = 'user_001' GROUP BY c.category_name"),
            ("Which merchants do I use most?", "SELECT m.merchant_name, COUNT(*) FROM payments p JOIN merchants m ON p.merchant_id = m.merchant_id WHERE p.user_id = 'user_001' GROUP BY m.merchant_name ORDER BY COUNT(*) DESC"),
            ("Compare budget vs actual", "SELECT c.category_name, b.budget_amount, SUM(p.amount) as actual FROM budgets b JOIN categories c ON b.category_id = c.category_id LEFT JOIN payments p ON c.category_id = p.category_id AND p.user_id = b.user_id WHERE b.user_id = 'user_001' GROUP BY c.category_name, b.budget_amount"),
        ]
        
        # Payload queries
        payload_queries = [
            ("Show customer payments", "SELECT * FROM payments WHERE customer_id = 22", {"CIN": 22}),
            ("Account balance for account", "SELECT balance FROM accounts WHERE account_number = 900914", {"account_number": 900914}),
            ("Customer spending by category", "SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id WHERE p.customer_id = 22 GROUP BY c.category_name", {"CIN": 22}),
        ]
        
        # Generate test cases
        case_id = 1
        target_categories = (args.categories.split(',') if args.categories else 
                           ['basic_queries', 'multi_table_queries', 'payload_queries', 'aggregation_queries'])
        
        for i in range(args.size):
            category = target_categories[i % len(target_categories)]
            
            if category == 'basic_queries' and basic_queries:
                query, sql = basic_queries[i % len(basic_queries)]
                case = GoldenDatasetBuilder.create_test_case(
                    test_id=f"generated_{case_id:03d}",
                    natural_query=query,
                    user_id="user_001",
                    expected_sql=sql,
                    expected_tables=["payments"] if "payments" in sql else ["accounts"],
                    expected_primary_table="payments" if "payments" in sql else "accounts",
                    difficulty="easy",
                    category=category,
                    description=f"Generated {category} test case"
                )
                
            elif category == 'multi_table_queries' and multi_table_queries:
                query, sql = multi_table_queries[i % len(multi_table_queries)]
                tables = []
                if "payments" in sql: tables.append("payments")
                if "categories" in sql: tables.append("categories")
                if "merchants" in sql: tables.append("merchants")
                if "budgets" in sql: tables.append("budgets")
                
                case = GoldenDatasetBuilder.create_test_case(
                    test_id=f"generated_{case_id:03d}",
                    natural_query=query,
                    user_id="user_001",
                    expected_sql=sql,
                    expected_tables=tables,
                    expected_primary_table="payments",
                    difficulty="medium",
                    category=category,
                    description=f"Generated {category} test case"
                )
                
            elif category == 'payload_queries' and payload_queries:
                query, sql, payload = payload_queries[i % len(payload_queries)]
                case = GoldenDatasetBuilder.create_test_case(
                    test_id=f"generated_{case_id:03d}",
                    natural_query=query,
                    payload=payload,
                    expected_sql=sql,
                    expected_tables=["payments"] if "payments" in sql else ["accounts"],
                    expected_primary_table="payments" if "payments" in sql else "accounts",
                    difficulty="medium",
                    category=category,
                    description=f"Generated {category} test case"
                )
            else:
                # Default case
                case = GoldenDatasetBuilder.create_test_case(
                    test_id=f"generated_{case_id:03d}",
                    natural_query="Show me payments",
                    expected_sql="SELECT * FROM payments",
                    expected_tables=["payments"],
                    expected_primary_table="payments",
                    difficulty="easy",
                    category="basic_queries",
                    description="Default generated test case"
                )
            
            sample_cases.append(case)
            case_id += 1
        
        # Save dataset
        GoldenDatasetBuilder.save_dataset(sample_cases, args.output)
        logger.info(f"Sample dataset generated and saved to: {args.output}")
    
    async def create_test_case(self, args):
        """Create test case interactively"""
        logger.info("Creating test case interactively")
        
        # Load existing dataset
        try:
            with open(args.dataset, 'r') as f:
                if args.dataset.endswith('.json'):
                    existing_data = json.load(f)
                else:
                    import yaml
                    existing_data = yaml.safe_load(f)
            
            existing_cases = [case for case in existing_data.get('test_cases', [])]
        except FileNotFoundError:
            existing_cases = []
            existing_data = {"metadata": {"version": "1.0"}, "test_cases": []}
        
        # Interactive input
        test_id = args.test_id or input("Test ID: ")
        query = args.query or input("Natural language query: ")
        
        print("\nOptional fields (press Enter to skip):")
        user_id = input("User ID: ") or None
        
        # Payload input
        payload_input = input("Payload (JSON format): ")
        payload = None
        if payload_input:
            try:
                payload = json.loads(payload_input)
            except json.JSONDecodeError:
                print("Invalid JSON format for payload. Skipping.")
        
        expected_sql = input("Expected SQL: ")
        expected_tables = input("Expected tables (comma-separated): ").split(',') if input("Expected tables (comma-separated): ") else []
        expected_primary_table = input("Expected primary table: ") or ""
        
        difficulty = input("Difficulty (easy/medium/hard/expert): ") or "medium"
        category = input("Category: ") or "general"
        description = input("Description: ") or ""
        
        tags_input = input("Tags (comma-separated): ")
        tags = tags_input.split(',') if tags_input else []
        
        # Auto-generate expected values if requested
        if args.auto_generate and expected_sql:
            logger.info("Auto-generating expected values...")
            try:
                # Initialize agent and run query
                database_url = os.getenv('DATABASE_URL')
                openai_key = os.getenv('OPENAI_API_KEY')
                
                if database_url and openai_key:
                    agent = create_agent(database_url, openai_key)
                    result = await agent.process_query(query, user_id, payload, store_result=False)
                    
                    if result['success']:
                        expected_tables = result.get('relevant_tables', expected_tables)
                        expected_primary_table = result.get('primary_table', expected_primary_table)
                        expected_sql = result.get('sql_query', expected_sql)
                        
                        print(f"Auto-generated SQL: {expected_sql}")
                        print(f"Auto-generated tables: {expected_tables}")
                        print(f"Auto-generated primary table: {expected_primary_table}")
            except Exception as e:
                logger.warning(f"Auto-generation failed: {e}")
        
        # Create test case
        new_case = GoldenDatasetBuilder.create_test_case(
            test_id=test_id,
            natural_query=query,
            user_id=user_id,
            payload=payload,
            expected_sql=expected_sql,
            expected_tables=expected_tables,
            expected_primary_table=expected_primary_table,
            difficulty=difficulty,
            category=category,
            description=description,
            tags=tags
        )
        
        # Add to existing cases
        existing_cases.append(new_case)
        
        # Save updated dataset
        GoldenDatasetBuilder.save_dataset(existing_cases, args.dataset)
        logger.info(f"Test case '{test_id}' added to dataset: {args.dataset}")
    
    def compare_reports(self, args):
        """Compare two evaluation reports"""
        logger.info("Comparing evaluation reports")
        
        # Load reports
        with open(args.baseline, 'r') as f:
            baseline_report = json.load(f)
        
        with open(args.experimental, 'r') as f:
            experimental_report = json.load(f)
        
        # Compare metrics
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "baseline_file": args.baseline,
            "experimental_file": args.experimental,
            "metrics_comparison": {}
        }
        
        # Key metrics to compare
        key_metrics = [
            ("evaluation_summary", "success_rate"),
            ("evaluation_summary", "execution_rate"),
            ("sql_quality_metrics", "average_similarity"),
            ("table_selection_metrics", "average_f1"),
            ("result_accuracy_metrics", "exact_match_rate"),
            ("latency_metrics", "average_latency_ms"),
            ("latency_metrics", "p95_latency_ms")
        ]
        
        for section, metric in key_metrics:
            baseline_value = baseline_report.get(section, {}).get(metric, 0)
            experimental_value = experimental_report.get(section, {}).get(metric, 0)
            
            improvement = experimental_value - baseline_value
            improvement_pct = (improvement / baseline_value * 100) if baseline_value != 0 else 0
            
            comparison["metrics_comparison"][f"{section}.{metric}"] = {
                "baseline": baseline_value,
                "experimental": experimental_value,
                "improvement": improvement,
                "improvement_percentage": improvement_pct,
                "better": improvement > 0
            }
        
        # Statistical significance (basic)
        baseline_cases = len(baseline_report.get("detailed_results", []))
        experimental_cases = len(experimental_report.get("detailed_results", []))
        
        comparison["statistical_summary"] = {
            "baseline_test_cases": baseline_cases,
            "experimental_test_cases": experimental_cases,
            "same_test_set": baseline_cases == experimental_cases
        }
        
        # Overall assessment
        improvements = sum(1 for m in comparison["metrics_comparison"].values() if m["better"])
        total_metrics = len(comparison["metrics_comparison"])
        
        comparison["overall_assessment"] = {
            "improved_metrics": improvements,
            "total_metrics": total_metrics,
            "improvement_rate": improvements / total_metrics,
            "recommendation": "Deploy experimental version" if improvements > total_metrics * 0.6 else "Keep baseline version"
        }
        
        # Save comparison
        output_path = args.output or f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPARISON SUMMARY")
        print("="*60)
        
        for metric_name, metric_data in comparison["metrics_comparison"].items():
            status = "↗️" if metric_data["better"] else "↘️"
            print(f"{status} {metric_name}: {metric_data['baseline']:.4f} → {metric_data['experimental']:.4f} "
                  f"({metric_data['improvement_percentage']:+.2f}%)")
        
        print(f"\nOverall: {improvements}/{total_metrics} metrics improved")
        print(f"Recommendation: {comparison['overall_assessment']['recommendation']}")
        print(f"\nDetailed comparison saved to: {output_path}")
    
    async def monitor_performance(self, args):
        """Continuous performance monitoring"""
        logger.info(f"Starting performance monitoring (interval: {args.interval}s)")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize agent
        database_url = os.getenv('DATABASE_URL')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not database_url or not openai_key:
            logger.error("Database URL and OpenAI API key are required")
            sys.exit(1)
        
        agent = create_agent(database_url, openai_key)
        evaluator = PerformanceEvaluator(agent, args.dataset)
        
        run_count = 0
        try:
            while True:
                run_count += 1
                logger.info(f"Starting monitoring run #{run_count}")
                
                # Run evaluation
                report = await evaluator.evaluate_all()
                
                # Save report
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = output_dir / f"monitoring_report_{timestamp}.json"
                evaluator.save_report(report, str(report_path))
                
                # Check for alerts
                success_rate = report['evaluation_summary']['success_rate']
                if success_rate < args.alert_threshold:
                    alert_msg = f"ALERT: Performance degraded to {success_rate:.2%} (threshold: {args.alert_threshold:.2%})"
                    logger.warning(alert_msg)
                    self._send_alert(alert_msg, report)
                
                logger.info(f"Monitoring run #{run_count} complete. Success rate: {success_rate:.2%}")
                
                # Check max runs limit
                if args.max_runs and run_count >= args.max_runs:
                    logger.info(f"Reached maximum runs limit ({args.max_runs})")
                    break
                
                # Wait for next run
                await asyncio.sleep(args.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def analyze_report(self, args):
        """Analyze evaluation report"""
        logger.info(f"Analyzing report: {args.report}")
        
        with open(args.report, 'r') as f:
            report = json.load(f)
        
        analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "report_file": args.report,
            "focus": args.focus or "general"
        }
        
        if args.focus == 'errors' or not args.focus:
            # Error analysis
            detailed_results = report.get('detailed_results', [])
            failed_cases = [r for r in detailed_results if not r['success']]
            
            error_patterns = {}
            for case in failed_cases:
                error_msg = case.get('error_message', 'Unknown error')
                # Extract error pattern
                if 'generation error' in error_msg.lower():
                    pattern = 'SQL Generation'
                elif 'execution error' in error_msg.lower():
                    pattern = 'SQL Execution'
                elif 'validation error' in error_msg.lower():
                    pattern = 'SQL Validation'
                else:
                    pattern = 'Other'
                
                error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
            
            analysis['error_analysis'] = {
                "total_failures": len(failed_cases),
                "failure_rate": len(failed_cases) / len(detailed_results) if detailed_results else 0,
                "error_patterns": error_patterns,
                "top_errors": sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        if args.focus == 'latency' or not args.focus:
            # Latency analysis
            detailed_results = report.get('detailed_results', [])
            latencies = [r.get('total_latency', 0) for r in detailed_results if r.get('total_latency')]
            
            if latencies:
                import statistics
                analysis['latency_analysis'] = {
                    "average_latency": statistics.mean(latencies),
                    "median_latency": statistics.median(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    "slow_queries": [r for r in detailed_results if r.get('total_latency', 0) > 3000],
                    "fast_queries": [r for r in detailed_results if r.get('total_latency', 0) < 500]
                }
        
        if args.focus == 'accuracy' or not args.focus:
            # Accuracy analysis
            detailed_results = report.get('detailed_results', [])
            sql_similarities = [r.get('sql_similarity_score', 0) for r in detailed_results if r.get('sql_similarity_score')]
            table_f1s = [r.get('table_f1', 0) for r in detailed_results if r.get('table_f1')]
            
            analysis['accuracy_analysis'] = {
                "sql_quality": {
                    "high_similarity": len([s for s in sql_similarities if s > 0.9]),
                    "medium_similarity": len([s for s in sql_similarities if 0.7 <= s <= 0.9]),
                    "low_similarity": len([s for s in sql_similarities if s < 0.7])
                },
                "table_selection": {
                    "perfect_f1": len([f for f in table_f1s if f == 1.0]),
                    "good_f1": len([f for f in table_f1s if 0.8 <= f < 1.0]),
                    "poor_f1": len([f for f in table_f1s if f < 0.8])
                }
            }
        
        # Save analysis
        output_path = args.output or f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        self._print_analysis_summary(analysis)
        logger.info(f"Analysis saved to: {output_path}")
    
    def validate_dataset(self, args):
        """Validate golden dataset"""
        logger.info(f"Validating dataset: {args.dataset}")
        
        try:
            with open(args.dataset, 'r') as f:
                if args.dataset.endswith('.json'):
                    data = json.load(f)
                else:
                    import yaml
                    data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
        
        test_cases = data.get('test_cases', [])
        validation_errors = []
        warnings = []
        
        required_fields = ['test_id', 'natural_query']
        recommended_fields = ['expected_sql', 'expected_tables', 'difficulty', 'category']
        
        for i, case in enumerate(test_cases):
            case_errors = []
            case_warnings = []
            
            # Check required fields
            for field in required_fields:
                if field not in case or not case[field]:
                    case_errors.append(f"Missing required field: {field}")
            
            # Check recommended fields
            for field in recommended_fields:
                if field not in case or not case[field]:
                    case_warnings.append(f"Missing recommended field: {field}")
            
            # Check test_id uniqueness
            test_id = case.get('test_id')
            if test_id:
                duplicate_count = sum(1 for other_case in test_cases if other_case.get('test_id') == test_id)
                if duplicate_count > 1:
                    case_errors.append(f"Duplicate test_id: {test_id}")
            
            # Check difficulty values
            difficulty = case.get('difficulty')
            if difficulty and difficulty not in ['easy', 'medium', 'hard', 'expert']:
                case_errors.append(f"Invalid difficulty value: {difficulty}")
            
            # Check SQL syntax (basic)
            expected_sql = case.get('expected_sql')
            if expected_sql:
                if not expected_sql.strip().upper().startswith('SELECT'):
                    case_warnings.append("Expected SQL should start with SELECT")
            
            if case_errors:
                validation_errors.append({
                    'case_index': i,
                    'test_id': case.get('test_id', f'case_{i}'),
                    'errors': case_errors
                })
            
            if case_warnings:
                warnings.append({
                    'case_index': i,
                    'test_id': case.get('test_id', f'case_{i}'),
                    'warnings': case_warnings
                })
        
        # Print validation results
        print(f"\nDataset Validation Results:")
        print(f"Total test cases: {len(test_cases)}")
        print(f"Validation errors: {len(validation_errors)}")
        print(f"Warnings: {len(warnings)}")
        
        if validation_errors:
            print(f"\nERRORS:")
            for error in validation_errors:
                print(f"  Case {error['test_id']} (index {error['case_index']}):")
                for err in error['errors']:
                    print(f"    - {err}")
        
        if warnings:
            print(f"\nWARNINGS:")
            for warning in warnings:
                print(f"  Case {warning['test_id']} (index {warning['case_index']}):")
                for warn in warning['warnings']:
                    print(f"    - {warn}")
        
        if not validation_errors and not warnings:
            print("✅ Dataset validation passed!")
        elif not validation_errors:
            print("⚠️ Dataset validation passed with warnings")
        else:
            print("❌ Dataset validation failed")
        
        return len(validation_errors) == 0
    
    def _print_evaluation_summary(self, report: Dict[str, Any], verbose: bool = False):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("PERFORMANCE EVALUATION SUMMARY")
        print("="*60)
        
        summary = report['evaluation_summary']
        print(f"📊 Total Test Cases: {summary['total_test_cases']}")
        print(f"✅ Success Rate: {summary['success_rate']:.2%}")
        print(f"⚡ Execution Rate: {summary['execution_rate']:.2%}")
        
        sql_metrics = report['sql_quality_metrics']
        print(f"🔍 SQL Similarity: {sql_metrics['average_similarity']:.3f}")
        print(f"🏗️  Structure Match Rate: {sql_metrics['structure_match_rate']:.2%}")
        
        table_metrics = report['table_selection_metrics']
        print(f"🎯 Table Selection F1: {table_metrics['average_f1']:.3f}")
        print(f"📋 Primary Table Accuracy: {table_metrics['primary_table_accuracy']:.2%}")
        
        latency_metrics = report['latency_metrics']
        print(f"⏱️  Average Latency: {latency_metrics['average_latency_ms']:.1f}ms")
        print(f"📈 P95 Latency: {latency_metrics['p95_latency_ms']:.1f}ms")
        
        if verbose:
            print(f"\n📊 SQL Quality Distribution:")
            dist = sql_metrics['similarity_distribution']
            print(f"   Excellent (>0.9): {dist['excellent (>0.9)']}")
            print(f"   Good (0.7-0.9): {dist['good (0.7-0.9)']}")
            print(f"   Poor (<0.7): {dist['poor (<0.7)']}")
            
            if report['error_analysis']['error_categories']:
                print(f"\n❌ Error Categories:")
                for category, count in report['error_analysis']['most_common_errors']:
                    print(f"   {category}: {count}")
        
        print("="*60)
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("REPORT ANALYSIS SUMMARY")
        print("="*60)
        
        if 'error_analysis' in analysis:
            error_analysis = analysis['error_analysis']
            print(f"❌ Failure Rate: {error_analysis['failure_rate']:.2%}")
            print(f"📊 Top Error Patterns:")
            for pattern, count in error_analysis['top_errors']:
                print(f"   {pattern}: {count}")
        
        if 'latency_analysis' in analysis:
            latency_analysis = analysis['latency_analysis']
            print(f"⏱️  Average Latency: {latency_analysis['average_latency']:.1f}ms")
            print(f"🐌 Slow Queries (>3s): {len(latency_analysis['slow_queries'])}")
            print(f"⚡ Fast Queries (<500ms): {len(latency_analysis['fast_queries'])}")
        
        if 'accuracy_analysis' in analysis:
            accuracy_analysis = analysis['accuracy_analysis']
            sql_quality = accuracy_analysis['sql_quality']
            print(f"🎯 High SQL Similarity: {sql_quality['high_similarity']}")
            print(f"📊 Perfect Table Selection: {accuracy_analysis['table_selection']['perfect_f1']}")
        
        print("="*60)
    
    def _send_alert(self, message: str, report: Dict[str, Any]):
        """Send performance alert"""
        # This is a placeholder - implement actual alerting (email, Slack, etc.)
        logger.warning(f"PERFORMANCE ALERT: {message}")
        
        # Could integrate with:
        # - Email notifications
        # - Slack webhooks
        # - PagerDuty
        # - Custom monitoring systems
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: str):
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .success { color: green; }
                .warning { color: orange; }
                .error { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Performance Evaluation Report</h1>
            <div class="metric">
                <h2>Summary</h2>
                <p>Success Rate: {success_rate:.2%}</p>
                <p>SQL Similarity: {sql_similarity:.3f}</p>
                <p>Table Selection F1: {table_f1:.3f}</p>
                <p>Average Latency: {latency:.1f}ms</p>
            </div>
            <!-- Add more detailed sections as needed -->
        </body>
        </html>
        """.format(
            success_rate=report['evaluation_summary']['success_rate'],
            sql_similarity=report['sql_quality_metrics']['average_similarity'],
            table_f1=report['table_selection_metrics']['average_f1'],
            latency=report['latency_metrics']['average_latency_ms']
        )
        
        with open(output_path, 'w') as f:
            f.write(html_template)
    
    async def run(self):
        """Main CLI entry point"""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        try:
            if args.command == 'run':
                await self.run_evaluation(args)
            elif args.command == 'generate-sample':
                self.generate_sample_dataset(args)
            elif args.command == 'create-case':
                await self.create_test_case(args)
            elif args.command == 'compare':
                self.compare_reports(args)
            elif args.command == 'monitor':
                await self.monitor_performance(args)
            elif args.command == 'analyze':
                self.analyze_report(args)
            elif args.command == 'validate':
                self.validate_dataset(args)
            else:
                logger.error(f"Unknown command: {args.command}")
                
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            if args.command == 'run' and hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    cli = EvaluationCLI()
    asyncio.run(cli.run())