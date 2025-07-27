# Complete Performance Evaluation Framework for Enhanced NL-SQL Agent

## 🎯 **Overview**

This comprehensive evaluation framework allows you to systematically assess all aspects of your NL-SQL application's performance against golden datasets, enabling rapid iteration and design improvements.

## 📊 **Evaluation Metrics Covered**

### 1. **Query Accuracy** 
- **SQL Similarity Score**: Sequence-based comparison with golden SQL
- **Edit Distance**: Number of changes needed to match expected SQL
- **Structure Match**: Whether major SQL components (SELECT, FROM, WHERE, JOIN) align

### 2. **Result Accuracy**
- **Exact Match**: Perfect alignment of query results with expected data
- **Numerical Accuracy**: Precision of financial calculations within tolerance
- **Row Count Match**: Correct number of rows returned

### 3. **Table Selection Accuracy**
- **Precision**: Correctly identified tables / Total identified tables
- **Recall**: Correctly identified tables / Total expected tables  
- **F1 Score**: Harmonic mean of precision and recall
- **Primary Table Accuracy**: Correct primary table selection rate

### 4. **Execution Success Metrics**
- **Success Rate**: Percentage of queries that execute successfully
- **Error Categorization**: SQL generation, validation, execution, timeout errors
- **Failure Root Cause Analysis**: Detailed error pattern analysis

### 5. **Latency Metrics**
- **Total Latency**: End-to-end processing time
- **Stage Latencies**: Time spent in each pipeline stage
- **P95/P99 Latencies**: 95th and 99th percentile response times
- **Performance Distribution**: Fast vs slow query analysis

## 🚀 **Quick Start Guide**

### **Step 1: Setup**
```bash
# Install additional dependencies
pip install sqlparse pyyaml

# Set environment variables
export OPENAI_API_KEY='your-api-key'
export DATABASE_URL='postgresql://user:pass@localhost/db'
```

### **Step 2: Create Golden Dataset**
```bash
# Generate sample dataset with 50 test cases
python evaluate.py generate-sample --output golden_dataset.yaml --size 50

# Or create test cases interactively
python evaluate.py create-case --dataset golden_dataset.yaml --auto-generate
```

### **Step 3: Run Evaluation**
```bash
# Basic evaluation
python evaluate.py run --dataset golden_dataset.yaml --output report.json

# Advanced evaluation with filtering
python evaluate.py run \
  --dataset golden_dataset.yaml \
  --filter-category payments,accounts \
  --sql-threshold 0.8 \
  --table-threshold 0.9 \
  --format html
```

### **Step 4: Analyze Results**
```bash
# Quick summary
python evaluate.py analyze --report report.json --focus accuracy

# Compare two versions
python evaluate.py compare \
  --baseline baseline_report.json \
  --experimental new_report.json
```

## 📋 **Golden Dataset Structure**

Your golden dataset should include comprehensive test cases across different categories:

### **Test Case Categories**
- **Basic Queries** (20%): Simple single-table operations
- **Multi-Table Queries** (30%): Complex JOINs and relationships  
- **Payload Queries** (25%): External payload filtering
- **Aggregation Queries** (15%): GROUP BY, SUM, AVG operations
- **Edge Cases** (10%): Error conditions and boundary cases

### **Sample Test Case**
```yaml
- test_id: "multi_001"
  natural_query: "Show my spending by category"
  user_id: "user_001"
  payload:
    CIN: 22
    sort_code: 123456
  expected_sql: "SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id WHERE p.customer_id = 22 AND p.user_id = 'user_001' GROUP BY c.category_name"
  expected_tables: ["payments", "categories"]
  expected_primary_table: "payments"
  expected_result:
    - category_name: "Food & Dining"
      sum: 450.25
    - category_name: "Transportation" 
      sum: 125.50
  expected_numerical_values:
    sum: 575.75
  difficulty: "medium"
  category: "multi_table_queries"
  tags: ["payments", "categories", "aggregation"]
```

## 🔄 **Continuous Integration Workflow**

### **CI/CD Pipeline Integration**
```bash
# Add to your CI pipeline
python evaluate.py run \
  --dataset production_golden_set.yaml \
  --sql-threshold 0.85 \
  --output ci_report.json

# Fail build if performance regresses
SUCCESS_RATE=$(jq '.evaluation_summary.success_rate' ci_report.json)
if (( $(echo "$SUCCESS_RATE < 0.85" | bc -l) )); then
  echo "❌ Performance regression detected: $SUCCESS_RATE"
  exit 1
fi
```

### **Automated Monitoring**
```bash
# Monitor production performance every hour
python evaluate.py monitor \
  --dataset production_dataset.yaml \
  --interval 3600 \
  --alert-threshold 0.8 \
  --output-dir monitoring_reports
```

## 📈 **A/B Testing & Experimentation**

### **Compare System Versions**
```python
# Test baseline vs experimental
baseline_report = await evaluate_version("baseline_config")
experimental_report = await evaluate_version("experimental_config")

# Automated comparison
python evaluate.py compare \
  --baseline baseline_report.json \
  --experimental experimental_report.json \
  --output comparison.json

# Decision making
improvement = experimental_report['success_rate'] - baseline_report['success_rate']
if improvement > 0.05:  # 5% improvement threshold
    deploy_experimental_version()
```

### **Feature Flag Testing**
```python
# Test with different feature configurations
configs = [
    {"retry_logic": True, "max_retries": 3},
    {"retry_logic": True, "max_retries": 5},
    {"retry_logic": False}
]

for config in configs:
    report = await evaluate_with_config(config)
    print(f"Config {config}: {report['success_rate']:.2%}")
```

## 🎯 **Performance Optimization Workflow**

### **1. Identify Bottlenecks**
```bash
# Focus on latency analysis
python evaluate.py analyze --report report.json --focus latency

# Identify slow queries
python -c "
import json
with open('report.json') as f:
    report = json.load(f)
slow_queries = [r for r in report['detailed_results'] if r.get('total_latency', 0) > 3000]
print(f'Slow queries: {len(slow_queries)}')
for q in slow_queries[:5]:
    print(f'  {q[\"test_id\"]}: {q[\"total_latency\"]}ms')
"
```

### **2. Error Pattern Analysis**
```bash
# Analyze failure patterns
python evaluate.py analyze --report report.json --focus errors

# Common failure categories:
# - sql_generation: Improve prompts/examples
# - sql_validation: Update validation rules  
# - sql_execution: Check database schema/data
# - table_detection: Enhance keyword mappings
```

### **3. Iterative Improvement**
```python
# Evaluation-driven development cycle
async def improvement_cycle():
    baseline = await run_evaluation("current_system")
    
    # Make improvements
    update_prompts()
    enhance_table_mapping()
    optimize_sql_generation()
    
    # Re-evaluate
    improved = await run_evaluation("improved_system")
    
    # Compare results
    if improved['success_rate'] > baseline['success_rate']:
        deploy_improvements()
        update_golden_dataset()  # Add new challenging cases
    else:
        revert_changes()
```

## 🔍 **Advanced Analysis Features**

### **Performance Regression Detection**
```python
# Historical performance tracking
reports = load_historical_reports()
success_rates = [r['evaluation_summary']['success_rate'] for r in reports]

# Detect trends
import numpy as np
trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
if trend < -0.01:  # Declining performance
    alert_team("Performance regression detected")
```

### **Test Case Difficulty Analysis**
```bash
# Performance by difficulty level
python -c "
import json
with open('report.json') as f:
    report = json.load(f)

difficulty_performance = {}
for result in report['detailed_results']:
    test_case = find_test_case(result['test_id'])
    difficulty = test_case.get('difficulty', 'unknown')
    
    if difficulty not in difficulty_performance:
        difficulty_performance[difficulty] = []
    difficulty_performance[difficulty].append(result['success'])

for difficulty, results in difficulty_performance.items():
    success_rate = sum(results) / len(results)
    print(f'{difficulty}: {success_rate:.2%} ({len(results)} cases)')
"
```

### **Payload Impact Analysis**
```python
# Compare performance with vs without payloads
payload_cases = [r for r in results if has_payload(r['test_id'])]
no_payload_cases = [r for r in results if not has_payload(r['test_id'])]

payload_success = sum(r['success'] for r in payload_cases) / len(payload_cases)
no_payload_success = sum(r['success'] for r in no_payload_cases) / len(no_payload_cases)

print(f"With payload: {payload_success:.2%}")
print(f"Without payload: {no_payload_success:.2%}")
print(f"Payload impact: {payload_success - no_payload_success:+.2%}")
```

## 🛠️ **Customization & Extension**

### **Custom Metrics**
```python
class CustomEvaluator(PerformanceEvaluator):
    def evaluate_custom_metric(self, result: EvaluationResult, test_case: GoldenTestCase):
        # Add your custom evaluation logic
        if "financial" in test_case.tags:
            # Special handling for financial queries
            accuracy = self.evaluate_financial_accuracy(result, test_case)
            result.custom_metrics = {"financial_accuracy": accuracy}
```

### **Domain-Specific Validation**
```python
def validate_financial_sql(sql: str) -> bool:
    """Custom validation for financial queries"""
    required_patterns = [
        r'SUM\s*\(\s*amount\s*\)',  # Financial aggregations
        r'WHERE.*amount\s*[><]=',   # Amount filtering
    ]
    
    for pattern in required_patterns:
        if re.search(pattern, sql, re.IGNORECASE):
            return True
    return False
```

### **Integration with External Tools**
```python
# Slack notifications
def send_slack_alert(report):
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    message = f"Performance Alert: Success rate dropped to {report['success_rate']:.2%}"
    requests.post(webhook_url, json={"text": message})

# Database logging
def log_to_database(report):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluation_history (timestamp, success_rate, avg_latency, report_data)
        VALUES (%s, %s, %s, %s)
    """, (datetime.now(), report['success_rate'], report['avg_latency'], json.dumps(report)))
    conn.commit()
```

## 📊 **Reporting & Visualization**

### **HTML Reports**
```bash
# Generate interactive HTML report
python evaluate.py run --dataset golden_dataset.yaml --format html --output report.html
```

### **Custom Dashboards**
```python
# Create dashboard-ready data
def create_dashboard_data(report):
    return {
        "success_rate_trend": extract_trend_data(report),
        "latency_distribution": create_latency_histogram(report),
        "error_breakdown": create_error_pie_chart(report),
        "table_accuracy_heatmap": create_table_heatmap(report)
    }
```

## 🔄 **Best Practices**

### **Golden Dataset Management**
1. **Version Control**: Store golden datasets in Git with proper versioning
2. **Regular Updates**: Add new challenging cases as system improves
3. **Balanced Coverage**: Ensure all query types and edge cases are covered
4. **Review Process**: Human review of all expected results before inclusion

### **Evaluation Frequency**
- **Pre-commit**: Run subset of critical test cases
- **CI/CD**: Full evaluation on feature branches
- **Daily**: Comprehensive evaluation on main branch
- **Production**: Continuous monitoring with alerts

### **Performance Baselines**
- **Success Rate**: >85% for production deployment
- **SQL Similarity**: >0.8 for acceptable quality
- **Table F1**: >0.9 for accurate table selection
- **P95 Latency**: <3000ms for good user experience

### **Debugging Failed Cases**
```python
# Analyze specific failures
def debug_failure(test_id, report):
    failed_case = find_result(test_id, report)
    test_case = find_test_case(test_id)
    
    print(f"Test ID: {test_id}")
    print(f"Query: {test_case['natural_query']}")
    print(f"Expected SQL: {test_case['expected_sql']}")
    print(f"Generated SQL: {failed_case['generated_sql']}")
    print(f"Error: {failed_case['error_message']}")
    print(f"SQL Similarity: {failed_case['sql_similarity_score']}")
    
    # Suggest improvements
    if failed_case['sql_similarity_score'] < 0.5:
        print("💡 Consider updating prompt examples")
    if failed_case['table_f1'] < 0.8:
        print("💡 Consider updating table keyword mappings")
```

## 🎯 **Success Metrics & KPIs**

Track these key performance indicators:

- **Overall Success Rate**: Target >90%
- **SQL Quality Score**: Target >0.85  
- **Table Selection Accuracy**: Target >0.95
- **P95 Latency**: Target <2000ms
- **Error Rate by Category**: <5% per category
- **Regression Detection**: <24 hours to identify issues

This framework provides everything you need to maintain high-quality performance while rapidly iterating on your NL-SQL system! 🚀