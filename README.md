# Enhanced Multi-Table NL to SQL Agent with LangGraph

A sophisticated Natural Language to SQL query generator that intelligently selects and queries multiple database tables based on input keywords and external payloads. Built with LangGraph, OpenAI GPT, and PostgreSQL for robust workflow orchestration with comprehensive JSON result storage and RESTful API interface.

## ✨ Features

- 🧠 **Intelligent Table Detection**: Automatically detects relevant tables based on query keywords and payload data
- 🔗 **Multi-Table Queries**: Supports complex queries spanning multiple related tables with automatic JOIN optimization
- 👤 **User Context Awareness**: Filters data based on user-specific context when appropriate
- 📦 **External Payload Support**: Accepts JSON payloads for precise filtering (CIN, sort_code, account_number)
- 🔄 **LangGraph Workflows**: Robust async workflow orchestration with retry logic and error recovery
- 💾 **JSON Result Storage**: Comprehensive query result storage in PostgreSQL with JSONB fields
- 🛡️ **SQL Injection Protection**: Multi-layer validation and sanitization of all generated queries
- 📊 **Visualization-Ready Output**: Formats data for external visualization systems with chart suggestions
- 🚀 **RESTful API Interface**: Complete API for external application integration
- 📈 **Query Analytics**: Performance tracking, success rates, and execution statistics
- 🔄 **Query Replay**: Store and replay previous queries with full context preservation
- 🎯 **Smart Chart Suggestions**: AI-powered recommendations for optimal data visualization

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  LangGraph       │───▶│  Table Mapping  │
│   + Payload     │    │  Workflow        │    │  & Detection    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ PostgreSQL      │◀───│   SQL Generation │◀───│ Schema Analysis │
│ Result Storage  │    │   & Execution    │    │ & Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ RESTful API     │◀───│ Formatted Data   │───▶│ Visualization   │
│ Interface       │    │ + Analytics      │    │ Systems         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Supported Tables

| Table | Keywords | Description | User-Specific | Payload Columns |
|-------|----------|-------------|---------------|-----------------|
| **payments** | payment, spend, transaction, expense | Payment transactions and spending data | ✅ | customer_id, account_number, sort_code, user_id |
| **accounts** | account, balance, statement, bank | Bank account information | ✅ | account_number, sort_code, customer_id, user_id |
| **categories** | category, type, classification | Spending categories and types | ❌ | - |
| **merchants** | merchant, store, shop, vendor | Store and vendor information | ❌ | - |
| **users** | user, customer, account | User account information | ✅ | customer_id, user_id |
| **budgets** | budget, limit, target, goal | Budget limits and financial goals | ✅ | user_id |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- OpenAI API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/enhanced-nl-to-sql.git
cd enhanced-nl-to-sql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
export OPENAI_API_KEY='your-openai-api-key-here'
export DATABASE_URL='postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db'
```

### Database Setup

```bash
# Install PostgreSQL and create database
sudo -u postgres psql
CREATE DATABASE nlsql_db;
CREATE USER nlsql_user WITH ENCRYPTED PASSWORD 'nlsql_password';
GRANT ALL PRIVILEGES ON DATABASE nlsql_db TO nlsql_user;
\q
```

### System Health Check & Setup

```bash
# Run comprehensive health check
python multi_table_nlsql.py health

# Initialize system with sample data
python multi_table_nlsql.py setup

# Start interactive mode
python multi_table_nlsql.py

# API example mode
python multi_table_nlsql.py api
```

## 💬 Example Queries

### Basic Queries
- "How much did I spend last month?"
- "Show me all payments over $100"
- "What's my account balance?"
- "Which category do I spend the most on?"

### Multi-Table Queries
- "Show my spending by category and merchant"
- "Which users spend most on food category?"
- "Compare budget vs actual spending by category"
- "Show merchant performance across all users"

### Payload-Enhanced Queries
```bash
# Customer-specific queries
"Show payments {'CIN': 22, 'sort_code': 123456}"

# Account-specific queries  
"Account balance {'account_number': 900914}"

# Multi-field filtering
"Transaction history {'CIN': 22, 'account_number': 900914}"

# User analysis
"User spending analysis {'user_id': 'user_001'}"
```

### Interactive Commands
- `help` - Show example queries and commands
- `tables` - Display table information and keywords  
- `user` - Change user context
- `history` - View recent query history
- `stats` - Show query performance statistics
- `replay <id>` - Replay a previous query by ID

## 📦 Payload Format

The system accepts JSON payloads for precise filtering:

```json
{
  "CIN": 22,
  "sort_code": 123456,
  "account_number": 900914,
  "user_id": "user_001"
}
```

### Payload Field Mapping

| Payload Field | Database Column | Description |
|---------------|-----------------|-------------|
| `CIN` | `customer_id` | Customer Identification Number |
| `sort_code` | `sort_code` | Bank sort code |
| `account_number` | `account_number` | Account number |
| `user_id` | `user_id` | User identifier |

### Flexible Payload Formats

```python
# Both formats supported
{'CIN': 22, 'sort_code': 123456}    # Python dict style
{"CIN": 22, "sort_code": 123456}    # JSON style
```

## 🔄 Enhanced LangGraph Workflow

The system uses LangGraph for robust workflow orchestration:

1. **Query Analysis** → Detect relevant tables and payload context
2. **Schema Retrieval** → Get table structures and relationships  
3. **SQL Generation** → Create optimized queries with payload filters
4. **Validation** → Multi-layer safety and correctness checking
5. **Execution** → Run queries against database with error handling
6. **Result Storage** → Store results in PostgreSQL with full metadata
7. **Formatting** → Prepare data for external visualization systems

### Advanced Workflow Features

- **Async Processing**: Non-blocking query execution with concurrent support
- **Intelligent Retry Logic**: Automatic retry on validation failures with backoff
- **Comprehensive Error Handling**: Graceful degradation and error recovery
- **State Management**: Persistent workflow state with session tracking
- **Query Deduplication**: SHA-256 hashing to prevent duplicate processing
- **Performance Monitoring**: Real-time execution time tracking

## 💾 Query Result Storage

All queries are stored in PostgreSQL with comprehensive metadata:

### Storage Schema
```sql
CREATE TABLE query_results (
    result_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    user_query TEXT NOT NULL,
    input_payload JSONB,           -- Input parameters as JSON
    sql_query TEXT,
    relevant_tables TEXT[],
    primary_table VARCHAR(100),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    result_data JSONB,            -- Raw query results as JSON
    formatted_data JSONB,         -- Visualization-ready data
    row_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100),      -- Session tracking
    query_hash VARCHAR(64)        -- Deduplication hash
);
```

### Query Analytics Features
- **Success/failure tracking** with detailed error logging
- **Performance metrics** (execution time, row counts)
- **User activity patterns** and query frequency analysis
- **Session-based query grouping** for workflow tracking
- **Query replay functionality** with full context restoration

## 🚀 RESTful API Interface

Complete API for external application integration:

### API Endpoints

#### Process Query
```python
# POST /query
{
    "query": "Show spending by category",
    "user_id": "user_001",
    "payload": {"CIN": 22, "sort_code": 123456},
    "session_id": "optional_session_id",
    "store_result": true
}
```

#### Query History
```python
# GET /history?user_id=user_001&limit=10
api.get_query_history(user_id='user_001', limit=10)
```

#### Query Statistics  
```python
# GET /stats?user_id=user_001
api.get_query_statistics(user_id='user_001')
```

#### Table Schema
```python
# GET /schema?tables=payments,accounts
api.get_table_schema(['payments', 'accounts'])
```

### API Usage Example

```python
import asyncio
from multi_table_nlsql import create_api

async def example_usage():
    # Create API instance
    api = create_api()
    
    # Process query with payload
    result = await api.query({
        'query': 'Show my recent payments',
        'user_id': 'user_001',
        'payload': {'CIN': 22, 'sort_code': 123456},
        'session_id': 'web_session_123'
    })
    
    if result['success']:
        print(f"SQL: {result['sql_query']}")
        print(f"Rows: {result['row_count']}")
        print(f"Data: {result['formatted_data']}")
    
    # Get analytics
    stats = api.get_query_statistics('user_001')
    print(f"Success rate: {stats['data']['success_rate']}%")

# Run the example
asyncio.run(example_usage())
```

## 📊 Enhanced Visualization Output

The system formats data for external visualization applications with AI-powered suggestions:

```json
{
  "status": "success",
  "data": [
    {"category": "Food", "amount": 150.50, "transaction_count": 12},
    {"category": "Transport", "amount": 89.25, "transaction_count": 8}
  ],
  "metadata": {
    "row_count": 2,
    "column_count": 3,
    "columns": [
      {
        "name": "category",
        "type": "object", 
        "sample_values": ["Food", "Transport"]
      },
      {
        "name": "amount",
        "type": "float64",
        "sample_values": [150.50, 89.25]
      }
    ],
    "suggested_charts": ["bar_chart", "pie_chart", "donut_chart"],
    "query_info": {
      "query": "Show spending by category",
      "sql_query": "SELECT c.category_name, SUM(p.amount), COUNT(*) FROM...",
      "relevant_tables": ["payments", "categories"],
      "primary_table": "payments",
      "user_id": "user_001",
      "input_payload": {"CIN": 22}
    }
  }
}
```

### Smart Chart Suggestions

The system analyzes data characteristics to suggest optimal visualizations:

| Data Pattern | Suggested Charts |
|--------------|------------------|
| Categorical + Numeric | Bar Chart, Horizontal Bar, Pie Chart |
| Time Series + Numeric | Line Chart, Area Chart |
| Multiple Numeric | Scatter Plot, Bubble Chart |
| Single Numeric | Histogram, Box Plot |
| Large Datasets (>20 rows) | Aggregated charts, Summary tables |

## 🧪 Comprehensive Testing

### Test Coverage

The project includes 400+ test cases covering:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing  
- **LangGraph Tests**: Workflow orchestration validation
- **API Tests**: RESTful interface testing
- **Security Tests**: SQL injection and input validation
- **Performance Tests**: Large dataset and concurrent processing
- **Edge Case Tests**: Error conditions and boundary cases

### Running Tests

```bash
# Run all tests
python -m pytest test_multi_table_nlsql.py -v

# Run with coverage
python -m pytest --cov=multi_table_nlsql --cov-report=html

# Run specific test categories
pytest -m "not performance" -v          # Skip performance tests
pytest -m security -v                   # Security tests only
pytest -k "test_payload" -v             # Payload-related tests

# Run tests with detailed output
pytest --tb=short --durations=10 -v
```

### Test Categories

- **🔒 Security Tests**: SQL injection prevention, input sanitization
- **⚡ Performance Tests**: Large dataset handling, concurrent processing
- **🔄 Integration Tests**: Full workflow validation with real data
- **📊 Data Tests**: Various data types, NULL handling, precision
- **🌐 API Tests**: RESTful interface and error handling
- **🧩 Edge Cases**: Malformed inputs, connection failures

## 📁 Enhanced Project Structure

```
├── multi_table_nlsql.py           # Main application with LangGraph & API
├── test_multi_table_nlsql.py      # Comprehensive test suite (400+ tests)
├── requirements.txt               # Enhanced dependencies with versions
├── .env.example                   # Environment template
├── README.md                      # This enhanced documentation
├── examples/
│   ├── api_usage.py               # API integration examples
│   ├── batch_processing.py        # Bulk query processing
│   └── visualization_examples.py  # Chart integration examples
├── docs/
│   ├── API.md                     # API documentation
│   ├── DEPLOYMENT.md              # Production deployment guide
│   └── TROUBLESHOOTING.md         # Common issues and solutions
└── scripts/
    ├── setup_database.sql         # Database initialization
    ├── sample_data.sql            # Test data insertion
    └── health_check.py            # System diagnostics
```

## 🔧 Advanced Configuration

### Environment Variables

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db

# Optional Configuration
LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR
MAX_RETRIES=3                      # SQL generation retry limit
QUERY_TIMEOUT=30                   # Query timeout in seconds
CACHE_ENABLED=true                 # Enable query result caching
MAX_ROWS_RETURN=10000             # Maximum rows per query
ENABLE_ANALYTICS=true              # Enable query analytics storage

# Performance Tuning
DB_POOL_SIZE=10                    # Database connection pool size
ASYNC_WORKERS=5                    # Concurrent query processing workers
RATE_LIMIT_QUERIES=100             # Queries per minute limit
```

### Custom Table Configuration

#### Adding New Tables

1. **Update TableMapping**:
```python
'custom_table': {
    'keywords': ['custom', 'specific', 'keywords'],
    'description': 'Custom table description',
    'primary_key': 'id',
    'user_column': 'user_id',  # if user-specific
    'payload_columns': ['customer_id', 'custom_field']
}
```

2. **Add Join Patterns**:
```python
('custom_table', 'payments'): 'custom_table.payment_id = payments.payment_id',
('custom_table', 'users'): 'custom_table.user_id = users.user_id'
```

3. **Update Database Schema**:
```sql
CREATE TABLE custom_table (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    customer_id INTEGER,
    custom_field VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for performance
CREATE INDEX idx_custom_table_user_id ON custom_table(user_id);
CREATE INDEX idx_custom_table_customer_id ON custom_table(customer_id);
```

## 🛠️ Development & Deployment

### Development Setup

```bash
# Development mode with hot reload
python multi_table_nlsql.py --debug

# Code quality checks
python -m black multi_table_nlsql.py
python -m flake8 multi_table_nlsql.py
python -m mypy multi_table_nlsql.py

# Security scanning
python -m bandit multi_table_nlsql.py
```

### Production Deployment

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "multi_table_nlsql.py", "api", "--host", "0.0.0.0", "--port", "8000"]
```

#### Environment-Specific Configuration
```bash
# Production
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export ENABLE_ANALYTICS=true

# Staging  
export ENVIRONMENT=staging
export LOG_LEVEL=INFO
export ENABLE_ANALYTICS=false

# Development
export ENVIRONMENT=development  
export LOG_LEVEL=DEBUG
export ENABLE_ANALYTICS=true
```

## 🔍 Monitoring & Analytics

### Built-in Analytics

The system provides comprehensive analytics:

- **Query Performance**: Execution times, success rates, error patterns
- **User Behavior**: Query frequency, preferred tables, common patterns
- **System Health**: Database performance, API response times
- **Data Insights**: Most queried data, popular visualizations

### Accessing Analytics

```python
# Get user statistics
stats = agent.result_storage.get_query_statistics('user_001')
print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['successful_queries'] / stats['total_queries'] * 100:.1f}%")
print(f"Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")

# Get system-wide analytics  
system_stats = agent.result_storage.get_query_statistics()
print(f"Unique users: {system_stats['unique_users']}")
print(f"Total queries processed: {system_stats['total_queries']}")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Key Areas for Contribution

- **New Table Integrations**: Add support for additional database tables
- **Visualization Enhancements**: Improve chart suggestion algorithms  
- **Performance Optimizations**: Database query optimization, caching strategies
- **Security Enhancements**: Additional validation layers, audit logging
- **API Extensions**: New endpoints, webhook support, batch processing
- **Documentation**: Examples, tutorials, deployment guides

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for robust workflow orchestration
- **OpenAI** for powerful language model capabilities  
- **PostgreSQL** for reliable data storage and JSONB support
- **Pydantic** for data validation and serialization
- **FastAPI** for API framework inspiration

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/enhanced-nl-to-sql/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/enhanced-nl-to-sql/discussions)
- **Email**: support@your-domain.com

---

**Built with ❤️ for the data community. Transform natural language into powerful SQL insights with intelligent multi-table detection and comprehensive analytics.**