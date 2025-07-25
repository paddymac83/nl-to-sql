# Enhanced Multi-Table NL to SQL Agent with LangGraph

A sophisticated Natural Language to SQL query generator that intelligently selects and queries multiple database tables based on input keywords and external payloads. Built with LangGraph, OpenAI GPT, and PostgreSQL for robust workflow orchestration.

## ✨ Features

- 🧠 **Intelligent Table Detection**: Automatically detects relevant tables based on query keywords and payload data
- 🔗 **Multi-Table Queries**: Supports complex queries spanning multiple related tables  
- 👤 **User Context Awareness**: Filters data based on user-specific context when appropriate
- 📦 **External Payload Support**: Accepts JSON payloads for precise filtering (CIN, sort_code, account_number)
- 🔄 **LangGraph Workflows**: Robust async workflow orchestration with retry logic
- 🛡️ **SQL Injection Protection**: Validates and sanitizes all generated queries
- 📊 **Visualization-Ready Output**: Formats data for external visualization systems
- 🎯 **Smart Chart Suggestions**: Recommends appropriate chart types based on data structure

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  LangGraph       │───▶│  Table Mapping  │
│   + Payload     │    │  Workflow        │    │  & Detection    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Formatted Data  │◀───│   SQL Generation │◀───│ Schema Analysis │
│ for Viz System  │    │   & Execution    │    │ & Validation    │
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
git clone https://github.com/paddymac83/nl-to-sql.git
cd nl-to-sql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and database URL
```

### Database Setup

```bash
# Install PostgreSQL and create database
sudo -u postgres psql
CREATE DATABASE nlsql_db;
CREATE USER nlsql_user WITH ENCRYPTED PASSWORD 'nlsql_password';
GRANT ALL PRIVILEGES ON DATABASE nlsql_db TO nlsql_user;
\q

# Run enhanced database setup script
psql -U nlsql_user -d nlsql_db -f setup_database_enhanced.sql
```

### Usage

```bash
# Verify setup
python config_and_runner.py setup

# Test payload functionality
python config_and_runner.py setup --test-payload

# Run the application
python config_and_runner.py run

# Run with demo queries including payloads
python config_and_runner.py run --demo

# Or run directly
python multi_table_nlsql.py
```

## 💬 Example Queries

### Basic Queries
- "How much did I spend last month?"
- "Show me all payments over $100"
- "What's my account balance?"

### Multi-Table Queries
- "Show my spending by category and merchant"
- "Which users spend most on food category?"
- "Compare budget vs actual spending by category"

### Payload-Enhanced Queries
- "Show payments `{'CIN': 22, 'sort_code': 123456}`"
- "Account balance `{'account_number': 900914}`"
- "Transaction history `{'CIN': 22, 'account_number': 900914}`"
- "User spending analysis `{'user_id': 'user_001'}`"

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

## 🔄 LangGraph Workflow

The system uses LangGraph for robust workflow orchestration:

1. **Query Analysis** → Detect relevant tables and payload context
2. **Schema Retrieval** → Get table structures and relationships
3. **SQL Generation** → Create optimized queries with payload filters
4. **Validation** → Check for safety and correctness
5. **Execution** → Run queries against database
6. **Formatting** → Prepare data for external visualization

### Workflow Features

- **Async Processing**: Non-blocking query execution
- **Retry Logic**: Automatic retry on validation failures
- **Error Handling**: Comprehensive error recovery
- **State Management**: Persistent workflow state

## 📊 Visualization Output

The system formats data for external visualization applications:

```json
{
  "status": "success",
  "data": [
    {"category": "Food", "amount": 150.50},
    {"category": "Transport", "amount": 89.25}
  ],
  "metadata": {
    "row_count": 2,
    "column_count": 2,
    "columns": [
      {
        "name": "category",
        "type": "object",
        "sample_values": ["Food", "Transport"]
      }
    ],
    "suggested_charts": ["bar_chart", "pie_chart"],
    "query_info": {
      "query": "Show spending by category",
      "sql_query": "SELECT category_name, SUM(amount) FROM ...",
      "relevant_tables": ["payments", "categories"]
    }
  }
}
```

## 🧪 Testing

### Run All Tests
```bash
python config_and_runner.py test
```

### Run Specific Test Types
```bash
# Unit tests only
python config_and_runner.py test --unit

# Integration tests only
python config_and_runner.py test --integration

# LangGraph workflow tests
python config_and_runner.py test --langgraph

# Specific test
python config_and_runner.py test -k "test_payload"
```

### Test Coverage
```bash
# With coverage report
python config_and_runner.py test --coverage

# View HTML coverage report
open htmlcov/index.html
```

## 📁 Project Structure

```
├── multi_table_nlsql.py           # Main application with LangGraph
├── test_multi_table_nlsql.py      # Comprehensive test suite
├── config_and_runner.py           # Enhanced configuration and test runner
├── setup_database_enhanced.sql    # Enhanced database setup script
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── pytest.ini                     # Test configuration
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db

# Optional
LOG_LEVEL=INFO
OUTPUT_DIR=outputs
```

### Adding New Tables

1. **Update TableMapping**:
```python
'new_table': {
    'keywords': ['keyword1', 'keyword2'],
    'description': 'Table description',
    'primary_key': 'id',
    'user_column': 'user_id',  # if user-specific
    'payload_columns': ['customer_id', 'account_id']  # for payload filtering
}
```

2. **Add Join Patterns**:
```python
('new_table', 'existing_table'): 'new_table.fk = existing_table.pk'
```

3. **Update Database Schema**:
```sql
CREATE TABLE new_table (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    -- other columns
);
```

## 🛠️ Development

### Code Quality
```bash
# Run linting
python config_and_runner.py lint

# Auto-fix formatting
python config_and_runner.py lint --fix
```

### Key Components

- **InputPayload**: Pydantic model for external payload validation
- **TableMapping**: Enhanced keyword detection and payload column mapping