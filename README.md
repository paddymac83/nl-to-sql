# Enhanced Multi-Table NL to SQL Agent

A sophisticated Natural Language to SQL query generator that intelligently selects and queries multiple database tables based on input keywords and context. Built with LangGraph, OpenAI, and PostgreSQL.

## ✨ Features

- **🧠 Intelligent Table Detection**: Automatically detects relevant tables based on query keywords
- **🔗 Multi-Table Queries**: Supports complex queries spanning multiple related tables
- **👤 User Context Awareness**: Filters data based on user-specific context when appropriate
- **📊 Smart Visualizations**: Generates context-aware charts and graphs
- **🛡️ SQL Injection Protection**: Validates and sanitizes all generated queries
- **🔄 Retry Logic**: Automatically retries failed queries with improvements
- **📈 Performance Monitoring**: Tracks query execution and provides detailed logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Table Mapping   │───▶│  Schema Info    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Visualization   │◀───│   LLM Service    │◀───│ Database Mgr    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Supported Tables

| Table | Keywords | Description | User-Specific |
|-------|----------|-------------|---------------|
| **payments** | payment, spend, transaction, expense | Payment transactions and spending data | ✅ |
| **categories** | category, type, classification | Spending categories and types | ❌ |
| **merchants** | merchant, store, shop, vendor | Store and vendor information | ❌ |
| **users** | user, customer, account | User account information | ✅ |
| **budgets** | budget, limit, target, goal | Budget limits and financial goals | ✅ |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-table-nlsql-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db
```

### 3. Database Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE nlsql_db;
CREATE USER nlsql_user WITH ENCRYPTED PASSWORD 'nlsql_password';
GRANT ALL PRIVILEGES ON DATABASE nlsql_db TO nlsql_user;
\q

# Run database setup script
psql -U nlsql_user -d nlsql_db -f setup_database.sql
```

### 4. Verify Setup

```bash
python config_and_runner.py setup
```

### 5. Run the Application

```bash
# Interactive mode
python config_and_runner.py run

# Or directly
python multi_table_nlsql.py
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

# Specific test
python config_and_runner.py test -k "test_table_detection"
```

### Test Coverage
```bash
# With coverage report
python config_and_runner.py test --coverage

# View HTML coverage report
open htmlcov/index.html
```

## 💬 Example Queries

### Payment Queries
```
"How much did I spend last month?"
"Show me all payments over $100"
"What's my average transaction amount?"
```

### Category Analysis
```
"Show me spending by category"
"Which category do I spend the most on?"
"Compare food vs entertainment spending"
```

### Merchant Analysis
```
"Which stores do I visit most?"
"Show me all Netflix payments"
"Compare spending at different merchants"
```

### Multi-Table Queries
```
"Show my spending by category and merchant"
"Which users spend most on food category?"
"Compare budget vs actual spending by category"
```

### User Comparisons
```
"Show top spending users"
"Compare my spending to average user"
"Who are the highest spending customers?"
```

## 🔧 Configuration

### Table Keyword Mapping

Add new tables or modify keywords in `multi_table_nlsql.py`:

```python
self.table_keywords = {
    'your_table': {
        'keywords': ['keyword1', 'keyword2'],
        'description': 'Table description',
        'primary_key': 'id_column',
        'user_column': 'user_id'  # or None
    }
}
```

### Join Patterns

Define relationships between tables:

```python
self.join_patterns = {
    ('table1', 'table2'): 'table1.foreign_key = table2.primary_key'
}
```

## 🛠️ Development

### Code Structure
```
├── multi_table_nlsql.py       # Main application code
├── test_multi_table_nlsql.py  # Comprehensive test suite
├── config_and_runner.py       # Configuration and test runner
├── setup_database.sql         # Database setup script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

### Key Components

- **TableMapping**: Manages keyword detection and table relationships
- **AgentState**: Pydantic model for workflow state management
- **DatabaseManager**: Handles all database operations with validation
- **LLMService**: Interfaces with OpenAI for SQL generation
- **VisualizationService**: Creates context-aware charts and graphs
- **NLToSQLAgent**: Main orchestrator using LangGraph workflows

### Adding New Tables

1. **Update TableMapping**:
```python
'new_table': {
    'keywords': ['keyword1', 'keyword2'],
    'description': 'Table description',
    'primary_key': 'id',
    'user_column': 'user_id'  # if user-specific
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
    -- other columns
);
```

4. **Add Tests**:
```python
def test_new_table_detection(self):
    tables = self.table_mapping.detect_relevant_tables("new table query")
    assert 'new_table' in tables
```

### Code Quality

```bash
# Run linting
python config_and_runner.py lint

# Auto-fix formatting
python config_and_runner.py lint --fix

# Type checking
mypy multi_table_nlsql.py
```

## 📊 Monitoring and Logging

The application provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Log levels:
- **INFO**: Query processing steps
- **WARNING**: Non-critical issues
- **ERROR**: Query failures and exceptions
- **DEBUG**: Detailed execution information

## 🔒 Security Features

- **SQL Injection Protection**: Validates queries for dangerous keywords
- **Query Sanitization**: Removes placeholders and comments
- **User Context Isolation**: Ensures users only see their own data
- **Environment Variable Security**: Sensitive data stored in environment variables

## 🚨 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Ensure your `.env` file contains a valid OpenAI API key
   - Check that the `.env` file is in the project root

2. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check database credentials in `.env`
   - Ensure database and user exist

3. **Table Not Found**
   - Run the database setup script
   - Verify table names match the configuration

4. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

### Debug Mode

Run with detailed logging:
```bash
export LOG_LEVEL=DEBUG
python multi_table_nlsql.py
```

### Test Database Connection
```python
from multi_table_nlsql import DatabaseManager
db = DatabaseManager("your_database_url")
db.connect()
print("✅ Database connection successful")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Workflow

```bash
# Setup development environment
python config_and_runner.py setup

# Run tests during development
python config_and_runner.py test --unit

# Check code quality
python config_and_runner.py lint

# Test with demo queries
python config_and_runner.py run --demo
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- Uses [OpenAI GPT-4](https://openai.com) for natural language understanding
- Database operations powered by [SQLAlchemy](https://sqlalchemy.org)
- Visualizations created with [Matplotlib](https://matplotlib.org) and [Seaborn](https://seaborn.pydata.org)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test cases for examples
3. Open an issue with detailed information about your setup and the problem

---

**Happy Querying! 🎉**