"""
Structured Output Implementation with Enhanced Validation and Monitoring
Uses Pydantic models and OpenAI function calling for precise control
"""

import os
from dotenv import load_dotenv
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import json
from dataclasses import dataclass, field
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
import time
import hashlib

# Monitoring and Evaluation
from abc import ABC, abstractmethod

class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class QueryType(str, Enum):
    AGGREGATE = "aggregate"
    FILTER = "filter"
    JOIN = "join"
    TIME_SERIES = "time_series"
    RANKING = "ranking"

class SQLQuery(BaseModel):
    """Structured SQL query with metadata"""
    query: str = Field(..., description="The SQL query string")
    query_type: QueryType = Field(..., description="Type of SQL query")
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    tables_used: List[str] = Field(default_factory=list, description="Tables referenced in query")
    columns_used: List[str] = Field(default_factory=list, description="Columns referenced in query")
    has_aggregation: bool = Field(default=False, description="Whether query uses aggregation")
    has_joins: bool = Field(default=False, description="Whether query uses joins")
    estimated_rows: Optional[int] = Field(None, description="Estimated result set size")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in query correctness")
    
    def get(self, key: str, default=None):
        """Dictionary-like access for backward compatibility"""
        return getattr(self, key, default)
    
    @field_validator('query')
    def validate_sql_safety(cls, v):
        """Validate SQL for safety"""
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC']
        query_upper = v.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
        
        if not v.strip().upper().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
            
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_consistency(cls, values):
        """Validate internal consistency"""
        if isinstance(values, dict):
            query = values.get('query', '').upper()
            
            # Auto-detect features
            if any(word in query for word in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(', 'GROUP BY']):
                values['has_aggregation'] = True
                
            if 'JOIN' in query:
                values['has_joins'] = True
                
        return values

class QueryExecutionResult(BaseModel):
    """Structured execution result with metadata"""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    execution_time_ms: float = 0
    error_message: Optional[str] = None
    query_plan: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    data_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    @field_validator('query_plan')
    @classmethod
    def validate_query_plan(cls, v):
        """Validate and normalize query plan"""
        if v is None:
            return v
        
        # If it's a list, extract the first element or convert to dict
        if isinstance(v, list):
            if len(v) > 0:
                return v[0]  # Take first element
            else:
                return None
        
        # If it's already a dict, return as is
        if isinstance(v, dict):
            return v
        
        # Convert other types to string representation
        return {"plan": str(v)}
    
class QuerySession(BaseModel):
    """Track query session with full context"""
    session_id: str
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    natural_language_query: str
    sql_query: Optional[SQLQuery] = None
    execution_result: Optional[QueryExecutionResult] = None
    feedback_score: Optional[float] = Field(None, ge=1.0, le=5.0)
    human_feedback: Optional[str] = None
    system_metrics: Dict[str, Any] = Field(default_factory=dict)

class ValidationRule(ABC):
    """Abstract base for validation rules"""
    
    @abstractmethod
    def validate(self, query: SQLQuery, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        pass

class TableExistenceRule(ValidationRule):
    """Validate that referenced tables exist"""
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
    
    def validate(self, query: SQLQuery, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        available_tables = set(self.schema_info.keys())
        
        for table in query.tables_used:
            if table.lower() not in [t.lower() for t in available_tables]:
                return False, f"Table '{table}' does not exist"
        
        return True, None

class ColumnExistenceRule(ValidationRule):
    """Validate that referenced columns exist"""
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
    
    def validate(self, query: SQLQuery, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        for table_name, table_info in self.schema_info.items():
            table_columns = [col['name'].lower() for col in table_info['columns']]
            
            for column in query.columns_used:
                # Simple validation - could be more sophisticated
                if '.' in column:  # table.column format
                    table_part, column_part = column.split('.', 1)
                    if table_part.lower() == table_name.lower():
                        if column_part.lower() not in table_columns:
                            return False, f"Column '{column_part}' does not exist in table '{table_name}'"
        
        return True, None

class ComplexityLimitRule(ValidationRule):
    """Limit query complexity"""
    
    def __init__(self, max_complexity: QueryComplexity = QueryComplexity.COMPLEX):
        self.max_complexity = max_complexity
    
    def validate(self, query: SQLQuery, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        complexity_levels = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MEDIUM: 2,
            QueryComplexity.COMPLEX: 3
        }
        
        if complexity_levels[query.complexity] > complexity_levels[self.max_complexity]:
            return False, f"Query complexity {query.complexity} exceeds limit {self.max_complexity}"
        
        return True, None

class StructuredSQLGenerator:
    """Generate structured SQL with validation"""
    
    def __init__(self, openai_client: OpenAI, schema_info: Dict[str, Any]):
        self.client = openai_client
        self.schema_info = schema_info
        self.validation_rules: List[ValidationRule] = []
        
        # Add default validation rules
        self.add_validation_rule(TableExistenceRule(schema_info))
        self.add_validation_rule(ColumnExistenceRule(schema_info))
        self.add_validation_rule(ComplexityLimitRule())
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.validation_rules.append(rule)
    
    def generate_sql(self, natural_query: str, user_id: Optional[str] = None) -> SQLQuery:
        """Generate structured SQL using function calling"""
        
        schema_description = self._format_schema()
        
        # Get available user IDs for context
        user_context = ""
        if user_id:
            user_context = f"User Context: Filter results for user_id = '{user_id}'"
        else:
            user_context = "User Context: No specific user filter - show aggregate or sample data"
        
        function_schema = {
            "name": "generate_sql_query",
            "description": "Generate a SQL query from natural language",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL SELECT query"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": [t.value for t in QueryType],
                        "description": "Type of the SQL query"
                    },
                    "complexity": {
                        "type": "string",
                        "enum": [c.value for c in QueryComplexity],
                        "description": "Complexity level of the query"
                    },
                    "tables_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of table names used in the query"
                    },
                    "columns_used": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of column names used in the query"
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score for query correctness (0-1)"
                    }
                },
                "required": ["query", "query_type", "complexity", "tables_used", "confidence_score"]
            }
        }
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert SQL generator. Generate PostgreSQL queries from natural language.

Database Schema:
{schema_description}

{user_context}

IMPORTANT Rules:
1. Only generate SELECT statements
2. Use proper PostgreSQL syntax
3. If user_id is provided, add WHERE clause: WHERE user_id = '{user_id}' (user_id is a VARCHAR)
4. If no user_id provided, generate queries that work for all users or use sample data
5. For date ranges like "last month", use appropriate date functions
6. Be conservative with complexity scoring
7. List ALL tables and columns referenced (use table.column format when ambiguous)
8. Provide realistic confidence scores

Guidelines for complexity:
- SIMPLE: Basic SELECT with simple WHERE clauses, no JOINs
- MEDIUM: Aggregations, GROUP BY, simple JOINs
- COMPLEX: Multiple JOINs, subqueries, window functions

Available user IDs in database: user_001, user_002, user_003

Example date handling:
- "last month": WHERE payment_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND payment_date < DATE_TRUNC('month', CURRENT_DATE)
- "this year": WHERE EXTRACT(YEAR FROM payment_date) = EXTRACT(YEAR FROM CURRENT_DATE)
"""
            },
            {"role": "user", "content": natural_query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=[function_schema],
                function_call={"name": "generate_sql_query"},
                temperature=0.1
            )
            
            function_call = response.choices[0].message.function_call
            arguments = json.loads(function_call.arguments)
            
            # Create SQLQuery object
            sql_query = SQLQuery(**arguments)
            
            # Run validation
            self._validate_query(sql_query)
            
            return sql_query
            
        except Exception as e:
            # Fallback to simple query generation if function calling fails
            logging.warning(f"Function calling failed: {e}. Attempting fallback generation.")
            return self._fallback_generation(natural_query, user_id)
    
    def _fallback_generation(self, natural_query: str, user_id: Optional[str] = None) -> SQLQuery:
        """Fallback SQL generation without function calling"""
        
        # Simple prompt-based generation
        user_filter = f"WHERE user_id = '{user_id}'" if user_id else ""
        
        # Basic query patterns
        if "spend" in natural_query.lower() or "spent" in natural_query.lower():
            if "last month" in natural_query.lower():
                query = f"""
                SELECT SUM(amount) as total_spent 
                FROM payments 
                WHERE payment_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') 
                AND payment_date < DATE_TRUNC('month', CURRENT_DATE)
                {f"AND user_id = '{user_id}'" if user_id else ""}
                """
            else:
                query = f"""
                SELECT SUM(amount) as total_spent 
                FROM payments 
                {f"WHERE user_id = '{user_id}'" if user_id else ""}
                """
        else:
            # Default to recent payments
            query = f"""
            SELECT * 
            FROM payments 
            {f"WHERE user_id = '{user_id}'" if user_id else ""}
            ORDER BY payment_date DESC 
            LIMIT 10
            """
        
        return SQLQuery(
            query=query.strip(),
            query_type=QueryType.AGGREGATE if "SUM" in query else QueryType.FILTER,
            complexity=QueryComplexity.SIMPLE,
            tables_used=["payments"],
            columns_used=["amount", "payment_date", "user_id"] if user_id else ["amount", "payment_date"],
            confidence_score=0.7
        )
    
    def _format_schema(self) -> str:
        """Format schema for prompt"""
        schema_str = ""
        for table_name, table_info in self.schema_info.items():
            columns = ", ".join([f"{col['name']} ({col['type']})" for col in table_info['columns']])
            schema_str += f"\nTable: {table_name}\nColumns: {columns}\n"
        return schema_str
    
    def _validate_query(self, query: SQLQuery):
        """Run all validation rules"""
        for rule in self.validation_rules:
            is_valid, error_message = rule.validate(query, {})
            if not is_valid:
                raise ValueError(f"Validation failed: {error_message}")

class QueryExecutor:
    """Execute queries with monitoring and guardrails"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.execution_timeout = 30  # seconds
        self.max_result_rows = 10000
    
    def execute_query(self, sql_query: SQLQuery) -> QueryExecutionResult:
        """Execute query with monitoring"""
        start_time = time.time()
        
        try:
            with self.engine.connect() as connection:
                # Set query timeout
                connection.execute(text(f"SET statement_timeout = {self.execution_timeout * 1000}"))
                
                # Get query plan first (optional)
                query_plan = None
                try:
                    plan_result = connection.execute(text(f"EXPLAIN (FORMAT JSON) {sql_query.query}"))
                    plan_data = plan_result.fetchone()[0]
                    
                    # Handle the plan data properly - it comes as a list
                    if isinstance(plan_data, list) and len(plan_data) > 0:
                        query_plan = plan_data[0]  # Extract first element
                    elif isinstance(plan_data, dict):
                        query_plan = plan_data
                    else:
                        query_plan = {"raw_plan": str(plan_data)}
                        
                except Exception as e:
                    logging.warning(f"Could not get query plan: {e}")
                    query_plan = None
                
                # Execute actual query
                result = pd.read_sql(text(sql_query.query), connection)
                
                # Check result size
                if len(result) > self.max_result_rows:
                    logging.warning(f"Result set truncated from {len(result)} to {self.max_result_rows} rows")
                    result = result.head(self.max_result_rows)
                
                execution_time = (time.time() - start_time) * 1000
                
                return QueryExecutionResult(
                    success=True,
                    data=result.to_dict('records') if not result.empty else [],
                    row_count=len(result),
                    execution_time_ms=execution_time,
                    query_plan=query_plan,
                    data_quality_score=self._assess_data_quality(result)
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Assess data quality of results"""
        if df.empty:
            return 1.0
        
        # Simple data quality metrics
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        null_ratio = null_cells / total_cells if total_cells > 0 else 0
        
        # Quality score based on completeness
        quality_score = max(0.0, 1.0 - (null_ratio * 2))  # Penalize nulls
        
        return min(1.0, quality_score)

class SessionManager:
    """Manage query sessions with persistence and analytics"""
    
    def __init__(self, storage_path: str = "query_sessions.json"):
        self.storage_path = storage_path
        self.sessions: Dict[str, QuerySession] = {}
        self.load_sessions()
    
    def create_session(self, natural_query: str, user_id: Optional[str] = None) -> str:
        """Create new query session"""
        session_id = self._generate_session_id(natural_query, user_id)
        
        session = QuerySession(
            session_id=session_id,
            user_id=user_id,
            natural_language_query=natural_query
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def update_session(self, session_id: str, **kwargs):
        """Update session with new data"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            self.save_sessions()
    
    def add_feedback(self, session_id: str, score: float, feedback: str = None):
        """Add human feedback to session"""
        if session_id in self.sessions:
            self.sessions[session_id].feedback_score = score
            self.sessions[session_id].human_feedback = feedback
            self.save_sessions()
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get session analytics"""
        if not self.sessions:
            return {}
        
        successful_queries = [s for s in self.sessions.values() 
                            if s.execution_result and s.execution_result.success]
        
        analytics = {
            "total_sessions": len(self.sessions),
            "successful_queries": len(successful_queries),
            "success_rate": len(successful_queries) / len(self.sessions),
            "average_execution_time": sum(s.execution_result.execution_time_ms 
                                        for s in successful_queries) / len(successful_queries) if successful_queries else 0,
            "feedback_sessions": len([s for s in self.sessions.values() if s.feedback_score is not None]),
            "average_feedback_score": sum(s.feedback_score for s in self.sessions.values() 
                                        if s.feedback_score is not None) / max(1, len([s for s in self.sessions.values() if s.feedback_score is not None])),
            "query_types": self._analyze_query_types(),
            "complexity_distribution": self._analyze_complexity(),
            "common_errors": self._analyze_errors()
        }
        
        return analytics
    
    def _generate_session_id(self, query: str, user_id: Optional[str]) -> str:
        """Generate unique session ID"""
        content = f"{query}_{user_id}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _analyze_query_types(self) -> Dict[str, int]:
        """Analyze distribution of query types"""
        types = {}
        for session in self.sessions.values():
            if session.sql_query:
                query_type = session.sql_query.query_type.value
                types[query_type] = types.get(query_type, 0) + 1
        return types
    
    def _analyze_complexity(self) -> Dict[str, int]:
        """Analyze complexity distribution"""
        complexity = {}
        for session in self.sessions.values():
            if session.sql_query:
                comp = session.sql_query.complexity.value
                complexity[comp] = complexity.get(comp, 0) + 1
        return complexity
    
    def _analyze_errors(self) -> List[Dict[str, Any]]:
        """Analyze common errors"""
        errors = []
        for session in self.sessions.values():
            if session.execution_result and not session.execution_result.success:
                errors.append({
                    "query": session.natural_language_query,
                    "error": session.execution_result.error_message,
                    "timestamp": session.timestamp.isoformat()
                })
        return errors[:10]  # Return top 10 errors
    
    def save_sessions(self):
        """Save sessions to file"""
        try:
            sessions_data = {}
            for session_id, session in self.sessions.items():
                sessions_data[session_id] = session.model_dump()
            
            with open(self.storage_path, 'w') as f:
                json.dump(sessions_data, f, default=str, indent=2)
        except Exception as e:
            logging.error(f"Failed to save sessions: {e}")
    
    def load_sessions(self):
        """Load sessions from file"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    sessions_data = json.load(f)
                
                for session_id, session_data in sessions_data.items():
                    # Convert datetime strings back to datetime objects
                    if 'timestamp' in session_data:
                        session_data['timestamp'] = datetime.fromisoformat(session_data['timestamp'])
                    
                    self.sessions[session_id] = QuerySession(**session_data)
        except Exception as e:
            logging.error(f"Failed to load sessions: {e}")
            self.sessions = {}

class StructuredNLToSQLApp:
    """Main application with structured output and monitoring"""
    
    def __init__(self, database_url: str, openai_api_key: str):
        self.database_url = database_url
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize components
        self.schema_info = self._get_schema_info()
        self.sql_generator = StructuredSQLGenerator(self.openai_client, self.schema_info)
        self.query_executor = QueryExecutor(database_url)
        self.session_manager = SessionManager()
        
        # Configure guardrails
        self.setup_guardrails()
    
    def setup_guardrails(self):
        """Setup additional guardrails"""
        # Add custom validation rules
        self.sql_generator.add_validation_rule(ComplexityLimitRule(QueryComplexity.MEDIUM))
        
        # Configure executor limits
        self.query_executor.execution_timeout = 15  # Shorter timeout
        self.query_executor.max_result_rows = 5000   # Smaller result sets
    
    def process_query(self, natural_query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query with full monitoring"""
        # Create session
        session_id = self.session_manager.create_session(natural_query, user_id)
        
        try:
            # Generate SQL
            sql_query = self.sql_generator.generate_sql(natural_query, user_id)
            self.session_manager.update_session(session_id, sql_query=sql_query)
            
            # Execute query
            execution_result = self.query_executor.execute_query(sql_query)
            self.session_manager.update_session(session_id, execution_result=execution_result)
            
            # Return structured response with backward compatibility
            return {
                "session_id": session_id,
                "natural_query": natural_query,
                "user_id": user_id,
                "sql_query": sql_query.query,  # Return just the string, not the object
                "success": execution_result.success,
                "error_message": execution_result.error_message if not execution_result.success else None,
                "data": execution_result.data,
                "row_count": execution_result.row_count,
                "execution_time_ms": execution_result.execution_time_ms,
                "visualization_path": None  # Add if you have visualization
            }
            
        except Exception as e:
            error_result = QueryExecutionResult(success=False, error_message=str(e))
            self.session_manager.update_session(session_id, execution_result=error_result)
            
            return {
                "session_id": session_id,
                "natural_query": natural_query,
                "user_id": user_id,
                "success": False,
                "error": str(e),
                "error_message": str(e),
                "data": None,
                "row_count": 0,
                "execution_time_ms": 0,
                "sql_query": "",
                "visualization_path": None
            }
    
    def add_feedback(self, session_id: str, score: float, feedback: str = None):
        """Add feedback for a query session"""
        self.session_manager.add_feedback(session_id, score, feedback)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get application analytics"""
        return self.session_manager.get_analytics()
    
    def _get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        engine = create_engine(self.database_url)
        inspector = inspect(engine)
        
        schema_info = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema_info[table_name] = {
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable']
                    }
                    for col in columns
                ]
            }
        
        return schema_info

# Example usage and testing
if __name__ == "__main__":
    # Initialize app
    load_dotenv()
    
    # Test with better error handling
    try:
        app = StructuredNLToSQLApp(
            database_url="postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Process a query with a specific user
        print("Testing with user_id...")
        result = app.process_query("How much did I spend last month?", user_id="user_001")
        print("Query Result:", json.dumps(result, indent=2, default=str))
        
        # Test without user_id
        print("\nTesting without user_id...")
        result2 = app.process_query("What are the top 5 merchants by total revenue?")
        print("Query Result 2:", json.dumps(result2, indent=2, default=str))
        
        # Add feedback
        if result.get("session_id"):
            app.add_feedback(result["session_id"], 4.5, "Query worked well!")
        
        # Get analytics
        analytics = app.get_analytics()
        print("\nAnalytics:", json.dumps(analytics, indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()