"""
Enhanced Multi-Table NL to SQL Agent with LangGraph
This implementation supports dynamic table selection based on query keywords and external input payloads.
"""

import os
from dotenv import load_dotenv
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import openai
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_URL = "postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db"

class InputPayload(BaseModel):
    """External input payload structure"""
    model_config = ConfigDict(extra='allow')
    
    # Common financial identifiers
    CIN: Optional[int] = Field(None, description="Customer Identification Number")
    sort_code: Optional[int] = Field(None, description="Bank sort code")
    account_number: Optional[int] = Field(None, description="Account number")
    user_id: Optional[str] = Field(None, description="User identifier")
    
    # Additional fields allowed via extra='allow'
    
    def to_filter_conditions(self) -> Dict[str, Any]:
        """Convert payload to SQL filter conditions"""
        conditions = {}
        
        # Map payload fields to database columns
        field_mapping = {
            'CIN': 'customer_id',
            'sort_code': 'sort_code', 
            'account_number': 'account_number',
            'user_id': 'user_id'
        }
        
        for payload_field, db_field in field_mapping.items():
            value = getattr(self, payload_field, None)
            if value is not None:
                conditions[db_field] = value
        
        # Handle additional fields
        for field, value in self.__dict__.items():
            if field not in field_mapping and value is not None:
                conditions[field] = value
                
        return conditions

class TableMapping:
    """Manages table mappings and keyword detection"""
    
    def __init__(self):
        # Define table keywords mapping
        self.table_keywords = {
            'payments': {
                'keywords': ['payment', 'pay', 'transaction', 'spend', 'spent', 'expense', 'cost', 'money', 'amount', 'purchase', 'buy', 'bought'],
                'description': 'Payment transactions and spending data',
                'primary_key': 'payment_id',
                'user_column': 'user_id',
                'payload_columns': ['customer_id', 'account_number', 'sort_code', 'user_id']
            },
            'categories': {
                'keywords': ['category', 'categories', 'type', 'kind', 'group', 'classification', 'segment'],
                'description': 'Spending categories and classifications',
                'primary_key': 'category_id',
                'user_column': None,
                'payload_columns': []
            },
            'merchants': {
                'keywords': ['merchant', 'store', 'shop', 'vendor', 'retailer', 'business', 'company'],
                'description': 'Merchant and store information',
                'primary_key': 'merchant_id',
                'user_column': None,
                'payload_columns': []
            },
            'users': {
                'keywords': ['user', 'customer', 'account', 'profile', 'member'],
                'description': 'User account information',
                'primary_key': 'user_id',
                'user_column': 'user_id',
                'payload_columns': ['customer_id', 'user_id']
            },
            'budgets': {
                'keywords': ['budget', 'limit', 'allowance', 'target', 'goal'],
                'description': 'Budget limits and targets',
                'primary_key': 'budget_id',
                'user_column': 'user_id',
                'payload_columns': ['user_id']
            },
            'accounts': {
                'keywords': ['account', 'balance', 'statement', 'bank'],
                'description': 'Bank account information',
                'primary_key': 'account_id',
                'user_column': 'user_id',
                'payload_columns': ['account_number', 'sort_code', 'customer_id', 'user_id']
            }
        }
        
        # Common join patterns
        self.join_patterns = {
            ('payments', 'categories'): 'payments.category_id = categories.category_id',
            ('payments', 'merchants'): 'payments.merchant_id = merchants.merchant_id',
            ('payments', 'users'): 'payments.user_id = users.user_id',
            ('payments', 'accounts'): 'payments.account_number = accounts.account_number',
            ('budgets', 'categories'): 'budgets.category_id = categories.category_id',
            ('budgets', 'users'): 'budgets.user_id = users.user_id',
            ('accounts', 'users'): 'accounts.user_id = users.user_id'
        }
    
    def detect_relevant_tables(self, query: str, payload: Optional[InputPayload] = None) -> List[str]:
        """Detect which tables are relevant based on query keywords and payload"""
        query_lower = query.lower()
        relevant_tables = set()
        
        # Detect tables based on keywords
        for table_name, table_info in self.table_keywords.items():
            for keyword in table_info['keywords']:
                if keyword in query_lower:
                    relevant_tables.add(table_name)
                    break
        
        # Add tables based on payload fields
        if payload:
            payload_conditions = payload.to_filter_conditions()
            for table_name, table_info in self.table_keywords.items():
                payload_columns = table_info.get('payload_columns', [])
                if any(col in payload_conditions for col in payload_columns):
                    relevant_tables.add(table_name)
        
        # Default to payments if no specific table detected
        if not relevant_tables:
            relevant_tables.add('payments')
        
        # Add related tables based on context
        if 'payments' in relevant_tables:
            if any(word in query_lower for word in ['category', 'type', 'kind']):
                relevant_tables.add('categories')
            if any(word in query_lower for word in ['merchant', 'store', 'shop']):
                relevant_tables.add('merchants')
            if any(word in query_lower for word in ['account', 'balance']):
                relevant_tables.add('accounts')
        
        return list(relevant_tables)
    
    def get_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """Get JOIN condition between two tables"""
        key = (table1, table2) if (table1, table2) in self.join_patterns else (table2, table1)
        return self.join_patterns.get(key)
    
    def get_primary_table(self, tables: List[str]) -> str:
        """Determine the primary table for the query"""
        # Priority order for primary tables
        priority = ['payments', 'accounts', 'budgets', 'users', 'categories', 'merchants']
        
        for table in priority:
            if table in tables:
                return table
        
        return tables[0] if tables else 'payments'

class AgentState(BaseModel):
    """State object for the LangGraph agent with enhanced table support"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    user_query: str = ""
    input_payload: Optional[InputPayload] = None
    user_id: Optional[str] = None
    relevant_tables: List[str] = []
    primary_table: str = ""
    sql_query: str = ""
    validation_result: Dict[str, Any] = {}
    execution_result: Optional[pd.DataFrame] = None
    formatted_data: Optional[Dict[str, Any]] = None
    error_message: str = ""
    retry_count: int = 0
    schema_info: Dict[str, Any] = {}
    
    @field_validator('user_id', mode='before')
    @classmethod
    def convert_user_id_to_string(cls, v):
        """Convert user_id to string if it's not None"""
        if v is not None:
            return str(v)
        return v
    
    def __getitem__(self, key):
        """Support dict-like access for LangGraph compatibility"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Support dict-like assignment for LangGraph compatibility"""
        setattr(self, key, value)
    
    def keys(self):
        """Support dict-like keys() for LangGraph compatibility"""
        return self.__dict__.keys()
    
    def items(self):
        """Support dict-like items() for LangGraph compatibility"""
        return self.__dict__.items()
    
    def get(self, key, default=None):
        """Support dict-like get() for LangGraph compatibility"""
        return getattr(self, key, default)

class QueryResultStorage:
    """Handles storing query results in PostgreSQL database"""
    
    def __init__(self, database_manager):
        self.db_manager = database_manager
        self.table_name = "query_results"
        self._ensure_results_table_exists()
    
    def _ensure_results_table_exists(self):
        """Create query_results table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS query_results (
            result_id SERIAL PRIMARY KEY,
            user_id VARCHAR(50),
            user_query TEXT NOT NULL,
            input_payload JSONB,
            sql_query TEXT,
            relevant_tables TEXT[],
            primary_table VARCHAR(100),
            success BOOLEAN NOT NULL,
            error_message TEXT,
            result_data JSONB,
            formatted_data JSONB,
            row_count INTEGER DEFAULT 0,
            execution_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(100),
            query_hash VARCHAR(64)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_query_results_user_id ON query_results(user_id);
        CREATE INDEX IF NOT EXISTS idx_query_results_created_at ON query_results(created_at);
        CREATE INDEX IF NOT EXISTS idx_query_results_success ON query_results(success);
        CREATE INDEX IF NOT EXISTS idx_query_results_query_hash ON query_results(query_hash);
        CREATE INDEX IF NOT EXISTS idx_query_results_session_id ON query_results(session_id);
        """
        
        try:
            if not self.db_manager.engine:
                self.db_manager.connect()
            
            with self.db_manager.engine.connect() as connection:
                connection.execute(text(create_table_sql))
                connection.commit()
                logger.info("Query results table ensured to exist")
        except Exception as e:
            logger.warning(f"Could not create query_results table: {e}")
    
    def store_query_result(self, result: Dict[str, Any], execution_time_ms: int = 0, 
                          session_id: Optional[str] = None) -> Optional[int]:
        """Store query result in database and return the result_id"""
        try:
            if not self.db_manager.engine:
                self.db_manager.connect()
            
            # Generate query hash for deduplication/caching
            import hashlib
            query_content = f"{result.get('query', '')}{result.get('user_id', '')}{result.get('input_payload', '')}"
            query_hash = hashlib.sha256(query_content.encode()).hexdigest()
            
            # Prepare data for storage
            insert_data = {
                'user_id': result.get('user_id'),
                'user_query': result.get('query', ''),
                'input_payload': json.dumps(result.get('input_payload')) if result.get('input_payload') else None,
                'sql_query': result.get('sql_query', ''),
                'relevant_tables': result.get('relevant_tables', []),
                'primary_table': result.get('primary_table', ''),
                'success': result.get('success', False),
                'error_message': result.get('error_message', '') if result.get('error_message') else None,
                'result_data': json.dumps(result.get('formatted_data', {}).get('data', [])) if result.get('formatted_data') else None,
                'formatted_data': json.dumps(result.get('formatted_data')) if result.get('formatted_data') else None,
                'row_count': result.get('row_count', 0),
                'execution_time_ms': execution_time_ms,
                'session_id': session_id,
                'query_hash': query_hash
            }
            
            insert_sql = """
            INSERT INTO query_results (
                user_id, user_query, input_payload, sql_query, relevant_tables, 
                primary_table, success, error_message, result_data, formatted_data,
                row_count, execution_time_ms, session_id, query_hash
            ) VALUES (
                :user_id, :user_query, :input_payload, :sql_query, :relevant_tables,
                :primary_table, :success, :error_message, :result_data, :formatted_data,
                :row_count, :execution_time_ms, :session_id, :query_hash
            ) RETURNING result_id
            """
            
            with self.db_manager.engine.connect() as connection:
                result_proxy = connection.execute(text(insert_sql), insert_data)
                result_id = result_proxy.fetchone()[0]
                connection.commit()
                
                logger.info(f"Query result stored with ID: {result_id}")
                return result_id
                
        except Exception as e:
            logger.error(f"Failed to store query result: {e}")
            return None
    
    def get_query_results(self, user_id: Optional[str] = None, limit: int = 100, 
                         successful_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve stored query results"""
        try:
            if not self.db_manager.engine:
                self.db_manager.connect()
            
            where_conditions = []
            params = {'limit': limit}
            
            if user_id:
                where_conditions.append("user_id = :user_id")
                params['user_id'] = user_id
            
            if successful_only:
                where_conditions.append("success = true")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            select_sql = f"""
            SELECT 
                result_id, user_id, user_query, input_payload, sql_query,
                relevant_tables, primary_table, success, error_message,
                result_data, formatted_data, row_count, execution_time_ms,
                created_at, session_id, query_hash
            FROM query_results 
            {where_clause}
            ORDER BY created_at DESC 
            LIMIT :limit
            """
            
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(text(select_sql), params)
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    row_dict = dict(row._mapping)
                    
                    # Parse JSON fields
                    if row_dict['input_payload']:
                        try:
                            row_dict['input_payload'] = json.loads(row_dict['input_payload'])
                        except:
                            pass
                    
                    if row_dict['result_data']:
                        try:
                            row_dict['result_data'] = json.loads(row_dict['result_data'])
                        except:
                            pass
                    
                    if row_dict['formatted_data']:
                        try:
                            row_dict['formatted_data'] = json.loads(row_dict['formatted_data'])
                        except:
                            pass
                    
                    results.append(row_dict)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve query results: {e}")
            return []
    
    def get_query_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about stored queries"""
        try:
            if not self.db_manager.engine:
                self.db_manager.connect()
            
            where_clause = "WHERE user_id = :user_id" if user_id else ""
            params = {'user_id': user_id} if user_id else {}
            
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_queries,
                COUNT(*) FILTER (WHERE success = true) as successful_queries,
                COUNT(*) FILTER (WHERE success = false) as failed_queries,
                AVG(execution_time_ms) as avg_execution_time_ms,
                AVG(row_count) FILTER (WHERE success = true) as avg_rows_returned,
                COUNT(DISTINCT user_id) as unique_users,
                MIN(created_at) as first_query,
                MAX(created_at) as last_query
            FROM query_results
            {where_clause}
            """
            
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(text(stats_sql), params)
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to get query statistics: {e}")
            return {}

class DatabaseManager:
    """Enhanced database manager with multi-table support"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.table_mapping = TableMapping()
        
    def connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.database_url)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def get_schema_info(self, relevant_tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get database schema information for relevant tables"""
        if not self.engine:
            self.connect()
            
        inspector = inspect(self.engine)
        schema_info = {}
        
        # Get all tables if none specified
        if not relevant_tables:
            table_names = inspector.get_table_names()
        else:
            # Filter to existing tables
            all_tables = inspector.get_table_names()
            table_names = [t for t in relevant_tables if t in all_tables]
        
        for table_name in table_names:
            try:
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                indexes = inspector.get_indexes(table_name)
                
                # Process columns with safe attribute access
                processed_columns = []
                for col in columns:
                    try:
                        processed_col = {
                            'name': col.get('name', 'unknown'),
                            'type': str(col.get('type', 'UNKNOWN')),
                            'nullable': col.get('nullable', True),
                            'primary_key': col.get('primary_key', False)
                        }
                        processed_columns.append(processed_col)
                    except Exception as col_error:
                        logger.warning(f"Error processing column in {table_name}: {col_error}")
                        # Add a basic column entry
                        processed_columns.append({
                            'name': str(col.get('name', 'unknown')),
                            'type': 'UNKNOWN',
                            'nullable': True,
                            'primary_key': False
                        })
                
                schema_info[table_name] = {
                    'columns': processed_columns,
                    'foreign_keys': foreign_keys or [],
                    'indexes': indexes or [],
                    'table_info': self.table_mapping.table_keywords.get(table_name, {})
                }
            except Exception as e:
                logger.warning(f"Could not get schema for table {table_name}: {e}")
                # Add minimal schema info to prevent complete failure
                schema_info[table_name] = {
                    'columns': [{'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True}],
                    'foreign_keys': [],
                    'indexes': [],
                    'table_info': self.table_mapping.table_keywords.get(table_name, {})
                }
        
        return schema_info
    
    def get_user_list(self) -> List[str]:
        """Get list of available user IDs from the database"""
        if not self.engine:
            self.connect()
            
        try:
            with self.engine.connect() as connection:
                # Try payments table first, then users table
                for table in ['payments', 'users']:
                    try:
                        result = connection.execute(text(f"SELECT DISTINCT user_id FROM {table} WHERE user_id IS NOT NULL ORDER BY user_id LIMIT 100"))
                        users = [str(row[0]) for row in result.fetchall()]
                        if users:
                            return users
                    except SQLAlchemyError:
                        continue
                return []
        except SQLAlchemyError as e:
            logger.warning(f"Could not fetch user list: {e}")
            return []
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        if not self.engine:
            self.connect()
            
        try:
            with self.engine.connect() as connection:
                result = pd.read_sql(text(query), connection)
                logger.info(f"Query executed successfully, returned {len(result)} rows")
                return result
        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {e}")
            raise
    
    def validate_query(self, query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced query validation with table awareness"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic SQL injection protection
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {query_upper} ':
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Potentially dangerous SQL keyword detected: {keyword}")
        
        # Check for placeholder patterns
        placeholder_patterns = [
            r'/\*.*?\*/',  # Comments
            r'your_user_id',  # Placeholder text
            r'\$\{.*?\}',  # Template literals
            r'<.*?>',  # XML-like placeholders
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Placeholder pattern detected in query: {pattern}")
        
        return validation_result

class LLMService:
    """Enhanced LLM service with multi-table support and payload integration"""
    
    def __init__(self, api_key: str):
        # Simple OpenAI client initialization with error handling
        os.environ['OPENAI_API_KEY'] = api_key
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.use_legacy = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            logger.info("Trying alternative initialization...")
            try:
                import openai
                openai.api_key = api_key
                self.client = openai
                self.use_legacy = True
            except Exception as e2:
                raise Exception(f"Could not initialize OpenAI client: {e}, {e2}")
        
        self.table_mapping = TableMapping()
        
    def generate_sql(self, natural_query: str, schema_info: Dict[str, Any], 
                    relevant_tables: List[str], primary_table: str, 
                    user_id: Optional[str] = None, 
                    input_payload: Optional[InputPayload] = None) -> str:
        """Generate SQL query with multi-table awareness and payload integration"""
        
        # Build comprehensive schema description
        schema_description = self._build_schema_description(schema_info, relevant_tables)
        
        # Build user context
        user_context = self._build_user_context(user_id, relevant_tables)
        
        # Build payload context
        payload_context = self._build_payload_context(input_payload, relevant_tables)
        
        # Build table relationship context
        relationship_context = self._build_relationship_context(relevant_tables)
        
        system_prompt = f"""You are an expert SQL query generator for a financial database. Convert natural language questions to SQL queries.

Database Schema:
{schema_description}

{user_context}

{payload_context}

{relationship_context}

Query Context:
- Primary table: {primary_table}
- Relevant tables: {', '.join(relevant_tables)}
- Focus on tables that match the user's intent

CRITICAL Rules:
1. Only use SELECT statements
2. Use appropriate JOINs when querying multiple tables
3. Include proper WHERE clauses for filtering
4. Use aggregate functions when appropriate (SUM, AVG, COUNT, etc.)
5. Return ONLY the SQL query, no explanation or markdown
6. Use PostgreSQL syntax
7. NEVER use placeholder comments or template variables
8. Apply payload filters as WHERE conditions when provided
9. Generate complete, executable SQL with real values
10. Choose the most appropriate table(s) based on the query intent

Filtering Priority:
1. Apply payload filters first (highest priority)
2. Apply user_id filters for user-specific queries
3. Apply query-based filters

Date/Time Handling:
- Use CURRENT_DATE for "today"
- Use INTERVAL for relative dates (e.g., CURRENT_DATE - INTERVAL '1 month')
- Format dates as YYYY-MM-DD

Example Multi-table Queries with Filters:
- SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id WHERE p.customer_id = 22 GROUP BY c.category_name
- SELECT m.merchant_name, AVG(p.amount) FROM payments p JOIN merchants m ON p.merchant_id = m.merchant_id WHERE p.account_number = 900914 GROUP BY m.merchant_name
- SELECT u.username, SUM(p.amount) FROM payments p JOIN users u ON p.user_id = u.user_id WHERE p.sort_code = 123456 GROUP BY u.username
"""
        
        try:
            if self.use_legacy:
                # Legacy OpenAI (v0.x)
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": natural_query}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                sql_query = response.choices[0].message.content.strip()
            else:
                # Modern OpenAI (v1.x)
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": natural_query}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response
            sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r'\s*```$', '', sql_query)
            sql_query = sql_query.strip()
            
            # Final validation - ensure no placeholders remain
            if re.search(r'/\*.*?\*/', sql_query) or 'your_user_id' in sql_query.lower():
                raise ValueError("Generated SQL contains placeholders")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"LLM SQL generation error: {e}")
            raise
    
    def _build_schema_description(self, schema_info: Dict[str, Any], relevant_tables: List[str]) -> str:
        """Build detailed schema description for relevant tables"""
        schema_description = ""
        
        for table_name in relevant_tables:
            if table_name in schema_info:
                table_info = schema_info[table_name]
                
                # Table header with description
                description = table_info.get('table_info', {}).get('description', '')
                schema_description += f"\nTable: {table_name}"
                if description:
                    schema_description += f" - {description}"
                schema_description += "\n"
                
                # Columns
                columns = []
                for col in table_info['columns']:
                    col_desc = f"{col['name']} ({col['type']})"
                    if col.get('primary_key', False):
                        col_desc += " [PK]"
                    if not col.get('nullable', True):  # Use .get() with default
                        col_desc += " NOT NULL"
                    columns.append(col_desc)
                
                schema_description += f"Columns: {', '.join(columns)}\n"
                
                # Foreign keys
                if table_info.get('foreign_keys'):
                    fks = []
                    for fk in table_info['foreign_keys']:
                        fks.append(f"{fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}")
                    schema_description += f"Foreign Keys: {', '.join(fks)}\n"
        
        return schema_description
    
    def _build_user_context(self, user_id: Optional[str], relevant_tables: List[str]) -> str:
        """Build user context based on relevant tables"""
        if user_id:
            user_tables = [t for t in relevant_tables 
                          if self.table_mapping.table_keywords.get(t, {}).get('user_column')]
            
            if user_tables:
                return f"""
User Context:
- Current user ID: {user_id}
- Apply user filter to: {', '.join(user_tables)}
- Use WHERE user_id = '{user_id}' for user-specific data
- For system-wide queries, omit user filter
"""
        
        return """
User Context:
- No specific user context provided
- Query all data without user-specific filtering
"""
    
    def _build_payload_context(self, payload: Optional[InputPayload], relevant_tables: List[str]) -> str:
        """Build payload context for filtering"""
        if not payload:
            return """
Payload Context:
- No external payload provided
- No additional filtering required
"""
        
        conditions = payload.to_filter_conditions()
        if not conditions:
            return """
Payload Context:
- Empty payload provided
- No additional filtering required
"""
        
        context = """
Payload Context:
- External payload provided with filters
- MUST apply the following WHERE conditions:
"""
        
        for field, value in conditions.items():
            if isinstance(value, str):
                context += f"  - {field} = '{value}'\n"
            else:
                context += f"  - {field} = {value}\n"
        
        context += "- These filters have HIGHEST PRIORITY and must be included in the query\n"
        
        return context
    
    def _build_relationship_context(self, relevant_tables: List[str]) -> str:
        """Build context about table relationships"""
        if len(relevant_tables) <= 1:
            return ""
        
        relationships = []
        for i, table1 in enumerate(relevant_tables):
            for table2 in relevant_tables[i+1:]:
                join_condition = self.table_mapping.get_join_condition(table1, table2)
                if join_condition:
                    relationships.append(f"{table1} ↔ {table2}: {join_condition}")
        
        if relationships:
            return f"""
Table Relationships:
{chr(10).join(f"- {rel}" for rel in relationships)}
"""
        return ""

class DataFormatter:
    """Formats query results for external visualization systems"""
    
    @staticmethod
    def format_for_visualization(data: pd.DataFrame, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format DataFrame for external visualization applications"""
        
        if data.empty:
            return {
                'status': 'success',
                'data': [],
                'metadata': {
                    'row_count': 0,
                    'column_count': 0,
                    'columns': [],
                    'query_info': query_info
                }
            }
        
        # Convert DataFrame to visualization-friendly format
        formatted_data = {
            'status': 'success',
            'data': data.to_dict('records'),
            'metadata': {
                'row_count': len(data),
                'column_count': len(data.columns),
                'columns': [
                    {
                        'name': col,
                        'type': str(data[col].dtype),
                        'sample_values': data[col].dropna().head(3).tolist()
                    }
                    for col in data.columns
                ],
                'query_info': query_info,
                'suggested_charts': DataFormatter._suggest_chart_types(data)
            }
        }
        
        return formatted_data
    
    @staticmethod
    def _suggest_chart_types(data: pd.DataFrame) -> List[str]:
        """Suggest appropriate chart types based on data characteristics"""
        suggestions = []
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        date_cols = data.select_dtypes(include=['datetime']).columns
        
        # Chart type suggestions based on data structure
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            if len(data) <= 20:
                suggestions.extend(['bar_chart', 'horizontal_bar_chart'])
            else:
                suggestions.extend(['pie_chart', 'donut_chart'])
        
        if len(numeric_cols) >= 2:
            suggestions.extend(['scatter_plot', 'bubble_chart'])
        
        if len(date_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.extend(['line_chart', 'area_chart'])
        
        if len(numeric_cols) == 1:
            suggestions.extend(['histogram', 'box_plot'])
        
        # Default suggestions
        if not suggestions:
            suggestions = ['table', 'bar_chart']
        
        return suggestions

class NLToSQLAgent:
    """Enhanced agent with LangGraph workflow"""
    
    def __init__(self, database_url: str, openai_api_key: str):
        self.db_manager = DatabaseManager(database_url)
        self.llm_service = LLMService(openai_api_key)
        self.table_mapping = TableMapping()
        self.data_formatter = DataFormatter()
        self.result_storage = QueryResultStorage(self.db_manager)
        self.graph = None
        self.setup_graph()
        
    def setup_graph(self):
        """Setup the LangGraph workflow using latest API"""
        from langgraph.graph import StateGraph, START, END
        
        # Create workflow with state type
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("get_schema", self.get_schema_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("validate_sql", self.validate_sql_node)
        workflow.add_node("execute_sql", self.execute_sql_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Add edges using latest API
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "get_schema")
        workflow.add_edge("get_schema", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_edge("execute_sql", "format_results")
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edge with proper routing
        def validation_router(state: AgentState) -> str:
            """Route after SQL validation"""
            if state.error_message:
                return "handle_error"
            
            if not state.validation_result.get('is_valid', False):
                if state.retry_count < 2:
                    state.retry_count += 1
                    return "generate_sql"
                else:
                    state.error_message = f"Max retries reached. Validation errors: {state.validation_result.get('errors', [])}"
                    return "handle_error"
            
            return "execute_sql"
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validate_sql",
            validation_router,
            {
                "generate_sql": "generate_sql",
                "execute_sql": "execute_sql",
                "handle_error": "handle_error"
            }
        )
        
        # Compile the graph
        self.graph = workflow.compile()
        logger.info("LangGraph workflow compiled successfully with latest API")
    
    def analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze query to determine relevant tables"""
        try:
            relevant_tables = self.table_mapping.detect_relevant_tables(
                state.user_query, 
                state.input_payload
            )
            primary_table = self.table_mapping.get_primary_table(relevant_tables)
            
            state.relevant_tables = relevant_tables
            state.primary_table = primary_table
            
            logger.info(f"Query analysis - Primary: {primary_table}, Relevant: {relevant_tables}")
        except Exception as e:
            state.error_message = f"Query analysis error: {str(e)}"
            logger.error(state.error_message)
        
        return state
    
    def get_schema_node(self, state: AgentState) -> AgentState:
        """Get schema information for relevant tables"""
        try:
            schema_info = self.db_manager.get_schema_info(state.relevant_tables)
            state.schema_info = schema_info
            logger.info(f"Schema retrieved for tables: {list(schema_info.keys())}")
        except Exception as e:
            state.error_message = f"Schema retrieval error: {str(e)}"
            logger.error(state.error_message)
        
        return state
    
    def generate_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL with multi-table awareness and payload integration"""
        try:
            sql_query = self.llm_service.generate_sql(
                state.user_query,
                state.schema_info,
                state.relevant_tables,
                state.primary_table,
                state.user_id,
                state.input_payload
            )
            state.sql_query = sql_query
            logger.info(f"Generated SQL: {sql_query}")
        except Exception as e:
            state.error_message = f"SQL generation error: {str(e)}"
            logger.error(state.error_message)
        
        return state
    
    def validate_sql_node(self, state: AgentState) -> AgentState:
        """Validate the generated SQL"""
        try:
            validation_result = self.db_manager.validate_query(state.sql_query, state.schema_info)
            state.validation_result = validation_result
            logger.info(f"SQL validation result: {validation_result}")
        except Exception as e:
            state.error_message = f"SQL validation error: {str(e)}"
            logger.error(state.error_message)
        
        return state
    
    def execute_sql_node(self, state: AgentState) -> AgentState:
        """Execute the validated SQL query"""
        try:
            result_df = self.db_manager.execute_query(state.sql_query)
            state.execution_result = result_df
            logger.info(f"SQL executed successfully, {len(result_df)} rows returned")
        except Exception as e:
            state.error_message = f"SQL execution error: {str(e)}"
            logger.error(state.error_message)
        
        return state
    
    def format_results_node(self, state: AgentState) -> AgentState:
        """Format results for external visualization"""
        try:
            if state.execution_result is not None:
                query_info = {
                    'query': state.user_query,
                    'sql_query': state.sql_query,
                    'relevant_tables': state.relevant_tables,
                    'primary_table': state.primary_table,
                    'user_id': state.user_id,
                    'input_payload': state.input_payload.dict() if state.input_payload else None
                }
                
                formatted_data = self.data_formatter.format_for_visualization(
                    state.execution_result, 
                    query_info
                )
                state.formatted_data = formatted_data
                logger.info(f"Results formatted for visualization: {len(state.execution_result)} rows")
            else:
                state.formatted_data = {
                    'status': 'success',
                    'data': [],
                    'metadata': {
                        'row_count': 0,
                        'column_count': 0,
                        'columns': [],
                        'query_info': {
                            'query': state.user_query,
                            'sql_query': state.sql_query
                        }
                    }
                }
                logger.info("No data to format")
        except Exception as e:
            state.error_message = f"Data formatting error: {str(e)}"
            logger.error(state.error_message)
        
        return state
    
    def handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        logger.error(f"Workflow error: {state.error_message}")
        return state
    
    async def process_query(self, user_query: str, user_id: Optional[str] = None, 
                          input_payload: Optional[Union[Dict[str, Any], InputPayload]] = None,
                          store_result: bool = True, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a natural language query through the LangGraph agent"""
        
        import time
        start_time = time.time()
        
        # Convert input_payload to InputPayload if it's a dict
        if isinstance(input_payload, dict):
            input_payload = InputPayload(**input_payload)
        
        initial_state = AgentState(
            user_query=user_query, 
            user_id=user_id,
            input_payload=input_payload
        )
        
        try:
            # Run the LangGraph workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Debug: Print the type and structure of the result
            logger.debug(f"LangGraph result type: {type(result)}")
            logger.debug(f"LangGraph result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
            
            # Safely extract values from the result
            def safe_get(obj, key, default=None):
                """Safely get value from object, whether it's dict-like or has attributes"""
                try:
                    if hasattr(obj, key):
                        return getattr(obj, key, default)
                    elif hasattr(obj, '__getitem__'):
                        return obj.get(key, default) if hasattr(obj, 'get') else obj[key]
                    else:
                        return default
                except (KeyError, AttributeError, TypeError):
                    return default
            
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Format response using safe extraction
            response = {
                'query': user_query,
                'user_id': user_id,
                'input_payload': input_payload.dict() if input_payload else None,
                'relevant_tables': safe_get(result, 'relevant_tables', []),
                'primary_table': safe_get(result, 'primary_table', ''),
                'sql_query': safe_get(result, 'sql_query', ''),
                'success': not bool(safe_get(result, 'error_message', '')),
                'error_message': safe_get(result, 'error_message', ''),
                'formatted_data': safe_get(result, 'formatted_data', None),
                'row_count': 0,
                'execution_time_ms': execution_time_ms
            }
            
            # Calculate row count safely
            execution_result = safe_get(result, 'execution_result', None)
            if execution_result is not None and hasattr(execution_result, '__len__'):
                response['row_count'] = len(execution_result)
            
            # Store result in database if requested
            if store_result:
                try:
                    result_id = self.result_storage.store_query_result(
                        response, 
                        execution_time_ms, 
                        session_id
                    )
                    response['result_id'] = result_id
                    logger.info(f"Query result stored with ID: {result_id}")
                except Exception as storage_error:
                    logger.warning(f"Failed to store query result: {storage_error}")
                    response['storage_warning'] = str(storage_error)
            
            return response
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Error during query processing: {e}")
            
            error_response = {
                'query': user_query,
                'user_id': user_id,
                'input_payload': input_payload.dict() if input_payload else None,
                'relevant_tables': [],
                'primary_table': '',
                'sql_query': '',
                'success': False,
                'error_message': f"Processing error: {str(e)}",
                'formatted_data': None,
                'row_count': 0,
                'execution_time_ms': execution_time_ms
            }
            
            # Store error result if requested
            if store_result:
                try:
                    result_id = self.result_storage.store_query_result(
                        error_response, 
                        execution_time_ms, 
                        session_id
                    )
                    error_response['result_id'] = result_id
                except Exception as storage_error:
                    logger.warning(f"Failed to store error result: {storage_error}")
            
            return error_response

# Enhanced application interface with payload support
class NLToSQLApp:
    """Enhanced application interface with multi-table support and payload integration"""
    
    def __init__(self):
        self.agent = None
        self.current_user_id = None
        self.available_users = []
        self.table_mapping = TableMapping()
        self.session_id = None
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment and initialize agent"""
        # Generate session ID for this session
        import uuid
        self.session_id = str(uuid.uuid4())
        
        # Get OpenAI API key from environment
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize agent
        self.agent = NLToSQLAgent(DATABASE_URL, openai_api_key)
        
        # Connect to database
        self.agent.db_manager.connect()
        
        # Get available users
        self.available_users = self.agent.db_manager.get_user_list()
        
        print("✅ Enhanced Multi-Table NL to SQL Agent with LangGraph initialized!")
        print("📊 Database connected")
        print("🤖 LLM service ready")
        print("🔄 LangGraph workflow compiled")
        print("💾 Query result storage enabled")
        print(f"👥 Found {len(self.available_users)} users in database")
        print(f"🏛️  Supported tables: {', '.join(self.table_mapping.table_keywords.keys())}")
        print(f"📝 Session ID: {self.session_id}")
        print("\n" + "="*60)
    
    def select_user(self):
        """Allow user to select a user context"""
        if not self.available_users:
            print("⚠️  No users found in database. Queries will run without user context.")
            return
        
        print("\n👥 Available Users:")
        print("0. Query all data (no user filter)")
        for i, user_id in enumerate(self.available_users, 1):
            print(f"{i}. {user_id}")
        
        while True:
            try:
                choice = input(f"\nSelect user (0-{len(self.available_users)}): ").strip()
                
                if choice == '0':
                    self.current_user_id = None
                    print("🌍 Set to query all data")
                    break
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_users):
                        self.current_user_id = str(self.available_users[idx])
                        print(f"👤 Set user context to: {self.current_user_id}")
                        break
                    else:
                        print("❌ Invalid selection")
                else:
                    # Allow direct user ID input
                    self.current_user_id = str(choice)
                    print(f"👤 Set user context to: {self.current_user_id}")
                    break
                        
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input")
    
    def parse_payload_input(self, user_input: str) -> Optional[InputPayload]:
        """Parse payload from user input with improved JSON handling"""
        try:
            # Look for JSON-like input (both single and double quotes)
            if '{' in user_input and '}' in user_input:
                json_start = user_input.find('{')
                json_end = user_input.rfind('}') + 1
                json_str = user_input[json_start:json_end]
                
                # Try to parse as-is first (in case it's already valid JSON)
                try:
                    payload_dict = json.loads(json_str)
                    return InputPayload(**payload_dict)
                except json.JSONDecodeError:
                    # If that fails, try to fix common issues
                    # Replace single quotes with double quotes for JSON compatibility
                    fixed_json = json_str.replace("'", '"')
                    
                    try:
                        payload_dict = json.loads(fixed_json)
                        return InputPayload(**payload_dict)
                    except json.JSONDecodeError:
                        # Try using ast.literal_eval for Python-style dictionaries
                        import ast
                        try:
                            payload_dict = ast.literal_eval(json_str)
                            return InputPayload(**payload_dict)
                        except (ValueError, SyntaxError):
                            # Last resort: try to manually parse simple cases
                            return self._manual_parse_payload(json_str)
                            
        except Exception as e:
            logger.debug(f"Could not parse payload: {e}")
        
        return None
    
    def _manual_parse_payload(self, json_str: str) -> Optional[InputPayload]:
        """Manually parse simple payload patterns"""
        try:
            # Remove outer braces and split by comma
            content = json_str.strip('{}').strip()
            pairs = content.split(',')
            
            payload_dict = {}
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    # Clean up key and value
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    
                    # Try to convert value to appropriate type
                    try:
                        # Try integer first
                        if value.isdigit():
                            payload_dict[key] = int(value)
                        # Try float
                        elif '.' in value and value.replace('.', '').isdigit():
                            payload_dict[key] = float(value)
                        # Keep as string
                        else:
                            payload_dict[key] = value
                    except:
                        payload_dict[key] = value
            
            if payload_dict:
                return InputPayload(**payload_dict)
                
        except Exception as e:
            logger.debug(f"Manual parsing failed: {e}")
        
        return None
    
    def show_table_info(self):
        """Display information about available tables"""
        print("\n🏛️  Available Tables and Keywords:")
        print("="*60)
        
        for table_name, table_info in self.table_mapping.table_keywords.items():
            print(f"\n📋 {table_name.upper()}")
            print(f"   Description: {table_info['description']}")
            print(f"   Keywords: {', '.join(table_info['keywords'][:8])}...")
            if table_info.get('user_column'):
                print(f"   User-specific: Yes ({table_info['user_column']})")
            else:
                print(f"   User-specific: No")
            
            payload_cols = table_info.get('payload_columns', [])
            if payload_cols:
                print(f"   Payload columns: {', '.join(payload_cols)}")
        
        print("\n💡 Tips:")
        print("   - Use keywords in your questions to target specific tables")
        print("   - The agent will automatically join related tables when needed")
        print("   - Questions with 'I', 'my', 'me' will use your selected user context")
        print("   - Include JSON payload like: {'CIN': 22, 'account_number': 900914}")
        print("="*60)
    
    def show_query_history(self, limit: int = 10):
        """Show recent query history for current user"""
        try:
            results = self.agent.result_storage.get_query_results(
                user_id=self.current_user_id,
                limit=limit,
                successful_only=False
            )
            
            if not results:
                print("📝 No query history found.")
                return
            
            print(f"\n📊 Recent Query History (last {len(results)} queries):")
            print("="*80)
            
            for i, result in enumerate(results, 1):
                status = "✅" if result['success'] else "❌"
                created_at = result['created_at'].strftime("%Y-%m-%d %H:%M:%S")
                execution_time = result['execution_time_ms']
                
                print(f"\n{i}. {status} [{created_at}] ({execution_time}ms)")
                print(f"   Query: {result['user_query'][:80]}...")
                print(f"   Tables: {', '.join(result['relevant_tables']) if result['relevant_tables'] else 'None'}")
                print(f"   Rows: {result['row_count']}")
                
                if not result['success']:
                    print(f"   Error: {result['error_message'][:60]}...")
            
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error retrieving query history: {e}")
    
    def show_query_statistics(self):
        """Show query statistics for current user"""
        try:
            stats = self.agent.result_storage.get_query_statistics(self.current_user_id)
            
            if not stats or stats.get('total_queries', 0) == 0:
                print("📊 No query statistics available.")
                return
            
            print("\n📊 Query Statistics:")
            print("="*50)
            print(f"Total queries: {stats.get('total_queries', 0)}")
            print(f"Successful: {stats.get('successful_queries', 0)}")
            print(f"Failed: {stats.get('failed_queries', 0)}")
            
            if stats.get('total_queries', 0) > 0:
                success_rate = (stats.get('successful_queries', 0) / stats.get('total_queries', 1)) * 100
                print(f"Success rate: {success_rate:.1f}%")
            
            if stats.get('avg_execution_time_ms'):
                print(f"Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")
            
            if stats.get('avg_rows_returned'):
                print(f"Avg rows returned: {stats['avg_rows_returned']:.1f}")
            
            if stats.get('first_query'):
                print(f"First query: {stats['first_query']}")
            
            if stats.get('last_query'):
                print(f"Last query: {stats['last_query']}")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error retrieving statistics: {e}")
    
    def replay_query(self, result_id: int):
        """Replay a stored query by ID"""
        try:
            results = self.agent.result_storage.get_query_results(limit=1000)
            stored_result = None
            
            for result in results:
                if result['result_id'] == result_id:
                    stored_result = result
                    break
            
            if not stored_result:
                print(f"❌ Query with ID {result_id} not found.")
                return
            
            print(f"\n🔄 Replaying query {result_id}:")
            print(f"Original query: {stored_result['user_query']}")
            print(f"Original user: {stored_result['user_id']}")
            print(f"Original payload: {stored_result['input_payload']}")
            
            # Convert stored payload back to InputPayload if exists
            payload = None
            if stored_result['input_payload']:
                payload = InputPayload(**stored_result['input_payload'])
            
            # Execute the query again
            print("🤔 Re-executing query...")
            result = asyncio.run(self.agent.process_query(
                stored_result['user_query'],
                stored_result['user_id'],
                payload,
                store_result=True,
                session_id=self.session_id
            ))
            
            # Display new results
            self.display_results(result)
            
        except Exception as e:
            print(f"❌ Error replaying query: {e}")
    
    def run_interactive(self):
        """Run enhanced interactive command-line interface"""
        print("🚀 Enhanced Multi-Table NL to SQL Agent with LangGraph")
        print("Ask questions about your financial data across multiple tables!")
        print("Type 'quit' to exit, 'help' for examples, 'tables' for table info, 'user' to change user")
        print("Include JSON payload for filtering: {'CIN': 22, 'sort_code': 123456}")
        print("Additional commands: 'history' (query history), 'stats' (statistics), 'replay <id>' (replay query)\n")
        
        # Initial user selection
        if self.available_users:
            self.select_user()
        
        while True:
            try:
                # Show current context
                context_info = f"👤 {self.current_user_id}" if self.current_user_id else "🌍 All Users"
                user_input = input(f"\n{context_info} | 💬 Your question: ").strip()
                
                if user_input.lower() == 'history':
                    self.show_query_history()
                    continue
                
                if user_input.lower() == 'stats':
                    self.show_query_statistics()
                    continue
                
                if user_input.lower().startswith('replay '):
                    try:
                        result_id = int(user_input.split()[1])
                        self.replay_query(result_id)
                    except (IndexError, ValueError):
                        print("❌ Usage: replay <result_id>")
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 Analyzing query and determining relevant tables...")
                
                # Parse payload from input
                payload = self.parse_payload_input(user_input)
                if payload:
                    print(f"📦 Detected payload: {payload.dict()}")
                
                # Process the query with current user context and payload
                result = asyncio.run(self.agent.process_query(
                    user_input, 
                    self.current_user_id,
                    payload,
                    store_result=True,
                    session_id=self.session_id
                ))
                
                # Display results
                self.display_results(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def show_examples(self):
        """Show enhanced example queries with payload examples"""
        examples = {
            "💳 Payment Queries": [
                "How much did I spend last month?",
                "Show me all payments over $100",
                "What's my average transaction amount?",
                "Show payments from last week"
            ],
            "🏷️  Category Queries": [
                "Show me spending by category",
                "Which category do I spend the most on?",
                "Compare food vs entertainment spending",
                "Show my top 5 spending categories"
            ],
            "🏪 Merchant Queries": [
                "Which stores do I shop at most?",
                "Show me all Netflix payments",
                "Compare spending at different merchants",
                "Which merchant had my highest single transaction?"
            ],
            "👥 User Queries": [
                "Show top spending users",
                "Compare my spending to average user",
                "List all users and their total spending",
                "Who are the highest spending customers?"
            ],
            "📊 Budget Queries": [
                "Show my budget vs actual spending",
                "Which categories am I over budget on?",
                "How much budget do I have remaining?",
                "Compare budget to actual by month"
            ],
            "🔄 Multi-Table Queries": [
                "Show my spending by category and merchant",
                "Which users spend most on food category?",
                "Compare budget vs actual spending by category",
                "Show merchant performance across all users"
            ],
            "📦 Payload Examples (use either format)": [
                'Show payments {"CIN": 22, "sort_code": 123456}',
                "Show payments {'CIN': 22, 'sort_code': 123456}",
                'Account balance {"account_number": 900914}',
                "Transaction history {'CIN': 22, 'account_number': 900914}",
                'User spending {"user_id": "user_001"}'
            ]
        }
        
        print("\n📝 Example Questions by Table Type:")
        print("="*60)
        
        for category, questions in examples.items():
            print(f"\n{category}:")
            for i, question in enumerate(questions, 1):
                print(f"   {i}. {question}")
        
        print(f"\n💡 The agent automatically detects which tables to use based on your keywords!")
        print("📦 Payload formats supported:")
        print('   - JSON style: {"CIN": 22, "sort_code": 123456}')
        print("   - Python style: {'CIN': 22, 'sort_code': 123456}")
        print("\n🔧 Additional Commands:")
        print("   - 'history' - Show recent query history")
        print("   - 'stats' - Show query statistics")
        print("   - 'replay <id>' - Replay a stored query by ID")
        print("   - 'user' - Change user context")
        print("   - 'tables' - Show table information")
        print("="*60)
    
    def display_results(self, result: Dict[str, Any]):
        """Display enhanced query results with formatted data"""
        print("\n" + "="*60)
        
        if not result['success']:
            print(f"❌ Error: {result['error_message']}")
            if result.get('result_id'):
                print(f"📝 Error logged with ID: {result['result_id']}")
            return
        
        print(f"🔍 Query: {result['query']}")
        if result['user_id']:
            print(f"👤 User Context: {result['user_id']}")
        else:
            print(f"🌍 User Context: All users")
        
        if result['input_payload']:
            print(f"📦 Input Payload: {result['input_payload']}")
        
        print(f"🏛️  Primary Table: {result['primary_table']}")
        print(f"🔗 Relevant Tables: {', '.join(result['relevant_tables'])}")
        print(f"📝 Generated SQL: {result['sql_query']}")
        print(f"📊 Rows returned: {result['row_count']}")
        print(f"⏱️  Execution time: {result['execution_time_ms']}ms")
        
        if result.get('result_id'):
            print(f"💾 Result stored with ID: {result['result_id']}")
        
        if result['formatted_data'] and result['formatted_data']['data']:
            print("\n📋 Formatted Results:")
            formatted_data = result['formatted_data']
            
            # Display metadata
            metadata = formatted_data['metadata']
            print(f"   Columns: {metadata['column_count']}")
            print(f"   Suggested charts: {', '.join(metadata['suggested_charts'])}")
            
            # Display sample data
            data = formatted_data['data']
            if len(data) <= 10:
                df = pd.DataFrame(data)
                print(f"\n{df.to_string(index=False)}")
            else:
                df = pd.DataFrame(data[:10])
                print(f"\n{df.to_string(index=False)}")
                print(f"\n... and {len(data) - 10} more rows")
            
            print(f"\n📊 Data ready for external visualization system")
        elif result['row_count'] == 0:
            print("\n📋 No data returned for this query")
        
        print("\n" + "="*60)

# Enhanced API interface for external applications
class NLToSQLAPI:
    """RESTful API interface for external applications"""
    
    def __init__(self, database_url: str, openai_api_key: str):
        self.agent = NLToSQLAgent(database_url, openai_api_key)
        self.agent.db_manager.connect()
    
    async def query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query request from external API"""
        try:
            # Extract request parameters
            user_query = request_data.get('query', '')
            user_id = request_data.get('user_id')
            payload_data = request_data.get('payload', {})
            session_id = request_data.get('session_id')
            store_result = request_data.get('store_result', True)
            
            # Validate required fields
            if not user_query:
                return {
                    'success': False,
                    'error_message': 'Query parameter is required',
                    'error_type': 'validation_error'
                }
            
            # Convert payload to InputPayload if provided
            input_payload = None
            if payload_data:
                try:
                    input_payload = InputPayload(**payload_data)
                except Exception as e:
                    return {
                        'success': False,
                        'error_message': f'Invalid payload format: {str(e)}',
                        'error_type': 'payload_error'
                    }
            
            # Process the query
            result = await self.agent.process_query(
                user_query=user_query,
                user_id=user_id,
                input_payload=input_payload,
                store_result=store_result,
                session_id=session_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"API query processing error: {e}")
            return {
                'success': False,
                'error_message': f'Internal processing error: {str(e)}',
                'error_type': 'processing_error'
            }
    
    def get_query_history(self, user_id: Optional[str] = None, limit: int = 100, 
                         successful_only: bool = False) -> Dict[str, Any]:
        """Get query history via API"""
        try:
            results = self.agent.result_storage.get_query_results(
                user_id=user_id,
                limit=limit,
                successful_only=successful_only
            )
            
            return {
                'success': True,
                'data': results,
                'count': len(results)
            }
            
        except Exception as e:
            logger.error(f"API history retrieval error: {e}")
            return {
                'success': False,
                'error_message': f'Failed to retrieve query history: {str(e)}',
                'error_type': 'retrieval_error'
            }
    
    def get_query_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get query statistics via API"""
        try:
            stats = self.agent.result_storage.get_query_statistics(user_id)
            
            return {
                'success': True,
                'data': stats
            }
            
        except Exception as e:
            logger.error(f"API statistics retrieval error: {e}")
            return {
                'success': False,
                'error_message': f'Failed to retrieve statistics: {str(e)}',
                'error_type': 'retrieval_error'
            }
    
    def get_table_schema(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get table schema information via API"""
        try:
            schema_info = self.agent.db_manager.get_schema_info(tables)
            
            return {
                'success': True,
                'data': schema_info
            }
            
        except Exception as e:
            logger.error(f"API schema retrieval error: {e}")
            return {
                'success': False,
                'error_message': f'Failed to retrieve schema: {str(e)}',
                'error_type': 'schema_error'
            }

# Utility functions for external integration
def create_agent(database_url: str = None, openai_api_key: str = None) -> NLToSQLAgent:
    """Factory function to create an agent instance"""
    if not database_url:
        database_url = DATABASE_URL
    
    if not openai_api_key:
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
    
    return NLToSQLAgent(database_url, openai_api_key)

def create_api(database_url: str = None, openai_api_key: str = None) -> NLToSQLAPI:
    """Factory function to create an API instance"""
    if not database_url:
        database_url = DATABASE_URL
    
    if not openai_api_key:
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
    
    return NLToSQLAPI(database_url, openai_api_key)

# Example usage for external applications
async def example_api_usage():
    """Example of how to use the API programmatically"""
    
    # Create API instance
    api = create_api()
    
    # Example query request
    request_data = {
        'query': 'Show me spending by category for the last month',
        'user_id': 'user_001',
        'payload': {
            'CIN': 22,
            'sort_code': 123456
        },
        'session_id': 'example_session_123',
        'store_result': True
    }
    
    # Process query
    result = await api.query(request_data)
    
    if result['success']:
        print("Query successful!")
        print(f"SQL: {result['sql_query']}")
        print(f"Rows: {result['row_count']}")
        print(f"Data: {result['formatted_data']}")
    else:
        print(f"Query failed: {result['error_message']}")
    
    # Get query history
    history = api.get_query_history(user_id='user_001', limit=10)
    print(f"Query history: {len(history['data'])} queries")
    
    # Get statistics
    stats = api.get_query_statistics(user_id='user_001')
    print(f"User statistics: {stats['data']}")

# Health check and setup utilities
def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import sqlalchemy
    except ImportError:
        missing_deps.append("sqlalchemy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import openai
    except ImportError:
        missing_deps.append("openai")
    
    try:
        import pydantic
    except ImportError:
        missing_deps.append("pydantic")
    
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        missing_deps.append("langgraph")
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_deps.append("python-dotenv")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def setup_database_tables():
    """Create sample database tables for testing (optional)"""
    create_tables_sql = """
    -- Create users table
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(50) PRIMARY KEY,
        username VARCHAR(100),
        email VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create categories table
    CREATE TABLE IF NOT EXISTS categories (
        category_id SERIAL PRIMARY KEY,
        category_name VARCHAR(100) NOT NULL,
        description TEXT
    );

    -- Create merchants table
    CREATE TABLE IF NOT EXISTS merchants (
        merchant_id SERIAL PRIMARY KEY,
        merchant_name VARCHAR(200) NOT NULL,
        merchant_category VARCHAR(100)
    );

    -- Create accounts table
    CREATE TABLE IF NOT EXISTS accounts (
        account_id SERIAL PRIMARY KEY,
        account_number BIGINT UNIQUE,
        sort_code INTEGER,
        user_id VARCHAR(50) REFERENCES users(user_id),
        account_type VARCHAR(50),
        balance DECIMAL(12,2) DEFAULT 0.00,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create payments table
    CREATE TABLE IF NOT EXISTS payments (
        payment_id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        account_number BIGINT,
        sort_code INTEGER,
        customer_id INTEGER,
        merchant_id INTEGER REFERENCES merchants(merchant_id),
        category_id INTEGER REFERENCES categories(category_id),
        amount DECIMAL(10,2) NOT NULL,
        transaction_date DATE,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create budgets table
    CREATE TABLE IF NOT EXISTS budgets (
        budget_id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        category_id INTEGER REFERENCES categories(category_id),
        budget_amount DECIMAL(10,2) NOT NULL,
        period_start DATE,
        period_end DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Insert sample data
    INSERT INTO categories (category_name, description) VALUES
    ('Food & Dining', 'Restaurants, groceries, food delivery'),
    ('Entertainment', 'Movies, games, streaming services'),
    ('Transportation', 'Gas, public transport, ride sharing'),
    ('Shopping', 'Retail purchases, online shopping'),
    ('Bills & Utilities', 'Electricity, water, internet, phone')
    ON CONFLICT DO NOTHING;

    INSERT INTO merchants (merchant_name, merchant_category) VALUES
    ('Netflix', 'Entertainment'),
    ('Uber', 'Transportation'),
    ('Amazon', 'Shopping'),
    ('Starbucks', 'Food & Dining'),
    ('Shell', 'Transportation')
    ON CONFLICT DO NOTHING;
    """
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            connection.execute(text(create_tables_sql))
            connection.commit()
        print("✅ Database tables created/verified successfully")
        return True
    except Exception as e:
        print(f"⚠️  Could not create database tables: {e}")
        print("   (Tables may already exist or you may need to create them manually)")
        return False

def health_check():
    """Perform a comprehensive health check"""
    print("🔍 Performing system health check...\n")
    
    # Check dependencies
    print("1. Checking dependencies...")
    if not check_dependencies():
        return False
    print("   ✅ All dependencies available\n")
    
    # Check environment variables
    print("2. Checking environment variables...")
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("   ❌ OPENAI_API_KEY not found in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    print("   ✅ OPENAI_API_KEY found\n")
    
    # Check database connection
    print("3. Checking database connection...")
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("   ✅ Database connection successful\n")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("   Check your DATABASE_URL and ensure PostgreSQL is running")
        return False
    
    # Optional: Setup database tables
    print("4. Setting up database tables...")
    setup_database_tables()
    print()
    
    print("🎉 Health check completed successfully!")
    return True

# Enhanced command line interface with options
def main():
    """Main entry point for command line usage"""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['health', 'check', 'healthcheck']:
            if health_check():
                print("\n✅ System ready! You can now run the application.")
                return
            else:
                print("\n❌ Health check failed. Please fix the issues above.")
                sys.exit(1)
        
        elif command in ['setup', 'init', 'initialize']:
            print("🔧 Setting up Enhanced NL to SQL Agent...")
            if health_check():
                print("\n🎉 Setup completed successfully!")
                print("You can now run: python script.py")
                return
            else:
                sys.exit(1)
        
        elif command in ['api', 'server']:
            print("🚀 Starting API mode...")
            print("Note: This is a basic example. For production, use FastAPI or Flask.")
            # Here you could start a web server
            try:
                asyncio.run(example_api_usage())
            except Exception as e:
                print(f"API example error: {e}")
            return
        
        elif command in ['help', '-h', '--help']:
            print("Enhanced Multi-Table NL to SQL Agent")
            print("=====================================")
            print("Usage:")
            print("  python script.py                 - Start interactive mode")
            print("  python script.py health          - Run health check")
            print("  python script.py setup           - Initialize system")
            print("  python script.py api             - API usage example")
            print("  python script.py help            - Show this help")
            return
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'python script.py help' for available commands")
            return
    
    # Default: Start interactive application
    try:
        print("🚀 Starting Enhanced Multi-Table NL to SQL Agent...")
        print("Run 'python script.py health' first if you encounter issues.\n")
        
        app = NLToSQLApp()
        app.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Run health check: python script.py health")
        print("2. Set OPENAI_API_KEY environment variable")
        print("3. Ensure PostgreSQL is running")
        print("4. Install dependencies: pip install sqlalchemy pandas openai pydantic langgraph python-dotenv")
        print("5. Check DATABASE_URL configuration")

# Export main classes for external use
__all__ = [
    'NLToSQLAgent',
    'NLToSQLAPI', 
    'NLToSQLApp',
    'InputPayload',
    'TableMapping',
    'QueryResultStorage',
    'DatabaseManager',
    'LLMService',
    'DataFormatter',
    'create_agent',
    'create_api',
    'main'
]

if __name__ == "__main__":
    main()