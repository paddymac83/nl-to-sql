"""
Enhanced Multi-Table NL to SQL Agent
This implementation supports dynamic table selection based on query keywords.
"""

import os
from dotenv import load_dotenv
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import openai
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_URL = "postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db"

class TableMapping:
    """Manages table mappings and keyword detection"""
    
    def __init__(self):
        # Define table keywords mapping
        self.table_keywords = {
            'payments': {
                'keywords': ['payment', 'pay', 'transaction', 'spend', 'spent', 'expense', 'cost', 'money', 'amount', 'purchase', 'buy', 'bought'],
                'description': 'Payment transactions and spending data',
                'primary_key': 'payment_id',
                'user_column': 'user_id'
            },
            'categories': {
                'keywords': ['category', 'categories', 'type', 'kind', 'group', 'classification', 'segment'],
                'description': 'Spending categories and classifications',
                'primary_key': 'category_id',
                'user_column': None
            },
            'merchants': {
                'keywords': ['merchant', 'store', 'shop', 'vendor', 'retailer', 'business', 'company'],
                'description': 'Merchant and store information',
                'primary_key': 'merchant_id',
                'user_column': None
            },
            'users': {
                'keywords': ['user', 'customer', 'account', 'profile', 'member'],
                'description': 'User account information',
                'primary_key': 'user_id',
                'user_column': 'user_id'
            },
            'budgets': {
                'keywords': ['budget', 'limit', 'allowance', 'target', 'goal'],
                'description': 'Budget limits and targets',
                'primary_key': 'budget_id',
                'user_column': 'user_id'
            }
        }
        
        # Common join patterns
        self.join_patterns = {
            ('payments', 'categories'): 'payments.category_id = categories.category_id',
            ('payments', 'merchants'): 'payments.merchant_id = merchants.merchant_id',
            ('payments', 'users'): 'payments.user_id = users.user_id',
            ('budgets', 'categories'): 'budgets.category_id = categories.category_id',
            ('budgets', 'users'): 'budgets.user_id = users.user_id'
        }
    
    def detect_relevant_tables(self, query: str) -> List[str]:
        """Detect which tables are relevant based on query keywords"""
        query_lower = query.lower()
        relevant_tables = set()
        
        for table_name, table_info in self.table_keywords.items():
            for keyword in table_info['keywords']:
                if keyword in query_lower:
                    relevant_tables.add(table_name)
                    break
        
        # Default to payments if no specific table detected
        if not relevant_tables:
            relevant_tables.add('payments')
        
        # Add related tables based on context
        if 'payments' in relevant_tables:
            # Check if we need category or merchant info
            if any(word in query_lower for word in ['category', 'type', 'kind']):
                relevant_tables.add('categories')
            if any(word in query_lower for word in ['merchant', 'store', 'shop']):
                relevant_tables.add('merchants')
        
        return list(relevant_tables)
    
    def get_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """Get JOIN condition between two tables"""
        key = (table1, table2) if (table1, table2) in self.join_patterns else (table2, table1)
        return self.join_patterns.get(key)
    
    def get_primary_table(self, tables: List[str]) -> str:
        """Determine the primary table for the query"""
        # Priority order for primary tables
        priority = ['payments', 'budgets', 'users', 'categories', 'merchants']
        
        for table in priority:
            if table in tables:
                return table
        
        return tables[0] if tables else 'payments'

class AgentState(BaseModel):
    """State object for the LangGraph agent with enhanced table support"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_query: str = ""
    user_id: Optional[str] = None
    relevant_tables: List[str] = []
    primary_table: str = ""
    sql_query: str = ""
    validation_result: Dict[str, Any] = {}
    execution_result: Optional[pd.DataFrame] = None
    visualization_path: Optional[str] = None
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
            logger.error(state.error_message)
        
        return state
    
    def create_visualization_node(self, state: AgentState) -> AgentState:
        """Create visualization from query results"""
        try:
            if state.execution_result is not None and not state.execution_result.empty:
                viz_path = self.viz_service.create_visualization(
                    state.execution_result, 
                    state.user_query,
                    state.relevant_tables
                )
                state.visualization_path = viz_path
                logger.info(f"Visualization created: {viz_path}")
            else:
                logger.info("No data to visualize")
        except Exception as e:
            # Don't fail the entire process for visualization errors
            logger.warning(f"Visualization error: {str(e)}")
        
        return state
    
    def handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        logger.error(f"Workflow error: {state.error_message}")
        return state
    
    async def process_query(self, user_query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a natural language query through the enhanced agent"""
        
        initial_state = AgentState(user_query=user_query, user_id=user_id)
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Format response
        response = {
            'query': user_query,
            'user_id': user_id,
            'relevant_tables': final_state.relevant_tables,
            'primary_table': final_state.primary_table,
            'sql_query': final_state.sql_query,
            'success': not bool(final_state.error_message),
            'error_message': final_state.error_message,
            'data': final_state.execution_result.to_dict('records') if final_state.execution_result is not None else None,
            'visualization_path': final_state.visualization_path,
            'row_count': len(final_state.execution_result) if final_state.execution_result is not None else 0
        }
        
        return response

# Enhanced application interface with multi-table support
class NLToSQLApp:
    """Enhanced application interface with multi-table support"""
    
    def __init__(self):
        self.agent = None
        self.current_user_id = None
        self.available_users = []
        self.table_mapping = TableMapping()
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment and initialize agent"""
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
        
        print("✅ Enhanced Multi-Table NL to SQL Agent initialized!")
        print("📊 Database connected")
        print("🤖 LLM service ready")
        print(f"👥 Found {len(self.available_users)} users in database")
        print(f"🏛️  Supported tables: {', '.join(self.table_mapping.table_keywords.keys())}")
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
        
        print("\n💡 Tips:")
        print("   - Use keywords in your questions to target specific tables")
        print("   - The agent will automatically join related tables when needed")
        print("   - Questions with 'I', 'my', 'me' will use your selected user context")
        print("="*60)
    
    def run_interactive(self):
        """Run enhanced interactive command-line interface"""
        print("🚀 Enhanced Multi-Table NL to SQL Agent")
        print("Ask questions about your financial data across multiple tables!")
        print("Type 'quit' to exit, 'help' for examples, 'tables' for table info, 'user' to change user\n")
        
        # Initial user selection
        if self.available_users:
            self.select_user()
        
        while True:
            try:
                # Show current context
                context_info = f"👤 {self.current_user_id}" if self.current_user_id else "🌍 All Users"
                user_input = input(f"\n{context_info} | 💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    self.show_examples()
                    continue
                
                if user_input.lower() in ['tables', 'table', 'schema']:
                    self.show_table_info()
                    continue
                
                if user_input.lower() == 'user':
                    self.select_user()
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 Analyzing query and determining relevant tables...")
                
                # Process the query with current user context
                result = asyncio.run(self.agent.process_query(user_input, self.current_user_id))
                
                # Display results
                self.display_results(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def show_examples(self):
        """Show enhanced example queries for different tables"""
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
            ]
        }
        
        print("\n📝 Example Questions by Table Type:")
        print("="*60)
        
        for category, questions in examples.items():
            print(f"\n{category}:")
            for i, question in enumerate(questions, 1):
                print(f"   {i}. {question}")
        
        print(f"\n💡 The agent automatically detects which tables to use based on your keywords!")
        print("="*60)
    
    def display_results(self, result: Dict[str, Any]):
        """Display enhanced query results with table information"""
        print("\n" + "="*60)
        
        if not result['success']:
            print(f"❌ Error: {result['error_message']}")
            return
        
        print(f"🔍 Query: {result['query']}")
        if result['user_id']:
            print(f"👤 User Context: {result['user_id']}")
        else:
            print(f"🌍 User Context: All users")
        
        print(f"🏛️  Primary Table: {result['primary_table']}")
        print(f"🔗 Relevant Tables: {', '.join(result['relevant_tables'])}")
        print(f"📝 Generated SQL: {result['sql_query']}")
        print(f"📊 Rows returned: {result['row_count']}")
        
        if result['data']:
            print("\n📋 Results:")
            df = pd.DataFrame(result['data'])
            
            # Limit display for large results
            if len(df) > 10:
                print(df.head(10).to_string(index=False))
                print(f"\n... and {len(df) - 10} more rows")
            else:
                print(df.to_string(index=False))
        
        if result['visualization_path']:
            print(f"\n📈 Visualization saved: {result['visualization_path']}")
            print("   Open the file to view the chart!")
        
        print("\n" + "="*60)

# Command line interface
def main():
    """Main entry point for command line usage"""
    try:
        app = NLToSQLApp()
        app.run_interactive()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. PostgreSQL running with the test database")
        print("3. Installed all required dependencies")
        print("4. Database tables exist (payments, categories, merchants, users, budgets)")

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
                
                schema_info[table_name] = {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable'],
                            'primary_key': col.get('primary_key', False)
                        }
                        for col in columns
                    ],
                    'foreign_keys': foreign_keys,
                    'indexes': indexes,
                    'table_info': self.table_mapping.table_keywords.get(table_name, {})
                }
            except Exception as e:
                logger.warning(f"Could not get schema for table {table_name}: {e}")
        
        return schema_info
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> Optional[pd.DataFrame]:
        """Get sample data from a table for context"""
        if not self.engine:
            self.connect()
            
        try:
            with self.engine.connect() as connection:
                query = f"SELECT * FROM {table_name} LIMIT {limit}"
                result = pd.read_sql(text(query), connection)
                return result
        except SQLAlchemyError as e:
            logger.warning(f"Could not fetch sample data from {table_name}: {e}")
            return None
    
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
        
        # Check for table existence in query
        mentioned_tables = []
        for table_name in schema_info.keys():
            if table_name.lower() in query.lower():
                mentioned_tables.append(table_name)
        
        if not mentioned_tables:
            validation_result['warnings'].append("No recognized tables found in query")
        
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
    """Enhanced LLM service with multi-table support"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.table_mapping = TableMapping()
        
    def generate_sql(self, natural_query: str, schema_info: Dict[str, Any], 
                    relevant_tables: List[str], primary_table: str, 
                    user_id: Optional[str] = None) -> str:
        """Generate SQL query with multi-table awareness"""
        
        # Build comprehensive schema description
        schema_description = self._build_schema_description(schema_info, relevant_tables)
        
        # Build user context
        user_context = self._build_user_context(user_id, relevant_tables)
        
        # Build table relationship context
        relationship_context = self._build_relationship_context(relevant_tables)
        
        system_prompt = f"""You are an expert SQL query generator for a financial database. Convert natural language questions to SQL queries.

Database Schema:
{schema_description}

{user_context}

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
8. If user_id is provided, filter user-specific tables accordingly
9. Generate complete, executable SQL with real values
10. Choose the most appropriate table(s) based on the query intent

Table Selection Guidelines:
- For spending/payment queries: Use payments table
- For category analysis: Join payments with categories
- For merchant analysis: Join payments with merchants  
- For user information: Use users table
- For budget queries: Use budgets table
- Always join related tables when additional context is needed

Date/Time Handling:
- Use CURRENT_DATE for "today"
- Use INTERVAL for relative dates (e.g., CURRENT_DATE - INTERVAL '1 month')
- Format dates as YYYY-MM-DD

Example Multi-table Queries:
- SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id GROUP BY c.category_name
- SELECT m.merchant_name, AVG(p.amount) FROM payments p JOIN merchants m ON p.merchant_id = m.merchant_id GROUP BY m.merchant_name
- SELECT u.username, SUM(p.amount) FROM payments p JOIN users u ON p.user_id = u.user_id GROUP BY u.username
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
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
                    if col.get('primary_key'):
                        col_desc += " [PK]"
                    if not col['nullable']:
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

class VisualizationService:
    """Enhanced visualization service with multi-table support"""
    
    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')
        plt.style.use('seaborn-v0_8')
        
    def create_visualization(self, data: pd.DataFrame, query: str, 
                           relevant_tables: List[str]) -> Optional[str]:
        """Create visualization with table-aware styling"""
        
        if data.empty:
            return None
            
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Enhanced visualization based on table context
                chart_created = self._create_context_aware_chart(data, ax, relevant_tables, query)
                
                if not chart_created:
                    # Fallback to generic visualization
                    self._create_generic_chart(data, ax)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"visualization_{timestamp}.png"
                filepath = os.path.join("outputs", filename)
                
                os.makedirs("outputs", exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                return filepath
                
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            plt.close('all')
            return None
    
    def _create_context_aware_chart(self, data: pd.DataFrame, ax, 
                                   relevant_tables: List[str], query: str) -> bool:
        """Create chart based on table context"""
        
        # Category-based visualizations
        if 'categories' in relevant_tables:
            return self._create_category_chart(data, ax)
        
        # Merchant-based visualizations  
        elif 'merchants' in relevant_tables:
            return self._create_merchant_chart(data, ax)
        
        # User-based visualizations
        elif 'users' in relevant_tables and len(data) > 1:
            return self._create_user_chart(data, ax)
        
        # Budget visualizations
        elif 'budgets' in relevant_tables:
            return self._create_budget_chart(data, ax)
        
        return False
    
    def _create_category_chart(self, data: pd.DataFrame, ax) -> bool:
        """Create category-specific visualization"""
        if 'category_name' in data.columns and len(data) <= 20:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1:
                data.plot(x='category_name', y=numeric_cols[0], kind='bar', ax=ax, color='skyblue')
                ax.set_title(f"{numeric_cols[0]} by Category")
                ax.set_xlabel("Category")
                return True
        return False
    
    def _create_merchant_chart(self, data: pd.DataFrame, ax) -> bool:
        """Create merchant-specific visualization"""
        if 'merchant_name' in data.columns and len(data) <= 15:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1:
                # Sort by value for better visualization
                sorted_data = data.nlargest(10, numeric_cols[0])
                sorted_data.plot(x='merchant_name', y=numeric_cols[0], kind='barh', ax=ax, color='lightcoral')
                ax.set_title(f"Top Merchants by {numeric_cols[0]}")
                ax.set_xlabel(numeric_cols[0])
                return True
        return False
    
    def _create_user_chart(self, data: pd.DataFrame, ax) -> bool:
        """Create user-specific visualization"""
        if 'username' in data.columns or 'user_id' in data.columns:
            user_col = 'username' if 'username' in data.columns else 'user_id'
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1 and len(data) <= 20:
                data.plot(x=user_col, y=numeric_cols[0], kind='bar', ax=ax, color='lightgreen')
                ax.set_title(f"{numeric_cols[0]} by User")
                return True
        return False
    
    def _create_budget_chart(self, data: pd.DataFrame, ax) -> bool:
        """Create budget-specific visualization"""
        # Look for budget vs actual comparisons
        if 'budget_amount' in data.columns and 'actual_amount' in data.columns:
            categories = data.get('category_name', range(len(data)))
            x = range(len(data))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], data['budget_amount'], width, label='Budget', color='lightblue')
            ax.bar([i + width/2 for i in x], data['actual_amount'], width, label='Actual', color='lightcoral')
            
            ax.set_xlabel('Categories')
            ax.set_ylabel('Amount')
            ax.set_title('Budget vs Actual Spending')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45)
            ax.legend()
            return True
        return False
    
    def _create_generic_chart(self, data: pd.DataFrame, ax):
        """Fallback generic chart creation"""
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1 and len(data) <= 20:
            data.plot(x=categorical_cols[0], y=numeric_cols[0], kind='bar', ax=ax)
            ax.set_title(f"{numeric_cols[0]} by {categorical_cols[0]}")
        elif len(numeric_cols) >= 2:
            ax.scatter(data[numeric_cols[0]], data[numeric_cols[1]])
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"{numeric_cols[1]} vs {numeric_cols[0]}")
        elif len(numeric_cols) == 1:
            data[numeric_cols[0]].hist(ax=ax, bins=20)
            ax.set_title(f"Distribution of {numeric_cols[0]}")
        else:
            # Show as table
            ax.axis('tight')
            ax.axis('off')
            table_data = data.head(10)
            table = ax.table(cellText=table_data.values,
                           colLabels=table_data.columns,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            ax.set_title("Query Results")

class NLToSQLAgent:
    """Enhanced agent with multi-table support"""
    
    def __init__(self, database_url: str, openai_api_key: str):
        self.db_manager = DatabaseManager(database_url)
        self.llm_service = LLMService(openai_api_key)
        self.viz_service = VisualizationService()
        self.table_mapping = TableMapping()
        self.graph = None
        self.setup_graph()
        
    def setup_graph(self):
        """Setup the enhanced LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("get_schema", self.get_schema_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("validate_sql", self.validate_sql_node)
        workflow.add_node("execute_sql", self.execute_sql_node)
        workflow.add_node("create_visualization", self.create_visualization_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Add edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "get_schema")
        workflow.add_edge("get_schema", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        
        # Conditional edges for validation
        workflow.add_conditional_edges(
            "validate_sql",
            self.should_retry,
            {
                "execute": "execute_sql",
                "retry": "generate_sql",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("execute_sql", "create_visualization")
        workflow.add_edge("create_visualization", END)
        workflow.add_edge("handle_error", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze query to determine relevant tables"""
        try:
            relevant_tables = self.table_mapping.detect_relevant_tables(state.user_query)
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
        """Generate SQL with multi-table awareness"""
        try:
            sql_query = self.llm_service.generate_sql(
                state.user_query,
                state.schema_info,
                state.relevant_tables,
                state.primary_table,
                state.user_id
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
    
    def should_retry(self, state: AgentState) -> str:
        """Decide whether to retry, execute, or handle error"""
        if state.error_message:
            return "error"
        
        if not state.validation_result.get('is_valid', False):
            if state.retry_count < 2:
                state.retry_count += 1
                return "retry"
            else:
                state.error_message = f"Max retries reached. Validation errors: {state.validation_result.get('errors', [])}"
                return "error"
        
        return "execute"
    
    def execute_sql_node(self, state: AgentState) -> AgentState:
        """Execute the validated SQL query"""
        try:
            result_df = self.db_manager.execute_query(state.sql_query)
            state.execution_result = result_df
            logger.info(f"SQL executed successfully, {len(result_df)} rows returned")
        except Exception as e:
            state.error_message = f"SQL execution error: {str(e)}"
            logger.error(state.error_message)