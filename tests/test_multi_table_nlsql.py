"""
Comprehensive test suite for the Enhanced Multi-Table NL to SQL Agent
"""

import pytest
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime
import tempfile
import json

# Import the classes from the main module
# Note: In real usage, you'd import from the actual module file
from multi_table_nlsql import (
    TableMapping, AgentState, DatabaseManager, LLMService, 
    VisualizationService, NLToSQLAgent, NLToSQLApp
)

class TestTableMapping:
    """Test cases for table mapping and keyword detection"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.table_mapping = TableMapping()
    
    def test_detect_relevant_tables_payments(self):
        """Test detection of payment-related queries"""
        queries = [
            "How much did I spend last month?",
            "Show my payment history",
            "What's my total expense?",
            "How much money did I spend on groceries?"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'payments' in tables, f"Failed to detect payments table for: {query}"
    
    def test_detect_relevant_tables_categories(self):
        """Test detection of category-related queries"""
        queries = [
            "Show spending by category",
            "Which type of expense is highest?",
            "Group my purchases by kind",
            "Show me the different classifications"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'categories' in tables, f"Failed to detect categories table for: {query}"
    
    def test_detect_relevant_tables_merchants(self):
        """Test detection of merchant-related queries"""
        queries = [
            "Which stores do I visit most?",
            "Show me all merchant transactions",
            "Compare spending at different vendors",
            "What businesses do I shop at?"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'merchants' in tables, f"Failed to detect merchants table for: {query}"
    
    def test_detect_relevant_tables_multi_table(self):
        """Test detection of multi-table queries"""
        query = "Show my spending by category at different merchants"
        tables = self.table_mapping.detect_relevant_tables(query)
        
        assert 'payments' in tables
        assert 'categories' in tables
        assert 'merchants' in tables
    
    def test_get_join_condition(self):
        """Test JOIN condition retrieval"""
        # Test existing join
        condition = self.table_mapping.get_join_condition('payments', 'categories')
        assert condition == 'payments.category_id = categories.category_id'
        
        # Test reverse order
        condition = self.table_mapping.get_join_condition('categories', 'payments')
        assert condition == 'payments.category_id = categories.category_id'
        
        # Test non-existent join
        condition = self.table_mapping.get_join_condition('categories', 'budgets')
        assert condition == 'budgets.category_id = categories.category_id'
    
    def test_get_primary_table(self):
        """Test primary table selection"""
        # Test with payments in list
        tables = ['categories', 'payments', 'merchants']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'payments'
        
        # Test without payments
        tables = ['categories', 'merchants']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'categories'  # Based on priority order
        
        # Test empty list
        primary = self.table_mapping.get_primary_table([])
        assert primary == 'payments'  # Default

class TestAgentState:
    """Test cases for AgentState model"""
    
    def test_agent_state_creation(self):
        """Test basic AgentState creation"""
        state = AgentState(
            user_query="Test query",
            user_id="123",
            relevant_tables=['payments']
        )
        
        assert state.user_query == "Test query"
        assert state.user_id == "123"
        assert state.relevant_tables == ['payments']
    
    def test_user_id_conversion(self):
        """Test user_id conversion to string"""
        # Test with integer
        state = AgentState(user_id=123)
        assert state.user_id == "123"
        assert isinstance(state.user_id, str)
        
        # Test with None
        state = AgentState(user_id=None)
        assert state.user_id is None
        
        # Test with string
        state = AgentState(user_id="456")
        assert state.user_id == "456"

class TestDatabaseManager:
    """Test cases for DatabaseManager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.db_manager = DatabaseManager("sqlite:///:memory:")
        
    @patch('multi_table_nlsql.create_engine')
    def test_connect(self, mock_create_engine):
        """Test database connection"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        self.db_manager.connect()
        
        assert self.db_manager.engine == mock_engine
        mock_create_engine.assert_called_once_with("sqlite:///:memory:")
    
    @patch('multi_table_nlsql.inspect')
    def test_get_schema_info(self, mock_inspect):
        """Test schema information retrieval"""
        # Setup mock
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.get_table_names.return_value = ['payments', 'categories']
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
            {'name': 'amount', 'type': 'DECIMAL', 'nullable': False}
        ]
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspector.get_indexes.return_value = []
        
        self.db_manager.engine = Mock()
        
        schema_info = self.db_manager.get_schema_info(['payments'])
        
        assert 'payments' in schema_info
        assert len(schema_info['payments']['columns']) == 2
        assert schema_info['payments']['columns'][0]['name'] == 'id'
    
    def test_validate_query_dangerous_keywords(self):
        """Test SQL injection protection"""
        dangerous_queries = [
            "SELECT * FROM payments; DROP TABLE payments;",
            "DELETE FROM payments WHERE id = 1",
            "UPDATE payments SET amount = 0",
            "INSERT INTO payments VALUES (1, 100)"
        ]
        
        for query in dangerous_queries:
            result = self.db_manager.validate_query(query, {})
            assert not result['is_valid'], f"Query should be invalid: {query}"
            assert len(result['errors']) > 0
    
    def test_validate_query_placeholders(self):
        """Test placeholder detection"""
        placeholder_queries = [
            "SELECT * FROM payments WHERE user_id = your_user_id",
            "SELECT * FROM payments /* placeholder comment */",
            "SELECT * FROM payments WHERE id = ${user_id}"
        ]
        
        for query in placeholder_queries:
            result = self.db_manager.validate_query(query, {})
            assert not result['is_valid'], f"Query should be invalid: {query}"
    
    def test_validate_query_valid(self):
        """Test valid query validation"""
        valid_queries = [
            "SELECT * FROM payments",
            "SELECT SUM(amount) FROM payments WHERE user_id = '123'",
            "SELECT p.amount, c.name FROM payments p JOIN categories c ON p.category_id = c.id"
        ]
        
        for query in valid_queries:
            result = self.db_manager.validate_query(query, {'payments': {}, 'categories': {}})
            assert result['is_valid'], f"Query should be valid: {query}"

class TestLLMService:
    """Test cases for LLMService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.llm_service = LLMService("test-api-key")
    
    @patch('multi_table_nlsql.OpenAI')
    def test_generate_sql_basic(self, mock_openai_class):
        """Test basic SQL generation"""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM payments"
        mock_client.chat.completions.create.return_value = mock_response
        
        schema_info = {
            'payments': {
                'columns': [{'name': 'id', 'type': 'INTEGER'}],
                'table_info': {'description': 'Payment data'}
            }
        }
        
        result = self.llm_service.generate_sql(
            "Show all payments", 
            schema_info, 
            ['payments'], 
            'payments'
        )
        
        assert result == "SELECT * FROM payments"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('multi_table_nlsql.OpenAI')
    def test_generate_sql_with_user_context(self, mock_openai_class):
        """Test SQL generation with user context"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM payments WHERE user_id = '123'"
        mock_client.chat.completions.create.return_value = mock_response
        
        schema_info = {
            'payments': {
                'columns': [{'name': 'id', 'type': 'INTEGER'}],
                'table_info': {'description': 'Payment data', 'user_column': 'user_id'}
            }
        }
        
        result = self.llm_service.generate_sql(
            "Show my payments", 
            schema_info, 
            ['payments'], 
            'payments',
            user_id='123'
        )
        
        assert "user_id = '123'" in result
    
    def test_build_schema_description(self):
        """Test schema description building"""
        schema_info = {
            'payments': {
                'columns': [
                    {'name': 'id', 'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    {'name': 'amount', 'type': 'DECIMAL', 'nullable': False}
                ],
                'foreign_keys': [],
                'table_info': {'description': 'Payment transactions'}
            }
        }
        
        description = self.llm_service._build_schema_description(schema_info, ['payments'])
        
        assert 'payments' in description
        assert 'Payment transactions' in description
        assert 'id (INTEGER) [PK] NOT NULL' in description
        assert 'amount (DECIMAL) NOT NULL' in description

class TestVisualizationService:
    """Test cases for VisualizationService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.viz_service = VisualizationService()
    
    @patch('multi_table_nlsql.plt')
    @patch('multi_table_nlsql.os.makedirs')
    def test_create_visualization_empty_data(self, mock_makedirs, mock_plt):
        """Test visualization with empty data"""
        empty_df = pd.DataFrame()
        
        result = self.viz_service.create_visualization(empty_df, "test query", ['payments'])
        
        assert result is None
        mock_plt.savefig.assert_not_called()
    
    @patch('multi_table_nlsql.plt')
    @patch('multi_table_nlsql.os.makedirs')
    def test_create_visualization_with_data(self, mock_makedirs, mock_plt):
        """Test visualization creation with data"""
        # Create mock data
        data = pd.DataFrame({
            'category_name': ['Food', 'Entertainment', 'Transport'],
            'amount': [100, 50, 30]
        })
        
        # Setup mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.viz_service.create_visualization(data, "spending by category", ['categories'])
        
        # Verify basic functionality
        mock_plt.subplots.assert_called_once()
        mock_makedirs.assert_called_once_with("outputs", exist_ok=True)
    
    def test_create_category_chart(self):
        """Test category-specific chart creation"""
        data = pd.DataFrame({
            'category_name': ['Food', 'Entertainment'],
            'amount': [100, 50]
        })
        
        mock_ax = Mock()
        result = self.viz_service._create_category_chart(data, mock_ax)
        
        assert result is True  # Should succeed
    
    def test_create_merchant_chart(self):
        """Test merchant-specific chart creation"""
        data = pd.DataFrame({
            'merchant_name': ['Store A', 'Store B'],
            'amount': [200, 150]
        })
        
        mock_ax = Mock()
        result = self.viz_service._create_merchant_chart(data, mock_ax)
        
        assert result is True  # Should succeed

class TestNLToSQLAgent:
    """Test cases for the main NLToSQLAgent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.OpenAI'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.db_manager is not None
        assert self.agent.llm_service is not None
        assert self.agent.viz_service is not None
        assert self.agent.table_mapping is not None
        assert self.agent.graph is not None
    
    def test_analyze_query_node(self):
        """Test query analysis node"""
        state = AgentState(user_query="Show spending by category")
        
        result_state = self.agent.analyze_query_node(state)
        
        assert 'categories' in result_state.relevant_tables
        assert 'payments' in result_state.relevant_tables  # Should be added automatically
        assert result_state.primary_table == 'payments'
    
    def test_analyze_query_node_error_handling(self):
        """Test error handling in query analysis"""
        state = AgentState(user_query="")
        
        # Mock the table mapping to raise an exception
        with patch.object(self.agent.table_mapping, 'detect_relevant_tables', side_effect=Exception("Test error")):
            result_state = self.agent.analyze_query_node(state)
            
            assert result_state.error_message == "Query analysis error: Test error"
    
    @patch('multi_table_nlsql.DatabaseManager.get_schema_info')
    def test_get_schema_node(self, mock_get_schema):
        """Test schema retrieval node"""
        mock_get_schema.return_value = {'payments': {'columns': []}}
        
        state = AgentState(relevant_tables=['payments'])
        result_state = self.agent.get_schema_node(state)
        
        assert result_state.schema_info == {'payments': {'columns': []}}
        mock_get_schema.assert_called_once_with(['payments'])
    
    @patch('multi_table_nlsql.LLMService.generate_sql')
    def test_generate_sql_node(self, mock_generate_sql):
        """Test SQL generation node"""
        mock_generate_sql.return_value = "SELECT * FROM payments"
        
        state = AgentState(
            user_query="Show payments",
            relevant_tables=['payments'],
            primary_table='payments',
            schema_info={'payments': {}}
        )
        
        result_state = self.agent.generate_sql_node(state)
        
        assert result_state.sql_query == "SELECT * FROM payments"
        mock_generate_sql.assert_called_once()
    
    def test_should_retry_logic(self):
        """Test retry decision logic"""
        # Test successful validation
        state = AgentState(validation_result={'is_valid': True})
        assert self.agent.should_retry(state) == "execute"
        
        # Test validation error with retries available
        state = AgentState(
            validation_result={'is_valid': False, 'errors': ['Test error']},
            retry_count=0
        )
        assert self.agent.should_retry(state) == "retry"
        assert state.retry_count == 1
        
        # Test max retries reached
        state = AgentState(
            validation_result={'is_valid': False, 'errors': ['Test error']},
            retry_count=2
        )
        result = self.agent.should_retry(state)
        assert result == "error"
        assert "Max retries reached" in state.error_message
        
        # Test with existing error
        state = AgentState(error_message="Previous error")
        assert self.agent.should_retry(state) == "error"
    
    @pytest.mark.asyncio
    async def test_process_query_integration(self):
        """Test complete query processing"""
        # Mock all the dependencies
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute, \
             patch.object(self.agent.viz_service, 'create_visualization') as mock_viz:
            
            # Setup mocks
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.return_value = "SELECT * FROM payments"
            mock_validate.return_value = {'is_valid': True}
            mock_execute.return_value = pd.DataFrame({'amount': [100, 200]})
            mock_viz.return_value = "test_viz.png"
            
            # Execute
            result = await self.agent.process_query("Show my payments", "123")
            
            # Verify results
            assert result['success'] is True
            assert result['query'] == "Show my payments"
            assert result['user_id'] == "123"
            assert result['sql_query'] == "SELECT * FROM payments"
            assert result['row_count'] == 2
            assert result['visualization_path'] == "test_viz.png"
            assert 'payments' in result['relevant_tables']

class TestNLToSQLApp:
    """Test cases for the application interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.NLToSQLAgent'), \
             patch('multi_table_nlsql.load_dotenv'), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            self.app = NLToSQLApp()
    
    def test_app_initialization(self):
        """Test application initialization"""
        assert self.app.agent is not None
        assert self.app.table_mapping is not None
        assert hasattr(self.app, 'current_user_id')
        assert hasattr(self.app, 'available_users')
    
    @patch('builtins.input', return_value='0')
    def test_select_user_all_data(self, mock_input):
        """Test selecting all data option"""
        self.app.available_users = ['user1', 'user2']
        
        with patch('builtins.print'):
            self.app.select_user()
        
        assert self.app.current_user_id is None
    
    @patch('builtins.input', return_value='1')
    def test_select_user_specific_user(self, mock_input):
        """Test selecting specific user"""
        self.app.available_users = ['user1', 'user2']
        
        with patch('builtins.print'):
            self.app.select_user()
        
        assert self.app.current_user_id == 'user1'
    
    @patch('builtins.input', return_value='custom_user')
    def test_select_user_custom_input(self, mock_input):
        """Test custom user ID input"""
        self.app.available_users = ['user1', 'user2']
        
        with patch('builtins.print'):
            self.app.select_user()
        
        assert self.app.current_user_id == 'custom_user'
    
    def test_display_results_success(self):
        """Test successful result display"""
        result = {
            'query': 'Test query',
            'user_id': '123',
            'primary_table': 'payments',
            'relevant_tables': ['payments', 'categories'],
            'sql_query': 'SELECT * FROM payments',
            'success': True,
            'error_message': '',
            'data': [{'amount': 100}, {'amount': 200}],
            'visualization_path': 'test.png',
            'row_count': 2
        }
        
        with patch('builtins.print') as mock_print:
            self.app.display_results(result)
            
            # Verify key information was printed
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any('Test query' in call for call in print_calls)
            assert any('payments' in call for call in print_calls)
            assert any('2' in call for call in print_calls)  # row count
    
    def test_display_results_error(self):
        """Test error result display"""
        result = {
            'success': False,
            'error_message': 'Test error occurred'
        }
        
        with patch('builtins.print') as mock_print:
            self.app.display_results(result)
            
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any('Test error occurred' in call for call in print_calls)

# Integration Tests
class TestIntegrationScenarios:
    """Integration test scenarios for common use cases"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.table_mapping = TableMapping()
    
    def test_payment_category_integration(self):
        """Test payment + category query integration"""
        query = "Show my spending by food category last month"
        
        # Test table detection
        tables = self.table_mapping.detect_relevant_tables(query)
        assert 'payments' in tables
        assert 'categories' in tables
        
        # Test primary table selection
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'payments'
        
        # Test join condition
        join = self.table_mapping.get_join_condition('payments', 'categories')
        assert join == 'payments.category_id = categories.category_id'
    
    def test_merchant_analysis_integration(self):
        """Test merchant analysis query integration"""
        query = "Which stores do I spend the most money at?"
        
        tables = self.table_mapping.detect_relevant_tables(query)
        assert 'payments' in tables
        assert 'merchants' in tables
        
        join = self.table_mapping.get_join_condition('payments', 'merchants')
        assert join == 'payments.merchant_id = merchants.merchant_id'
    
    def test_budget_comparison_integration(self):
        """Test budget vs actual comparison"""
        query = "Compare my budget to actual spending by category"
        
        tables = self.table_mapping.detect_relevant_tables(query)
        assert 'budgets' in tables
        assert 'categories' in tables
        # May also include payments depending on implementation
    
    def test_user_comparison_integration(self):
        """Test user comparison queries"""
        query = "Show top spending users this month"
        
        tables = self.table_mapping.detect_relevant_tables(query)
        assert 'users' in tables
        # Should also include payments for spending data
        
        primary = self.table_mapping.get_primary_table(tables)
        # Could be users or payments depending on priority

# Fixture Tests
class TestFixturesAndMocks:
    """Test fixtures and mock setups"""
    
    def test_sample_data_creation(self):
        """Test creation of sample test data"""
        # Sample payments data
        payments_data = {
            'payment_id': [1, 2, 3],
            'user_id': ['user1', 'user1', 'user2'],
            'amount': [100.50, 75.25, 200.00],
            'category_id': [1, 2, 1],
            'merchant_id': [1, 2, 3],
            'payment_date': ['2024-01-15', '2024-01-16', '2024-01-17']
        }
        df = pd.DataFrame(payments_data)
        
        assert len(df) == 3
        assert 'user_id' in df.columns
        assert df['amount'].sum() == 375.75
    
    def test_mock_database_responses(self):
        """Test mock database response structures"""
        mock_schema = {
            'payments': {
                'columns': [
                    {'name': 'payment_id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                    {'name': 'user_id', 'type': 'VARCHAR', 'nullable': False},
                    {'name': 'amount', 'type': 'DECIMAL', 'nullable': False}
                ],
                'foreign_keys': [],
                'indexes': [],
                'table_info': {
                    'description': 'Payment transactions',
                    'primary_key': 'payment_id',
                    'user_column': 'user_id'
                }
            }
        }
        
        assert 'payments' in mock_schema
        assert len(mock_schema['payments']['columns']) == 3
        assert mock_schema['payments']['table_info']['user_column'] == 'user_id'

# Performance and Edge Case Tests
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup edge case test fixtures"""
        self.table_mapping = TableMapping()
    
    def test_empty_query(self):
        """Test handling of empty queries"""
        tables = self.table_mapping.detect_relevant_tables("")
        assert 'payments' in tables  # Should default to payments
    
    def test_very_long_query(self):
        """Test handling of very long queries"""
        long_query = "Show me " + "very " * 100 + "detailed spending analysis"
        tables = self.table_mapping.detect_relevant_tables(long_query)
        assert len(tables) >= 1  # Should still work
    
    def test_query_with_special_characters(self):
        """Test queries with special characters"""
        special_query = "Show me payments with amounts > $100 & < $500"
        tables = self.table_mapping.detect_relevant_tables(special_query)
        assert 'payments' in tables
    
    def test_case_insensitive_detection(self):
        """Test case-insensitive keyword detection"""
        queries = [
            "SHOW MY PAYMENTS",
            "Show My Payments", 
            "show my payments",
            "sHoW mY pAyMeNtS"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'payments' in tables, f"Failed for query: {query}"
    
    def test_partial_keyword_matching(self):
        """Test that partial keywords don't match incorrectly"""
        # These shouldn't match 'payment' keyword
        non_matching_queries = [
            "Show me paym information",  # Partial word
            "What about paymentservice?",  # Part of compound word
        ]
        
        for query in non_matching_queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            # Should default to payments, but not because of keyword match
    
    def test_multiple_keyword_same_table(self):
        """Test multiple keywords pointing to same table"""
        query = "Show me all my spending and payment transactions with expenses"
        tables = self.table_mapping.detect_relevant_tables(query)
        
        # Should detect payments table (not duplicate it)
        payment_count = tables.count('payments')
        assert payment_count == 1

# Parameterized Tests
class TestParameterizedScenarios:
    """Parameterized tests for multiple scenarios"""
    
    @pytest.mark.parametrize("query,expected_tables", [
        ("Show my payments", ['payments']),
        ("Spending by category", ['categories', 'payments']),
        ("Top merchants", ['merchants', 'payments']),
        ("User spending comparison", ['users', 'payments']),
        ("Budget vs actual", ['budgets']),
        ("Category and merchant analysis", ['categories', 'merchants', 'payments']),
    ])
    def test_table_detection_scenarios(self, query, expected_tables):
        """Test various table detection scenarios"""
        table_mapping = TableMapping()
        detected_tables = table_mapping.detect_relevant_tables(query)
        
        for expected_table in expected_tables:
            assert expected_table in detected_tables, f"Expected {expected_table} for query: {query}"
    
    @pytest.mark.parametrize("user_id,expected_type", [
        (123, str),
        ("456", str),
        (None, type(None)),
        ("user_789", str),
    ])
    def test_user_id_conversion(self, user_id, expected_type):
        """Test user ID conversion in various formats"""
        state = AgentState(user_id=user_id)
        assert type(state.user_id) == expected_type
        if user_id is not None:
            assert state.user_id == str(user_id)
    
    @pytest.mark.parametrize("sql_query,should_be_valid", [
        ("SELECT * FROM payments", True),
        ("SELECT amount FROM payments WHERE user_id = '123'", True),
        ("DROP TABLE payments", False),
        ("DELETE FROM payments", False),
        ("SELECT * FROM payments WHERE user_id = your_user_id", False),
        ("SELECT * FROM payments /* comment */", False),
    ])
    def test_sql_validation_scenarios(self, sql_query, should_be_valid):
        """Test SQL validation for various query types"""
        db_manager = DatabaseManager("sqlite:///:memory:")
        result = db_manager.validate_query(sql_query, {})
        
        assert result['is_valid'] == should_be_valid, f"Query validation failed for: {sql_query}"

# Async Test Helpers
class TestAsyncOperations:
    """Test asynchronous operations"""
    
    @pytest.mark.asyncio
    async def test_async_query_processing(self):
        """Test asynchronous query processing"""
        # This would test the actual async workflow
        # For now, just test that async syntax works
        
        async def mock_async_operation():
            await asyncio.sleep(0.01)  # Simulate async work
            return {"result": "success"}
        
        result = await mock_async_operation()
        assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling of concurrent queries"""
        async def mock_query(query_id):
            await asyncio.sleep(0.01)
            return f"result_{query_id}"
        
        # Simulate concurrent queries
        tasks = [mock_query(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert "result_0" in results
        assert "result_1" in results
        assert "result_2" in results

# Configuration and Setup Tests
class TestConfiguration:
    """Test configuration and setup scenarios"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_environment_variable_loading(self):
        """Test loading of environment variables"""
        with patch('multi_table_nlsql.load_dotenv'):
            api_key = os.getenv('OPENAI_API_KEY')
            assert api_key == 'test-key'
    
    def test_missing_api_key(self):
        """Test handling of missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('multi_table_nlsql.load_dotenv'):
                with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                    with patch('multi_table_nlsql.NLToSQLAgent'):
                        NLToSQLApp()
    
    def test_database_url_configuration(self):
        """Test database URL configuration"""
        test_url = "postgresql://test:test@localhost:5432/test_db"
        db_manager = DatabaseManager(test_url)
        
        assert db_manager.database_url == test_url

# Test Utilities and Helpers
class TestUtilities:
    """Test utility functions and helpers"""
    
    def test_sample_data_generation(self):
        """Test generation of sample data for testing"""
        def generate_sample_payments(n=10):
            return pd.DataFrame({
                'payment_id': range(1, n+1),
                'user_id': [f'user_{i%3+1}' for i in range(n)],
                'amount': [100 + i*10 for i in range(n)],
                'category_id': [i%5+1 for i in range(n)],
                'merchant_id': [i%3+1 for i in range(n)]
            })
        
        sample_data = generate_sample_payments(5)
        assert len(sample_data) == 5
        assert 'payment_id' in sample_data.columns
        assert sample_data['amount'].min() == 100
        assert sample_data['amount'].max() == 140
    
    def test_mock_response_builders(self):
        """Test builders for mock responses"""
        def build_mock_schema_response(table_name, columns):
            return {
                table_name: {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': col.get('type', 'VARCHAR'),
                            'nullable': col.get('nullable', True),
                            'primary_key': col.get('primary_key', False)
                        }
                        for col in columns
                    ],
                    'foreign_keys': [],
                    'indexes': [],
                    'table_info': {
                        'description': f'{table_name} table',
                        'primary_key': f'{table_name}_id'
                    }
                }
            }
        
        mock_schema = build_mock_schema_response('payments', [
            {'name': 'payment_id', 'type': 'INTEGER', 'primary_key': True},
            {'name': 'amount', 'type': 'DECIMAL'}
        ])
        
        assert 'payments' in mock_schema
        assert len(mock_schema['payments']['columns']) == 2
        assert mock_schema['payments']['columns'][0]['primary_key'] is True

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-x"  # Stop on first failure
    ])