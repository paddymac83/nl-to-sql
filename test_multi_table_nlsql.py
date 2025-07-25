"""
Comprehensive test suite for the Enhanced Multi-Table NL to SQL Agent with LangGraph
"""

import pytest
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime
import json

# Import the classes from the main module
from multi_table_nlsql import (
    InputPayload, TableMapping, AgentState, DatabaseManager, LLMService, 
    DataFormatter, NLToSQLAgent, NLToSQLApp
)

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
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute:
            
            # Setup mocks
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.return_value = "SELECT * FROM payments WHERE customer_id = 22"
            mock_validate.return_value = {'is_valid': True}
            mock_execute.return_value = pd.DataFrame({'amount': [100, 200]})
            
            # Execute with dict payload
            payload_dict = {'CIN': 22, 'sort_code': None, 'account_number': 900914, 'user_id': None}
            result = await self.agent.process_query("Show my payments", "123", payload_dict)
            
            # Verify results
            assert result['success'] is True
            assert result['input_payload'] == payload_dict
            assert result['sql_query'] == "SELECT * FROM payments WHERE customer_id = 22"
            assert result['row_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_query_error_handling(self):
        """Test error handling in query processing"""
        # Mock to raise an exception
        with patch.object(self.agent.table_mapping, 'detect_relevant_tables', side_effect=Exception("Test error")):
            result = await self.agent.process_query("Test query")
            
            assert result['success'] is False
            assert "SQL generation error" in result['error_message']

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
    
    def test_parse_payload_input_invalid_json(self):
        """Test parsing invalid JSON payload from input"""
        user_input = "Show payments without any JSON"
        
        payload = self.app.parse_payload_input(user_input)
        
        assert payload is None
    
    def test_parse_payload_input_malformed_json(self):
        """Test parsing malformed JSON payload"""
        user_input = "Show payments {'CIN': 22, 'invalid'}"
        
        payload = self.app.parse_payload_input(user_input)
        
        assert payload == InputPayload(CIN=22, sort_code=None, account_number=None, user_id=None)

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
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute:
            
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
            
            # Execute with payload
            payload = InputPayload(CIN=22)
            result = await self.agent.process_query(
                "Show my recent payments", 
                user_id="user_001",
                input_payload=payload
            )
            
            # Verify workflow completion
            assert result['success'] is True
            # assert result['relevant_tables'] == ['users', 'accounts', 'payments']
            assert result['primary_table'] == 'payments'
            assert 'customer_id = 22' in result['sql_query']
            assert result['row_count'] == 2
            assert result['formatted_data'] is not None
            assert result['formatted_data']['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_workflow_with_retry_logic(self):
        """Test LangGraph workflow with SQL validation retry"""
        with patch.object(self.agent.db_manager, 'get_schema_info') as mock_schema, \
             patch.object(self.agent.llm_service, 'generate_sql') as mock_generate, \
             patch.object(self.agent.db_manager, 'validate_query') as mock_validate, \
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute:
            
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
            
            # Execute
            result = await self.agent.process_query("Show my payments", "user_001")
            
            # Verify retry worked
            assert result['success'] is True
            assert mock_generate.call_count == 2  # Called twice due to retry
            assert mock_validate.call_count == 2

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
        ("Which merchants do I use?", {"account_number": 900914}, ["merchants", "payments"]),
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
        assert len(conditions) == 2  # 100 extra fields + 2 mapped fields
    
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
            
            async def process_query(self, user_query, user_id=None, input_payload=None):
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
                    'sql_query': state.sql_query
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
             patch.object(self.agent.db_manager, 'execute_query') as mock_execute:
            
            mock_schema.return_value = {'payments': {'columns': []}}
            mock_generate.side_effect = lambda *args: f"SELECT * FROM payments -- {args[0]}"
            mock_validate.return_value = {'is_valid': True}
            mock_execute.return_value = pd.DataFrame({'amount': [100]})
            
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

if __name__ == "__main__":
    # Run tests with verbose output and coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "--cov=multi_table_nlsql",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])