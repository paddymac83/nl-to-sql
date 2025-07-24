"""
Simple test suite for the Enhanced Multi-Table NL to SQL Agent
"""

import pytest
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import asyncio

# Import the classes from the main module
from multi_table_nlsql import (
    TableMapping, DatabaseManager, LLMService, 
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
            "What's my total expense?"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'payments' in tables, f"Failed to detect payments table for: {query}"
    
    def test_detect_relevant_tables_categories(self):
        """Test detection of category-related queries"""
        queries = [
            "Show spending by category",
            "Which type of expense is highest?",
            "Group my purchases by kind"
        ]
        
        for query in queries:
            tables = self.table_mapping.detect_relevant_tables(query)
            assert 'categories' in tables, f"Failed to detect categories table for: {query}"
    
    def test_get_primary_table(self):
        """Test primary table selection"""
        # Test with payments in list
        tables = ['categories', 'payments', 'merchants']
        primary = self.table_mapping.get_primary_table(tables)
        assert primary == 'payments'
        
        # Test empty list
        primary = self.table_mapping.get_primary_table([])
        assert primary == 'payments'  # Default

class TestDatabaseManager:
    """Test cases for DatabaseManager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.db_manager = DatabaseManager("sqlite:///:memory:")
    
    def test_validate_query_dangerous_keywords(self):
        """Test SQL injection protection"""
        dangerous_queries = [
            "SELECT * FROM payments; DROP TABLE payments;",
            "DELETE FROM payments WHERE id = 1",
            "UPDATE payments SET amount = 0"
        ]
        
        for query in dangerous_queries:
            result = self.db_manager.validate_query(query, {})
            assert not result['is_valid'], f"Query should be invalid: {query}"
            assert len(result['errors']) > 0
    
    def test_validate_query_valid(self):
        """Test valid query validation"""
        valid_queries = [
            "SELECT * FROM payments",
            "SELECT SUM(amount) FROM payments WHERE user_id = '123'"
        ]
        
        for query in valid_queries:
            result = self.db_manager.validate_query(query, {'payments': {}})
            assert result['is_valid'], f"Query should be valid: {query}"

class TestLLMService:
    """Test cases for LLMService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            self.llm_service = LLMService("test-api-key")
    
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

class TestNLToSQLAgent:
    """Test cases for the main NLToSQLAgent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('multi_table_nlsql.OpenAI'):
                self.agent = NLToSQLAgent("sqlite:///:memory:", "test-api-key")
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.db_manager is not None
        assert self.agent.llm_service is not None
        assert self.agent.viz_service is not None
        assert self.agent.table_mapping is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])