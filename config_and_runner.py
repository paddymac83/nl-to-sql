#!/usr/bin/env python3
"""
Enhanced Configuration and Test Runner for Multi-Table NL to SQL Agent with LangGraph
"""

import os
import sys
import subprocess
import argparse
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Debug: Print Python path and check for module
print(f"Python path includes: {str(project_root)}")
multi_table_file = project_root / "multi_table_nlsql.py"
if multi_table_file.exists():
    print(f"✅ Found multi_table_nlsql.py at: {multi_table_file}")
else:
    print(f"❌ multi_table_nlsql.py not found at {multi_table_file}")
    print("Available files:")
    for f in project_root.glob("*.py"):
        print(f"  {f}")

class ConfigManager:
    """Manages configuration for the enhanced application"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.required_env_vars = [
            "OPENAI_API_KEY",
            "DATABASE_URL"
        ]
    
    def check_environment(self):
        """Check if all required environment variables are set"""
        missing_vars = []
        
        if self.env_file.exists():
            print(f"✅ Found .env file at {self.env_file}")
        else:
            print(f"⚠️  No .env file found. Please copy .env.example to .env")
            return False
        
        # Load .env file manually for checking
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        if key in self.required_env_vars and (not value or value == 'put_your_openai_api_key_here'):
                            missing_vars.append(key)
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
            return False
        
        if missing_vars:
            print(f"❌ Missing or incomplete environment variables: {', '.join(missing_vars)}")
            print("Please update your .env file with valid values.")
            return False
        
        print("✅ All required environment variables are configured")
        return True
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        try:
            # Test basic imports
            import pandas, sqlalchemy, openai, langgraph, pydantic, pytest
            
            # Test our module import
            try:
                import multi_table_nlsql
                from multi_table_nlsql import NLToSQLAgent, NLToSQLApp, InputPayload, TableMapping
                print("✅ All required dependencies and modules are available")
                return True
            except ImportError as e:
                print(f"❌ Cannot import multi_table_nlsql module: {e}")
                print("Make sure multi_table_nlsql.py is in the correct location")
                return False
                
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Please run: pip install -r requirements.txt")
            return False
    
    def check_langgraph_compatibility(self):
        """Check LangGraph compatibility"""
        try:
            from langgraph.graph import StateGraph, END
            from langchain_core.pydantic_v1 import BaseModel
            print("✅ LangGraph dependencies are compatible")
            return True
        except ImportError as e:
            print(f"❌ LangGraph compatibility issue: {e}")
            print("Try: pip install --upgrade langgraph langchain-core")
            return False
    
    def setup_database(self):
        """Instructions for enhanced database setup"""
        print("\n📊 Enhanced Database Setup Instructions:")
        print("1. Install PostgreSQL if not already installed")
        print("2. Create the database and user:")
        print("   sudo -u postgres psql")
        print("   CREATE DATABASE nlsql_db;")
        print("   CREATE USER nlsql_user WITH ENCRYPTED PASSWORD 'nlsql_password';")
        print("   GRANT ALL PRIVILEGES ON DATABASE nlsql_db TO nlsql_user;")
        print("   \\q")
        print("3. Run the enhanced setup script:")
        print("   psql -U nlsql_user -d nlsql_db -f setup_database_enhanced.sql")
        print("4. Verify the setup includes new tables: accounts, enhanced payments")
        print("5. Test payload functionality with sample data")

class TestRunner:
    """Enhanced test runner with LangGraph-specific tests"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def run_unit_tests(self, verbose=True):
        """Run unit tests only"""
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(self.project_root)
        
        cmd = [
            "python", "-m", "pytest",
            str(self.project_root / "test_multi_table_nlsql.py"),
            "-m", "unit",
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]
        
        print("🧪 Running unit tests...")
        result = subprocess.run(cmd, cwd=self.project_root, env=test_env)
        return result.returncode == 0
    
    def run_integration_tests(self, verbose=True):
        """Run integration tests only"""
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(self.project_root)
        
        cmd = [
            "python", "-m", "pytest", 
            str(self.project_root / "test_multi_table_nlsql.py"),
            "-m", "integration",
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]
        
        print("🔗 Running integration tests...")
        result = subprocess.run(cmd, cwd=self.project_root, env=test_env)
        return result.returncode == 0
    
    def run_langgraph_tests(self, verbose=True):
        """Run LangGraph-specific tests"""
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(self.project_root)
        
        cmd = [
            "python", "-m", "pytest",
            str(self.project_root / "test_multi_table_nlsql.py"),
            "-m", "langgraph",
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]
        
        print("🔄 Running LangGraph workflow tests...")
        result = subprocess.run(cmd, cwd=self.project_root, env=test_env)
        return result.returncode == 0
    
    def run_all_tests(self, verbose=True, coverage=True):
        """Run all tests with optional coverage"""
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(self.project_root)
        
        cmd = ["python", "-m", "pytest", str(self.project_root / "test_multi_table_nlsql.py")]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        if coverage:
            cmd.extend([
                "--cov=multi_table_nlsql",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=75"
            ])
        
        print("🚀 Running all tests with LangGraph support...")
        result = subprocess.run(cmd, cwd=self.project_root, env=test_env)
        
        if coverage and result.returncode == 0:
            print("📊 Coverage report generated in htmlcov/index.html")
        
        return result.returncode == 0
    
    def run_specific_test(self, test_name, verbose=True):
        """Run a specific test or test class"""
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(self.project_root)
        
        cmd = [
            "python", "-m", "pytest",
            str(self.project_root / "test_multi_table_nlsql.py"),
            "-k", test_name,
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]
        
        print(f"🎯 Running specific test: {test_name}")
        result = subprocess.run(cmd, cwd=self.project_root, env=test_env)
        return result.returncode == 0

class ApplicationRunner:
    """Enhanced application runner with LangGraph support"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def run_interactive(self):
        """Run the application in interactive mode"""
        print("🚀 Starting Enhanced Multi-Table NL to SQL Agent with LangGraph...")
        try:
            from multi_table_nlsql import NLToSQLApp
            app = NLToSQLApp()
            app.run_interactive()
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except Exception as e:
            print(f"❌ Error starting application: {e}")
            print("Common solutions:")
            print("1. Check your .env file has valid OPENAI_API_KEY")
            print("2. Ensure PostgreSQL is running and accessible")
            print("3. Verify database tables exist (run setup script)")
            print("4. Check LangGraph dependencies: pip install --upgrade langgraph")
            return False
        return True
    
    def run_demo_queries(self):
        """Run enhanced demo queries with payload examples"""
        demo_queries = [
            ("Show total spending by category", None, None),
            ("What are my recent payments?", "user_001", None),
            ("Show account balance", None, {"account_number": 900914}),
            ("Customer spending analysis", None, {"CIN": 22, "sort_code": 123456}),
            ("Show payments for specific account", "user_001", {"account_number": 900914, "CIN": 22}),
        ]
        
        print("🎪 Running enhanced demo queries with payload support...")
        
        try:
            from multi_table_nlsql import NLToSQLAgent, InputPayload
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            db_url = os.getenv('DATABASE_URL', 'postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db')
            
            if not api_key:
                print("❌ OPENAI_API_KEY not found in environment")
                return False
            
            agent = NLToSQLAgent(db_url, api_key)
            
            for i, (query, user_id, payload_dict) in enumerate(demo_queries, 1):
                print(f"\n--- Demo Query {i} ---")
                print(f"Query: {query}")
                print(f"User: {user_id or 'All users'}")
                print(f"Payload: {payload_dict or 'None'}")
                
                try:
                    # Convert payload dict to InputPayload if provided
                    payload = InputPayload(**payload_dict) if payload_dict else None
                    
                    result = asyncio.run(agent.process_query(query, user_id, payload))
                    
                    if result['success']:
                        print(f"✅ SQL: {result['sql_query']}")
                        print(f"📊 Rows: {result['row_count']}")
                        print(f"🏛️  Tables: {', '.join(result['relevant_tables'])}")
                        if result['formatted_data']:
                            metadata = result['formatted_data']['metadata']
                            print(f"📈 Suggested charts: {', '.join(metadata['suggested_charts'])}")
                    else:
                        print(f"❌ Error: {result['error_message']}")
                
                except Exception as e:
                    print(f"❌ Demo query failed: {e}")
        
        except Exception as e:
            print(f"❌ Failed to run demo: {e}")
            return False
        
        return True
    
    def test_payload_functionality(self):
        """Test payload functionality specifically"""
        print("📦 Testing payload functionality...")
        
        try:
            from multi_table_nlsql import InputPayload, TableMapping
            
            # Test InputPayload creation
            payload = InputPayload(CIN=22, sort_code=123456, account_number=900914)
            print(f"✅ Created payload: {payload.dict()}")
            
            # Test filter conditions
            conditions = payload.to_filter_conditions()
            print(f"✅ Filter conditions: {conditions}")
            
            # Test table detection with payload
            table_mapping = TableMapping()
            tables = table_mapping.detect_relevant_tables("Show payments", payload)
            print(f"✅ Detected tables with payload: {tables}")
            
            return True
            
        except Exception as e:
            print(f"❌ Payload functionality test failed: {e}")
            return False

def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Table NL to SQL Agent with LangGraph")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Check setup and configuration')
    setup_parser.add_argument('--db-help', action='store_true', help='Show database setup instructions')
    setup_parser.add_argument('--test-payload', action='store_true', help='Test payload functionality')
    
    # Test commands
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    test_parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    test_parser.add_argument('--langgraph', action='store_true', help='Run LangGraph tests only')
    test_parser.add_argument('--coverage', action='store_true', default=True, help='Include coverage report')
    test_parser.add_argument('--no-coverage', dest='coverage', action='store_false', help='Skip coverage report')
    test_parser.add_argument('-k', '--keyword', help='Run tests matching keyword')
    test_parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
    
    # Run commands
    run_parser = subparsers.add_parser('run', help='Run the application')
    run_parser.add_argument('--demo', action='store_true', help='Run demo queries with payload examples')
    
    # Lint command
    lint_parser = subparsers.add_parser('lint', help='Run code linting')
    lint_parser.add_argument('--fix', action='store_true', help='Auto-fix linting issues')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config_manager = ConfigManager()
    test_runner = TestRunner()
    app_runner = ApplicationRunner()
    
    if args.command == 'setup':
        print("🔧 Checking Enhanced Multi-Table NL to SQL Agent Setup...")
        
        deps_ok = config_manager.check_dependencies()
        langgraph_ok = config_manager.check_langgraph_compatibility()
        env_ok = config_manager.check_environment()
        
        if args.db_help:
            config_manager.setup_database()
        
        if args.test_payload:
            payload_ok = app_runner.test_payload_functionality()
            print(f"📦 Payload functionality: {'✅ OK' if payload_ok else '❌ Failed'}")
        
        if deps_ok and langgraph_ok and env_ok:
            print("\n✅ Setup complete! You can now run the enhanced application.")
            print("Try: python config_and_runner.py run")
            print("Or with payload demo: python config_and_runner.py run --demo")
        else:
            print("\n❌ Setup incomplete. Please address the issues above.")
            sys.exit(1)
    
    elif args.command == 'test':
        print("🧪 Running Enhanced Multi-Table NL to SQL Agent Tests...")
        
        success = False
        
        if args.unit:
            success = test_runner.run_unit_tests(args.verbose)
        elif args.integration:
            success = test_runner.run_integration_tests(args.verbose)
        elif args.langgraph:
            success = test_runner.run_langgraph_tests(args.verbose)
        elif args.keyword:
            success = test_runner.run_specific_test(args.keyword, args.verbose)
        else:
            success = test_runner.run_all_tests(args.verbose, args.coverage)
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    
    elif args.command == 'run':
        if args.demo:
            print("🎪 Running enhanced demo mode with payload support...")
            success = app_runner.run_demo_queries()
            if not success:
                sys.exit(1)
        else:
            app_runner.run_interactive()
    
    elif args.command == 'lint':
        print("🧹 Running code linting...")
        
        files = ["multi_table_nlsql.py", "test_multi_table_nlsql.py", "config_and_runner.py"]
        
        # Run flake8
        print("Running flake8...")
        result = subprocess.run(["flake8"] + files, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ No flake8 issues found")
        else:
            print("❌ Flake8 issues:")
            print(result.stdout)
        
        # Run black
        if args.fix:
            print("Running black (auto-fix)...")
            subprocess.run(["black"] + files)
            print("✅ Code formatted with black")
        else:
            print("Running black (check only)...")
            result = subprocess.run(["black", "--check"] + files, capture_output=True)
            if result.returncode == 0:
                print("✅ Code formatting is correct")
            else:
                print("❌ Code needs formatting. Run with --fix to auto-format")
        
        # Run mypy
        print("Running mypy...")
        result = subprocess.run(["mypy", "multi_table_nlsql.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ No mypy type issues found")
        else:
            print("❌ MyPy issues:")
            print(result.stdout)

if __name__ == "__main__":
    main()