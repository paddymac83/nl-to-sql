#!/usr/bin/env python3
"""
Configuration and Test Runner for Multi-Table NL to SQL Agent
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to Python path so we can import multi_table_nlsql
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also try current directory
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Debug: Print Python path and check for module
print(f"Python path includes: {[str(project_root), str(current_dir)]}")
multi_table_file = project_root / "multi_table_nlsql.py"
if multi_table_file.exists():
    print(f"✅ Found multi_table_nlsql.py at: {multi_table_file}")
else:
    alt_file = current_dir / "multi_table_nlsql.py"
    if alt_file.exists():
        print(f"✅ Found multi_table_nlsql.py at: {alt_file}")
    else:
        print(f"❌ multi_table_nlsql.py not found at {multi_table_file} or {alt_file}")
        print("Available files:")
        for p in [project_root, current_dir]:
            for f in p.glob("*.py"):
                print(f"  {f}")

class ConfigManager:
    """Manages configuration for the application"""
    
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
                        if key in self.required_env_vars and (not value or value == 'your_openai_api_key_here'):
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
            import pandas, matplotlib, sqlalchemy, openai, langgraph, pydantic, pytest
            
            # Test our module import
            try:
                import multi_table_nlsql
                from multi_table_nlsql import NLToSQLAgent, NLToSQLApp
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
    
    def setup_database(self):
        """Instructions for database setup"""
        print("\n📊 Database Setup Instructions:")
        print("1. Install PostgreSQL if not already installed")
        print("2. Create the database and user:")
        print("   sudo -u postgres psql")
        print("   CREATE DATABASE nlsql_db;")
        print("   CREATE USER nlsql_user WITH ENCRYPTED PASSWORD 'nlsql_password';")
        print("   GRANT ALL PRIVILEGES ON DATABASE nlsql_db TO nlsql_user;")
        print("   \\q")
        print("3. Run the setup script:")
        print("   psql -U nlsql_user -d nlsql_db -f setup_database.sql")
        print("4. Verify the setup by checking tables exist")

class TestRunner:
    """Handles running different types of tests"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def run_unit_tests(self, verbose=True):
        """Run unit tests only"""
        cmd = [
            "python", "-m", "pytest",
            "test_multi_table_nlsql.py",
            "-m", "unit",
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]  # Remove empty strings
        
        print("🧪 Running unit tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
    
    def run_integration_tests(self, verbose=True):
        """Run integration tests only"""
        cmd = [
            "python", "-m", "pytest", 
            "test_multi_table_nlsql.py",
            "-m", "integration",
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]
        
        print("🔗 Running integration tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
    
    def run_all_tests(self, verbose=True, coverage=True):
        """Run all tests with optional coverage"""
        cmd = ["python", "-m", "pytest", "test_multi_table_nlsql.py"]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        if coverage:
            cmd.extend([
                "--cov=multi_table_nlsql",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])
        
        print("🚀 Running all tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if coverage and result.returncode == 0:
            print("📊 Coverage report generated in htmlcov/index.html")
        
        return result.returncode == 0
    
    def run_specific_test(self, test_name, verbose=True):
        """Run a specific test or test class"""
        cmd = [
            "python", "-m", "pytest",
            "test_multi_table_nlsql.py",
            "-k", test_name,
            "-v" if verbose else "",
            "--tb=short"
        ]
        cmd = [c for c in cmd if c]
        
        print(f"🎯 Running specific test: {test_name}")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0

class ApplicationRunner:
    """Handles running the application in different modes"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def run_interactive(self):
        """Run the application in interactive mode"""
        print("🚀 Starting Multi-Table NL to SQL Agent...")
        try:
            # Import here to avoid import issues
            from multi_table_nlsql import NLToSQLApp
            app = NLToSQLApp()
            app.run_interactive()
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except Exception as e:
            print(f"❌ Error starting application: {e}")
            return False
        return True
    
    def run_demo_queries(self):
        """Run a set of demo queries to test the system"""
        demo_queries = [
            ("Show total spending by category", None),
            ("What are my top 5 merchants by spending?", "user_001"),
            ("Compare budget vs actual spending", "user_001"),
            ("Show spending trends over time", None),
            ("Which users spend the most on food?", None)
        ]
        
        print("🎪 Running demo queries...")
        
        try:
            # Import here to avoid circular imports and path issues
            import sys
            import os
            from dotenv import load_dotenv
            
            # Ensure we can import the module
            if 'multi_table_nlsql' not in sys.modules:
                import multi_table_nlsql
            
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            db_url = os.getenv('DATABASE_URL', 'postgresql://nlsql_user:nlsql_password@localhost:5432/nlsql_db')
            
            from multi_table_nlsql import NLToSQLAgent
            agent = NLToSQLAgent(db_url, api_key)
            
            for i, (query, user_id) in enumerate(demo_queries, 1):
                print(f"\n--- Demo Query {i} ---")
                print(f"Query: {query}")
                print(f"User: {user_id or 'All users'}")
                
                try:
                    result = asyncio.run(agent.process_query(query, user_id))
                    
                    if result['success']:
                        print(f"✅ SQL: {result['sql_query']}")
                        print(f"📊 Rows: {result['row_count']}")
                        print(f"🏛️  Tables: {', '.join(result['relevant_tables'])}")
                    else:
                        print(f"❌ Error: {result['error_message']}")
                
                except Exception as e:
                    print(f"❌ Demo query failed: {e}")
        
        except Exception as e:
            print(f"❌ Failed to run demo: {e}")
            return False
        
        return True

def main():
    """Main entry point for the configuration and test runner"""
    parser = argparse.ArgumentParser(description="Multi-Table NL to SQL Agent - Setup and Test Runner")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Check setup and configuration')
    setup_parser.add_argument('--db-help', action='store_true', help='Show database setup instructions')
    
    # Test commands
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    test_parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    test_parser.add_argument('--coverage', action='store_true', default=True, help='Include coverage report')
    test_parser.add_argument('--no-coverage', dest='coverage', action='store_false', help='Skip coverage report')
    test_parser.add_argument('-k', '--keyword', help='Run tests matching keyword')
    test_parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
    
    # Run commands
    run_parser = subparsers.add_parser('run', help='Run the application')
    run_parser.add_argument('--demo', action='store_true', help='Run demo queries')
    
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
        print("🔧 Checking Multi-Table NL to SQL Agent Setup...")
        
        deps_ok = config_manager.check_dependencies()
        env_ok = config_manager.check_environment()
        
        if args.db_help:
            config_manager.setup_database()
        
        if deps_ok and env_ok:
            print("\n✅ Setup complete! You can now run the application.")
            print("Try: python config_and_runner.py run")
        else:
            print("\n❌ Setup incomplete. Please address the issues above.")
            sys.exit(1)
    
    elif args.command == 'test':
        print("🧪 Running Multi-Table NL to SQL Agent Tests...")
        
        success = False
        
        if args.unit:
            success = test_runner.run_unit_tests(args.verbose)
        elif args.integration:
            success = test_runner.run_integration_tests(args.verbose)
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
            print("🎪 Running demo mode...")
            app_runner.run_demo_queries()
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