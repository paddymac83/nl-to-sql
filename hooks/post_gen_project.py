#!/usr/bin/env python3
"""
Post-generation hook for NL to SQL Agent cookiecutter template.
This script runs after the project is generated to set up the environment.
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors gracefully."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stderr:
                print(f"   Warning: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def setup_git_repo():
    """Initialize git repository and create initial commit."""
    if not os.path.exists('.git'):
        run_command('git init', 'Initializing Git repository')
        run_command('git add .', 'Adding files to Git')
        run_command('git commit -m "Initial commit from cookiecutter template"', 'Creating initial commit')

def create_env_file():
    """Create .env file from template."""
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        print("📄 Creating .env file from template...")
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("✅ .env file created. Please update with your actual values.")

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("🐍 Setting up Python environment...")
    
    # Create virtual environment
    venv_success = run_command(
        f'{sys.executable} -m venv venv',
        'Creating virtual environment'
    )
    
    if not venv_success:
        print("❌ Failed to create virtual environment")
        return False
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = 'venv\\Scripts\\pip'
        python_cmd = 'venv\\Scripts\\python'
    else:  # Unix/Linux/macOS
        pip_cmd = 'venv/bin/pip'
        python_cmd = 'venv/bin/python'
    
    # Upgrade pip
    run_command(f'{pip_cmd} install --upgrade pip', 'Upgrading pip')
    
    # Install requirements
    if Path('requirements.txt').exists():
        run_command(f'{pip_cmd} install -r requirements.txt', 'Installing production dependencies')
    
    if Path('requirements-dev.txt').exists():
        run_command(f'{pip_cmd} install -r requirements-dev.txt', 'Installing development dependencies', check=False)
    
    return True

def setup_pre_commit():
    """Set up pre-commit hooks if available."""
    if Path('requirements-dev.txt').exists():
        requirements_dev = Path('requirements-dev.txt').read_text()
        if 'pre-commit' in requirements_dev:
            if os.name == 'nt':  # Windows
                precommit_cmd = 'venv\\Scripts\\pre-commit'
            else:  # Unix/Linux/macOS
                precommit_cmd = 'venv/bin/pre-commit'
            
            run_command(f'{precommit_cmd} install', 'Setting up pre-commit hooks', check=False)

def create_directory_structure():
    """Ensure all necessary directories exist."""
    directories = [
        'logs',
        'data',
        'exports',
        'tests/fixtures',
        'src/utils',
        'src/visualization',
        'docs/examples',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def print_next_steps():
    """Print next steps for the user."""
    project_name = "{{ cookiecutter.project_name }}"
    project_slug = "{{ cookiecutter.project_slug }}"
    
    print("\n" + "="*60)
    print(f"🎉 {project_name} project created successfully!")
    print("="*60)
    print("\n📋 Next Steps:")
    print(f"1. cd {project_slug}")
    
    if os.name == 'nt':  # Windows
        print("2. venv\\Scripts\\activate")
    else:
        print("2. source venv/bin/activate")
    
    print("3. Update .env file with your API keys and database settings")
    print("4. Run database setup: python scripts/database_setup.py")
    print("5. Run tests: make test or python scripts/run_tests.py")
    print("6. Start developing!")
    
    print("\n📚 Useful Commands:")
    print("   make help              - Show all available commands")
    print("   make test              - Run tests")
    print("   make format            - Format code")
    print("   make lint              - Run linting")
    print("   make setup-db          - Set up database")
    
    {% if cookiecutter.use_fastapi == "yes" %}
    print("\n🚀 FastAPI Server:")
    print("   uvicorn src.main:app --reload")
    {% endif %}
    
    {% if cookiecutter.include_streamlit == "yes" %}
    print("\n🌟 Streamlit App:")
    print("   streamlit run src/streamlit_app.py")
    {% endif %}
    
    print("\n📖 Documentation:")
    print("   README.md              - Getting started guide")
    print("   docs/API.md            - API documentation")
    print("   docs/DEPLOYMENT.md     - Deployment guide")

def main():
    """Main post-generation setup function."""
    print("🚀 Setting up your NL to SQL Agent project...")
    
    try:
        # Create directory structure
        create_directory_structure()
        
        # Create .env file
        create_env_file()
        
        # Set up Python environment
        if not setup_python_environment():
            print("⚠️  Python environment setup failed. You may need to set it up manually.")
        
        # Set up pre-commit hooks
        setup_pre_commit()
        
        # Initialize git repository
        setup_git_repo()
        
        # Print next steps
        print_next_steps()
        
    except Exception as e:
        print(f"❌ Post-generation setup failed: {e}")
        print("You may need to complete the setup manually.")
        sys.exit(1)

if __name__ == "__main__":
    main()
