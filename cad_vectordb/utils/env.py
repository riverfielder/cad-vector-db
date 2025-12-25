"""
Load environment variables from .env file

Usage:
    from cad_vectordb.utils.env import load_env
    load_env()
"""
import os
from pathlib import Path
from typing import Optional


def load_env(env_file: Optional[str] = None, override: bool = False):
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file (default: .env in project root)
        override: Whether to override existing environment variables
    """
    try:
        from dotenv import load_dotenv
        
        if env_file is None:
            # Find .env file in project root
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            env_file = project_root / '.env'
        
        if Path(env_file).exists():
            load_dotenv(env_file, override=override)
            print(f"✓ Loaded environment from: {env_file}")
        else:
            print(f"⚠ No .env file found at: {env_file}")
            print("  Using default configuration or environment variables")
    
    except ImportError:
        print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
        print("  Using default configuration or environment variables")


if __name__ == "__main__":
    load_env()
    
    # Print some config values
    print("\nCurrent configuration:")
    print(f"  ENV: {os.getenv('ENV', 'development')}")
    print(f"  API_PORT: {os.getenv('API_PORT', '8000')}")
    print(f"  LOG_LEVEL: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"  ENABLE_AUTH: {os.getenv('ENABLE_AUTH', 'false')}")
