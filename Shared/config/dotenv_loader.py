import os
from dotenv import load_dotenv

def load_environment():
    """
    Load environment-specific .env file and validate required variables.

    - Uses FLASK_ENV (default "development") to pick .env.development, .env.testing, or .env.production
    - Requires:
      * SCENGEN_DEV_DB_URL if env="development"
      * SCENGEN_TEST_DB_URL if env="testing"
      * SCENGEN_DB_URL      if env="production"
      * ALPHAVANTAGE_API_KEY always
    Returns the current environment name.
    """
    # Determine current environment
    env = os.getenv("FLASK_ENV", "development")
    dotenv_file = f".env.{env}"

    # Load the .env file for this environment
    load_dotenv(dotenv_file, override=True)

    # Choose the appropriate DB URL variable
    if env == "testing":
        db_key = "SCENGEN_TEST_DB_URL"
    elif env == "production":
        db_key = "SCENGEN_DB_URL"
    else:
        db_key = "SCENGEN_DEV_DB_URL"

    # Validate presence of required environment variables
    missing = [key for key in (db_key, "ALPHAVANTAGE_API_KEY") if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")

    return env
