# 

import os
from dotenv import load_dotenv

def load_environment():
    """
    Load environment-specific .env file and validate required variables.

    - Uses FLASK_ENV (default "development") to pick .env.development,
      .env.test/.env.testing, or .env.production
    - Requires:
      * SCENGEN_DEV_DB_URL if env="development"
      * SCENGEN_TEST_DB_URL if env in {"test","testing"}
      * SCENGEN_DB_URL      if env="production"
      * ALPHAVANTAGE_API_KEY always
    Returns the current (normalized) environment name.
    """
    # 1) Read FLASK_ENV (default to development)
    env = os.getenv("FLASK_ENV", "development")

    # 2) Load the corresponding .env file
    #    (so if env == "testing", this loads ".env.testing")
    dotenv_path = f".env.{env}"
    load_dotenv(dotenv_path, override=True)

    # 3) Map env to the right DB key
    if env == "production":
        db_key = "SCENGEN_DB_URL"
    elif env == "testing":
        db_key = "SCENGEN_TEST_DB_URL"
    else:
        db_key = "SCENGEN_DEV_DB_URL"

    # 4) Validate that the DB URL + API key exist
    missing = [k for k in (db_key, "ALPHAVANTAGE_API_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")

    return env
