# Shared/config/database.py
import os

def get_database_uri():
    env = os.getenv("FLASK_ENV", "development")
    if env == "testing":
        return os.getenv("SCENGEN_TEST_DB_URL")
    elif env == "production":
        return os.getenv("SCENGEN_DB_URL")
    else:
        return os.getenv("SCENGEN_DEV_DB_URL")
