# ScenGen Project Overview

The **ScenGen** repository contains a scenario generation and latent space analysis application. Its components are organized to separate concerns, facilitate collaboration, and support testing, deployment, and extension.


## Project Layout & Responsibilities

ScenGen/
├── .env.development       # Development environment variables (git-ignored)
├── .env.test              # Testing environment variables
├── .env.production        # Production environment variables
├── .env.example           # Template listing required environment variables
├── folder_structure.py    # Script documenting project layout (optional)
├── init_proj_structure.bat# Windows script to initialize project scaffolding
├── pyproject.toml         # PEP 621 project metadata and build configuration
├── requirements.txt       # Top-level Python dependencies
├── README.md              # Project overview and quickstart guide
├── .pytest_cache/         # pytest cache (auto-generated)
├── Backend/               # Backend application (Flask/Dash + data models)
│   ├── app/
│   │   ├── __init__.py    # Application factory and extension initialization
│   │   ├── models/        # SQLAlchemy model definitions
│   │   ├── routes/        # HTTP and Dash route definitions
│   │   └── utils/         # Reusable backend utilities and data loaders
│   └── tests/             # Unit and integration tests for backend
├── Frontend/              # Frontend application (React/Dash components and pages)
│   ├── assets/            # Static assets (CSS, images)
│   ├── components/        # Reusable UI components
│   ├── pages/             # Page-level modules
│   └── tests/             # Frontend unit tests
├── Pytorch/               # PyTorch modeling and training code
│   ├── models/            # Neural network definitions
│   ├── trainers/          # Training loops and procedures
│   └── tests/             # PyTorch-specific tests
├── scripts/               # Standalone or scheduled jobs (ETL, data fetchers, migrations)
└── Shared/                # Cross-cutting configuration and utility modules
    ├── config/            # Configuration loaders (dotenv, database settings)
    └── utils/             # Helper functions shared across Backend and Frontend


## Folder Roles

- **Backend/app/utils/**  
  Contains application-specific helper modules and data loaders. Examples include:  
  - _fx_loader.py_: Fetches and upserts foreign exchange data.  
  - _init_db.py_: Initializes or migrates the database schema.  
  - Other ETL functions consumed by both scripts and the running service.

- **scripts/**  
  Holds executable Python scripts for one-off or scheduled tasks. These scripts:  
  1. Load environment variables (from Shared/config or .env files)  
  2. Import functions from Backend/app/utils  
  3. Perform tasks such as ingesting data, resetting databases, or fetching external APIs  
  4. Exit with appropriate status codes for automation pipelines

- **Shared/config/**  
  Contains the code artifacts that **load and manage configuration**, but **not** the `.env` files themselves (those stay at the project root). Typical modules include:  
  - `dotenv_loader.py`: Detects the current environment and loads the corresponding `.env.*` file into process variables.  
  - `database.py`: Constructs the SQLAlchemy connection URI (or other service URIs) from those environment variables.  
  - Other config utilities (e.g. secrets decryption, validation, or default fallbacks).

- **Shared/utils/**  
  Implements generic helper functions and constants used across the codebase, such as:  
  - Logging setup  
  - Date/time utilities  
  - Common validation routines

## Quickstart Guide

1. **Clone repository**
   ```bash
   git clone <repo-url> ScenGen
   cd ScenGen


2. **Configure environment variables**

 - Copy .env.example to each of .env.development, .env.test, and .env.production.
 - Provide real credentials and connection strings.
 - Add the Alpha Vantage API key to each environment file:
        ```
        ALPHAVANTAGE_API_KEY=<your_free_api_key>
        ```
 - Ensure Shared/config/dotenv_loader.py loads these into the process environment.

3. **Set up Python environment**
    ```
        python -m venv .venv
        source .venv/bin/activate         # Windows: .venv\Scripts\activate
        pip install -r requirements.txt
    ```

4. **Install in editable mode (optional)**
```
pip install -e .
```

5. **Initialize database schema**
```
python scripts/init_db.py
```

6. **Ingest initial data**
```
python scripts/data_ingestion.py
```

7. **Start backend server**
```
python -m Backend.app.run
```

8. **Start frontend application**







## Program Flow

This project uses a single Flask factory (`create_app()`) and a `.env.<env>` loader to switch between development, testing, and production. Below is the high‑level flow in each environment:

### Common Startup Steps

1. **Entry Point**  
   - **Dev**: `flask run` (with `FLASK_APP=Backend.app:create_app`)  
   - **Prod**: your WSGI server (e.g. Gunicorn) calls `Backend.app:create_app`  
   - **Test**: pytest’s `conftest.py` imports and calls `create_app()`

2. **Load Environment**  
   ```python
   env = os.getenv("FLASK_ENV", "development")
   load_dotenv(f".env.{env}", override=True)
   ```
   → picks `.env.development`, `.env.test`, or `.env.production`

3. **Configure Database**  
   ```python
   app.config["SQLALCHEMY_DATABASE_URI"] = get_database_uri()
   ```
   → reads `SCENGEN_DEV_DB_URL`, `SCENGEN_TEST_DB_URL`, or `SCENGEN_DB_URL`

4. **Create App & Register Blueprints**  
   - `db.init_app(app)`  
   - `app.register_blueprint(model_configs_bp)`  
   - `app.register_blueprint(model_runs_bp)`

### Development Mode

```bash
export FLASK_ENV=development   # or leave unset
flask run
```

- Loads `.env.development`  
- Connects to your dev database  
- Serves your API with debug off/on per config

### Testing Mode

```bash
pytest
```

- In `conftest.py`: `os.environ.setdefault("FLASK_ENV", "test")`  
- Loads `.env.test`  
- Spins up test DB, does `db.create_all()` once, uses transactions + rollbacks per test  
- Exercises the same Flask routes via the `client` and `session` fixtures

### Production Mode

```bash
export FLASK_ENV=production
gunicorn Backend.app:create_app
```

- Loads `.env.production`  
- Connects to your production database  
- Serves your API under Gunicorn (or your chosen WSGI server)


### Start redis server:
cd C:\Users\Kevin\Redis
.\redis-server.exe
