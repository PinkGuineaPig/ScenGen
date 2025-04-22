# scripts/test_psql_dataset.py
"""
Smoke test for PSQLDataset: connects to the development database and
pulls real FX rates to verify PSQLDataset returns windowed data.
"""
import os

# 1. Ensure we use the development environment and load variables
# Default FLASK_ENV is 'development', so no override needed
from Shared.config.dotenv_loader import load_environment
load_environment()  # loads .env.development and validates SCENGEN_DEV_DB_URL, ALPHAVANTAGE_API_KEY

import torch

from Backend.app import create_app
from Pytorch.dataset.psql_dataset import PSQLDataset


def main():
    # 2. Create Flask app in development mode
    app = create_app()

    # 3. Instantiate dataset against real dev data (e.g., EUR/USD)
    ds = PSQLDataset(app, ["EUR/USD"], seq_len=6)

    # 4. Print dataset info
    print(f"Total windows (seq_len=6): {len(ds)}")
    if len(ds) > 0:
        window0 = ds[0]
        print("Sample window shape:", window0.shape)
        print("Window tensor (first 6 rows):", window0.tolist())
        print("Dates:", ds.get_common_dates()[:6])
        print("Pairs:", ds.get_pair_codes())
    else:
        print("No data available for the given pair and sequence length.")

if __name__ == "__main__":
    main()
