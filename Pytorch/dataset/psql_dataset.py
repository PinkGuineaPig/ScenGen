import torch
from torch.utils.data import Dataset
from Backend.app import db
from Backend.app.models import Currency, ExchangeRate
from datetime import timezone
import pandas as pd
import numpy as np

def to_native(dt):
    """Convert timezone-aware datetime to naive UTC datetime."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

class PSQLDataset(Dataset):
    """
    PyTorch Dataset for loading FX time series of selected currency pairs.

    Each sample is a window of `seq_len` consecutive timestamps of close prices
    for each currency pair, based on the timestamps of the first pair provided.
    Currency pairs can be in the form 'EUR/USD' or 'EURUSD'.
    """
    def __init__(self, app, currency_pairs, seq_len=6):
        """
        :param app: Flask app instance (for app_context)
        :param currency_pairs: list of strings, e.g. ['EUR/USD', 'GBPJPY']
        :param seq_len: sequence length for each sample
        """
        self.seq_len = seq_len
        self.pairs = currency_pairs
        self.common_dates = None

        # Load data into pandas Series per pair
        with app.app_context():
            series_dict = {}
            for idx, pair in enumerate(self.pairs):
                # parse pair codes
                if '/' in pair:
                    base_code, quote_code = pair.split('/')
                else:
                    base_code, quote_code = pair[:3], pair[3:]

                # fetch Currency objects
                base = Currency.query.filter_by(code=base_code).one()
                quote = Currency.query.filter_by(code=quote_code).one()

                # fetch exchange rates ordered by timestamp
                rates = (
                    ExchangeRate.query
                    .filter_by(
                        base_currency_id=base.id,
                        quote_currency_id=quote.id
                    )
                    .order_by(ExchangeRate.timestamp)
                    .all()
                )

                # build pandas Series: timestamp index -> close price
                timestamps = [to_native(r.timestamp) for r in rates]
                values = [float(r.close) for r in rates]
                s = pd.Series(data=values, index=pd.to_datetime(timestamps))

                # normalize index to midnight UTC (naive)
                s.index = s.index.normalize()

                series_dict[pair] = s

                # reference timeline from first series
                if idx == 0:
                    self.common_dates = s.index

        # Align all series to the reference timeline with forward-fill
        data_columns = []
        for pair in self.pairs:
            s = series_dict[pair]
            # reindex to common_dates, forward-fill missing days
            aligned = s.reindex(self.common_dates, method='ffill')
            data_columns.append(aligned.values)

        # Build data matrix [num_dates, num_pairs]
        matrix = np.stack(data_columns, axis=1)
        self.data = torch.from_numpy(matrix).float()

    def __len__(self):
        # number of sliding windows
        return self.data.size(0) - self.seq_len + 1

    def __getitem__(self, idx):
        # returns window of shape [seq_len, num_pairs]
        return self.data[idx : idx + self.seq_len]

    def get_common_dates(self):
        """Return the list of timestamps corresponding to data rows."""
        return list(self.common_dates)

    def get_pair_codes(self):
        """Return the original list of currency pair strings for this dataset."""
        return self.pairs