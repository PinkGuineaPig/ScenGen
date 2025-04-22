import torch
from torch.utils.data import Dataset
from Backend.app import db
from Backend.app.models import Currency, ExchangeRate
from datetime import timezone

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
        self.pair_data = {}
        self.common_dates = None

        with app.app_context():
            for idx, pair in enumerate(self.pairs):
                # support both 'EUR/USD' and 'EURUSD'
                if '/' in pair:
                    base_code, quote_code = pair.split('/')
                else:
                    base_code, quote_code = pair[:3], pair[3:]

                base = Currency.query.filter_by(code=base_code).one()
                quote = Currency.query.filter_by(code=quote_code).one()

                rates = (
                    ExchangeRate.query
                    .filter_by(
                        base_currency_id=base.id,
                        quote_currency_id=quote.id
                    )
                    .order_by(ExchangeRate.timestamp)
                    .all()
                )
                # map timestamp to float close price
                series = {to_native(r.timestamp): float(r.close) for r in rates}
                self.pair_data[pair] = series

                # for the first pair, record its timestamps as the reference timeline
                if idx == 0:
                    self.common_dates = sorted(series.keys())

        # prepare tensor data: shape [num_dates, num_pairs]
        data_matrix = []
        for pair in self.pairs:
            series = self.pair_data[pair]
            # align to reference timeline; assume series contains all timestamps
            column = [series[ts] for ts in self.common_dates]
            data_matrix.append(column)

        # convert to Tensor of shape [num_dates, num_pairs]
        # transpose data_matrix so that rows=time, cols=pairs
        self.data = torch.tensor(data_matrix).T.float()

    def __len__(self):
        # number of sliding windows
        return self.data.size(0) - self.seq_len + 1

    def __getitem__(self, idx):
        # returns window of shape [seq_len, num_pairs]
        return self.data[idx : idx + self.seq_len]

    def get_common_dates(self):
        """Return the list of timestamps corresponding to data rows."""
        return self.common_dates

    def get_pair_codes(self):
        """Return the original list of currency pair strings for this dataset."""
        return self.pairs
