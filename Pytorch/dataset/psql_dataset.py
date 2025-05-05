#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
from Backend.app import db
from Backend.app.models.data_models import Currency, ExchangeRate
import pandas as pd
import numpy as np
from datetime import timezone
from sqlalchemy import tuple_, func, distinct
import logging

# Configure logging
logging.basicConfig(
    filename='logs/dataset.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def to_native(dt):
    """Convert an aware datetime to a naive UTC datetime in UTC."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def _prepare_tensor_from_price_df(price_df: pd.DataFrame, seq_len: int, stride: int, ffill_limit=None):
    idx = price_df.index.sort_values()
    price_df = price_df.reindex(idx).ffill(limit=ffill_limit)
    price_df = price_df.map(lambda x: np.nan if x <= 0 else x).dropna(how="any")
    price_df = price_df.astype(float)
    logger.info(f"Dataset size: {len(price_df)} rows")
    logger.info(f"Raw price_df min: {price_df.min().min():.6f}, max: {price_df.max().max():.6f}")

    ret = price_df.pct_change().fillna(0)
    squared_return = ret ** 2  # Compute raw squared returns
    logger.info(f"Squared returns min: {squared_return.min().min():.6f}, max: {squared_return.max().max():.6f}")

    ma_price = price_df.rolling(window=10, min_periods=1).mean().bfill()
    logger.info(f"MA price min: {ma_price.min().min():.6f}, max: {ma_price.max().max():.6f}")

    combined = pd.concat([ma_price, squared_return], axis=1)
    matrix = combined.values.astype(np.float32)
    data = torch.from_numpy(matrix)
    T = data.size(0)
    n_windows = max(0, (T - seq_len) // stride + 1)
    starts = np.arange(0, n_windows * stride, stride, dtype=np.int64)
    logger.info(f"Number of windows: {n_windows}")

    ma_price_min, ma_price_max = ma_price.min().min(), ma_price.max().max()
    squared_return_min, squared_return_max = squared_return.min().min(), squared_return.max().max() + 1e-6  # Small buffer
    logger.info(f"Global ranges: ma_price_min={ma_price_min:.6f}, ma_price_max={ma_price_max:.6f}, "
                f"squared_return_min={squared_return_min:.6f}, squared_return_max={squared_return_max:.6f}")

    return data, starts, idx, ma_price_min, ma_price_max, squared_return_min, squared_return_max

class PSQLDataset(Dataset):
    def __init__(self, app, currency_pairs, seq_len=6, stride=1, ffill_limit=None):
        self.seq_len = seq_len
        self.stride = stride
        self.pairs = currency_pairs
        self.ffill_limit = ffill_limit

        with app.app_context():
            series = []
            for pair in currency_pairs:
                base_code, quote_code = pair.split('/')
                base = Currency.query.filter_by(code=base_code).one()
                quote = Currency.query.filter_by(code=quote_code).one()
                rates = (ExchangeRate.query
                         .filter_by(base_currency_id=base.id, quote_currency_id=quote.id)
                         .order_by(ExchangeRate.timestamp)
                         .all())
                times = [to_native(r.timestamp) for r in rates]
                prices = [float(r.close) for r in rates]
                s = pd.Series(prices, index=pd.to_datetime(times), name=pair)
                series.append(s)
            price_df = pd.concat(series, axis=1)
            data, starts, idx, ma_price_min, ma_price_max, squared_return_min, squared_return_max = _prepare_tensor_from_price_df(
                price_df, seq_len, stride, ffill_limit
            )

            self.data = data
            self.starts = starts
            self.input_size = data.size(1)
            self.common_index = idx
            self.ma_price_min = ma_price_min
            self.ma_price_max = ma_price_max
            self.squared_return_min = squared_return_min
            self.squared_return_max = squared_return_max

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        window = self.data[s : s + self.seq_len]
        ma_price = window[:, 0]
        squared_return = window[:, 1]
        ma_price_min, ma_price_max = ma_price.min(), ma_price.max()
        squared_return_min, squared_return_max = squared_return.min(), squared_return.max() + 1e-6  # Small buffer
        if ma_price_max > ma_price_min:
            norm_ma_price = (ma_price - ma_price_min) / (ma_price_max - ma_price_min)
        else:
            norm_ma_price = torch.zeros_like(ma_price)
        if squared_return_max > squared_return_min:
            norm_squared_return = (squared_return - squared_return_min) / (squared_return_max - squared_return_min)
        else:
            norm_squared_return = torch.zeros_like(squared_return)
        norm_window = torch.stack([norm_ma_price, norm_squared_return], dim=-1)
        return norm_window, (ma_price_min, ma_price_max, squared_return_min, squared_return_max)

    def get_common_dates(self):
        return list(self.common_index)

    def get_pair_codes(self):
        return list(self.pairs)

class InMemoryWindowDataset(Dataset):
    def __init__(self, app, currency_pairs, seq_len=6, stride=1, ffill_limit=None):
        self.seq_len = seq_len
        self.stride = stride
        self.pairs = currency_pairs
        self.ffill_limit = ffill_limit

        with app.app_context():
            session = db.session
            id_pairs = []
            for pair in currency_pairs:
                bc, qc = pair.split('/')
                b = session.query(Currency.id).filter_by(code=bc).scalar()
                q = session.query(Currency.id).filter_by(code=qc).scalar()
                id_pairs.append((b, q))
            pair_count = len(id_pairs)

            common_cte = (
                session.query(ExchangeRate.timestamp.label('ts'))
                .filter(tuple_(ExchangeRate.base_currency_id, ExchangeRate.quote_currency_id).in_(id_pairs))
                .group_by(ExchangeRate.timestamp)
                .having(func.count(distinct(tuple_(ExchangeRate.base_currency_id, ExchangeRate.quote_currency_id))) == pair_count)
                .cte('common_ts')
            )
            rows = (
                session.query(ExchangeRate.timestamp, ExchangeRate.base_currency_id, ExchangeRate.quote_currency_id, ExchangeRate.close)
                .join(common_cte, ExchangeRate.timestamp == common_cte.c.ts)
                .filter(tuple_(ExchangeRate.base_currency_id, ExchangeRate.quote_currency_id).in_(id_pairs))
                .order_by(ExchangeRate.timestamp)
                .all()
            )
            df = pd.DataFrame(rows, columns=['timestamp', 'base_id', 'quote_id', 'close'])
            df['close'] = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['pair'] = df['base_id'].astype(str) + '/' + df['quote_id'].astype(str)
            price_df = df.pivot(index='timestamp', columns='pair', values='close')

            data, starts, idx, ma_price_min, ma_price_max, squared_return_min, squared_return_max = _prepare_tensor_from_price_df(
                price_df, seq_len, stride, ffill_limit
            )
            self.data = data
            self.starts = starts
            self.input_size = data.size(1)
            self.common_index = idx
            self.ma_price_min = ma_price_min
            self.ma_price_max = ma_price_max
            self.squared_return_min = squared_return_min
            self.squared_return_max = squared_return_max

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        window = self.data[s : s + self.seq_len]
        ma_price = window[:, 0]
        squared_return = window[:, 1]
        ma_price_min, ma_price_max = ma_price.min(), ma_price.max()
        squared_return_min, squared_return_max = squared_return.min(), squared_return.max() + 1e-6  # Small buffer
        if ma_price_max > ma_price_min:
            norm_ma_price = (ma_price - ma_price_min) / (ma_price_max - ma_price_min)
        else:
            norm_ma_price = torch.zeros_like(ma_price)
        if squared_return_max > squared_return_min:
            norm_squared_return = (squared_return - squared_return_min) / (squared_return_max - squared_return_min)
        else:
            norm_squared_return = torch.zeros_like(squared_return)
        norm_window = torch.stack([norm_ma_price, norm_squared_return], dim=-1)
        return norm_window, (ma_price_min, ma_price_max, squared_return_min, squared_return_max)

    def get_common_dates(self):
        return list(self.common_index)

    def get_pair_codes(self):
        return list(self.pairs)