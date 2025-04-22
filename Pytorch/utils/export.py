import pandas as pd

# === Helper: Dump Entire Dataset as DataFrame ===
def export_dataset_to_dataframe(dataset):
    data = {}
    with dataset.app.app_context():
        group_ids = dataset.group_ids
        common_dates = dataset.common_dates
        for group_id in group_ids:
            group_series = dataset.group_data[group_id]  # dict: {date: value}
            data[group_id] = [group_series.get(date, None) for date in common_dates]

    df = pd.DataFrame(data, index=dataset.common_dates)
    df.index.name = "date"
    df.columns = [f"group_{gid}" for gid in df.columns]
    return df

#r = redis.Redis(host="localhost", port=6379, decode_responses=True)