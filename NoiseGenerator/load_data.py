import pandas as pd


def load_data(csvfile):
    df = pd.read_csv(csvfile, dtype=str, names=['x', 'y', 'bits'], header=None)
    xs = df['x'].astype(float).values
    ys = df['y'].astype(float).values
    bits = df['bits'].apply(lambda x: format(int(x), '04b')).values
    return xs, ys, bits
