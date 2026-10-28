"""Read the historical comparison preserved in the cloud revision.

The earlier panel and fit inputs are absent from that revision. These rows are
archived observations, not new estimates or substitutes for the missing inputs.
"""
from pathlib import Path
import pandas as pd


def load_historical_comparison():
    directory = Path(__file__).resolve().parents[1] / 'data' / 'legacy'
    likelihoods = pd.read_csv(directory / 'historical_quadrature_nll.csv')
    fits = pd.read_csv(directory / 'historical_quadrature_fits.csv')
    return likelihoods, fits
