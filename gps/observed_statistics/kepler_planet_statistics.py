import numpy as np
import pandas as pd
from gps.source import datapath, P_completeness, get_nkic
from gps.helper import fulton_redges
from .obsdata_manager import ObsStats


def load_kepler_planet_candidates_from_cksgaia(file=datapath('fulton+2017', 'cksgaia_planets_filtered.csv'), keepall=False, add_columns={}, ):
    df = pd.read_csv(file)
    # print([col for col in df.columns if 'iso_' in col])
    rename = {'koi_period': 'p', 'iso_prad': 'rp', 'iso_srad': 'rs', 'koi_sma': 'a', 'koi_teq': 'T', 'iso_smass': 'ms', 'iso_steff': 'Ts'}
    orgcols = list(rename)
    newcols = []
    for key in orgcols:
        rename[key + '_err1'] = rename[key] + 'e+'
        rename[key + '_err2'] = rename[key] + 'e-'
        newcols += [rename[key], rename[key] + 'e+', rename[key] + 'e-']
    if add_columns:
        if type(add_columns) != dict:
            newcols += add_columns
        else:
            rename.update(add_columns)
            newcols += list(add_columns.values())
    df = df.rename(columns=rename)
    for key in filter(lambda x: 'e-' in x, df):
        df[key] = np.abs(df[key])
        col = key[:-1]
        df[col] = (df[col + '+'] + df[col + '-']) / 2
        newcols += [col]
    if not keepall:
        df = df[newcols].reset_index(drop=True)
    return ObsStats(df)


def kepler_radius_histogram_from_cksgaia(df=None, bins=None, rplim=(1, 6), nstars=get_nkic(), p_edges=None):
    if df is None:
        df = load_kepler_planet_candidates_from_cksgaia().df
    if bins is None:
        bins = fulton_redges(*rplim)
    if p_edges is None:
        p_edges = np.geomspace(0.1, 100, 40)
    df = df[(df['p'] > min(p_edges)) & (df['p'] < max(p_edges))].reset_index(drop=True)
    if len(bins) > 0:
        df = df[(df['rp'] > min(bins)) & (df['rp'] < max(bins))].reset_index(drop=True)
    weights = 1 / P_completeness(df)
    wdetections, _, _ = np.histogram2d(df['p'], df['rp'], bins=[p_edges, bins], weights=weights)
    whist = np.sum(wdetections, axis=0)
    whistn = whist / nstars
    return whistn, bins


def kepler_radius_histogram_from_cksgaia_with_errorbar(df=None, bins=None, rplim=(1, 6), nstars=get_nkic(), p_lim=(0.1, 100), nsim=100):
    if df is None:
        df = load_kepler_planet_candidates_from_cksgaia().df
    if bins is None:
        bins = fulton_redges(*rplim)
    df = df[(df['p'] > min(p_lim)) & (df['p'] < max(p_lim))].reset_index(drop=True)
    if hasattr(bins, '__len__'):
        df = df[(df['rp'] > min(bins)) & (df['rp'] < max(bins))].reset_index(drop=True)
    weights = 1 / P_completeness(df)
    whists = []
    for i in range(nsim):
        rp = np.random.uniform(df['rp'] - 2*df['rpe'], df['rp'] + 2*df['rpe'])
        whists.append(np.histogram(rp, bins=bins, weights=weights)[0] / nstars)
    whists_pctls = np.percentile(whists, q=[50, 16, 84], axis=0)
    return whists_pctls[0], whists_pctls[0] - whists_pctls[1], whists_pctls[2] - whists_pctls[0], bins


def kepler_period_histogram_from_cksgaia(df=None, bins=None, plim=(0.1, 100, 20), nstars=get_nkic(), rp_edges=None):
    if df is None:
        df = load_kepler_planet_candidates_from_cksgaia().df
    if bins is None:
        bins = np.geomspace(*plim)
    if rp_edges is None:
        rp_edges = np.logspace(np.log10(0.5), np.log10(6), 40)
    df = df[(df['rp'] > min(rp_edges)) & (df['rp'] < max(rp_edges))].reset_index(drop=True)
    if len(bins) > 0:
        df = df[(df['p'] > min(bins)) & (df['p'] < max(bins))].reset_index(drop=True)
    weights = 1 / P_completeness(df)
    wdetections, _, _ = np.histogram2d(df['p'], df['rp'], bins=[bins, rp_edges], weights=weights)
    whist = np.sum(wdetections, axis=1)
    whistn = whist / nstars
    return whistn, bins


def corr_with_kepler_rphist(rp, fs=1, **kwargs):
    okep, bins = kepler_radius_histogram_from_cksgaia(**kwargs)
    o, _ = np.histogram(rp, bins=bins, weights=np.ones(len(rp))*fs/len(rp))
    return np.corrcoef(okep, o)[0, 1]