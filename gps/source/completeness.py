import numpy as np
import pandas as pd
import os
from .utils import *
from scipy.stats import gamma


def P_detection_single(kicdf, rp, per, cumdist=(17.56, 1, 0.49)):
    nkic = kicdf['id_kic'].count()
    a = (G * kicdf['m17_smass'].values * MS * ((per * (24 * 3600.)) / (2 * np.pi)) ** 2) ** (1 / 3.)
    Rs = kicdf['gaia2_srad'].values * RS
    durations = (per * 24. / np.pi) * np.arcsin(Rs / a)
    rors = rp * RE / Rs
    x = 1 / np.sqrt(durations)
    cdpp_durs = kicdf['m17_cdpp_fit0'].values + \
                kicdf['m17_cdpp_fit1'].values * x + \
                kicdf['m17_cdpp_fit2'].values * x ** 2

    # Calculate SNR for other stars
    snr = rors ** 2 * (per / kicdf['m17_tobs'].values) ** -0.5 * (1 / (cdpp_durs * 1e-6))
    s = gamma.cdf(snr, *cumdist)
    s[np.isnan(s)] = 0.0
    return np.sum(s) / nkic
    

def P_detection(rp, per, cumdist=(17.56, 1, 0.49)):
    kicdf = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fulton+2017', 'kic_filtered.csv'))
    if isinstance(rp, (int, float)):
        return P_detection_single(kicdf, rp, per, cumdist)
    pdet = np.zeros(len(rp))
    for i in range(len(rp)):
        pdet[i] = P_detection_single(kicdf, rp[i], per[i], cumdist)
    return pdet

def P_transit(rs, a, bcut=0.7):
    return bcut*rs*RS/a/AU

def P_completeness(df, rs, bcut=0.7, cumdist=(17.56, 1, 0.49)):
    if type(rs) == str:
        rs = df[rs].values
    return P_transit(rs, df['a'].values, bcut) * P_detection(df['rp'].values, df['p'].values, cumdist)

def get_nkic():
    kicdf = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fulton+2017', 'kic_filtered.csv'))
    return kicdf['id_kic'].count()