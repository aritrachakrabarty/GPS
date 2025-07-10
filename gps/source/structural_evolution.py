import numpy as np
import pandas as pd
from .planet_radius_calc_methods import *
from .photoevaporation import calc_eta, atm_structure_photevap
from .utils import defstars


def max_imp_mass_ratio(col, tgasdisp):
    if len(col.get('collision_time', [])) == 0:
        return 0
    mask = col['collision_time'] > tgasdisp
    return np.max(col['mass_j'][mask] / col['mass_i'][mask])

class StrucEvol:

    def __init__(self, df, star=''):
        self.df = pd.DataFrame()
        if df:
            self.load(df)
        if star:
            self.add_star(star)

    def load(self, df):
        if type(df) == str:
            self.df = pd.read_csv(df)
            if 'ims' in self.df:
                self.df['ims'] = self.df['ims'].apply(eval)
            if 'sys' in self.df:
                self.df['sys'] = self.df['sys'].apply(eval)
        else:
            self.df = df

    def add_star(self, star):
        if type(star) == str:
            self.star = defstars[star]
        elif isinstance(star, dict):
            self.star = star

    def add_atmosphere(self, X, *args):
        if X is not callable():
            self.df['x'] = X
        elif not args:
            self.df['x'] = X()
        else:
            if type(args) == str:
                args = [args]
            self.df['x'] = X(*self.df[args].values)

    def calc_core_radii(self, meth='Z19-fit', wf_cutoff=0.3, args=()):
        if meth == 'Z19-fit':
            self.df['rc'] = f_mass_radius(self.df['wf']) * (self.df['m'])**(1/3.7)
        elif meth == 'Z19-grid':
            self.df['rc'] = rad_from_zeng_grid(self.df['m'], self.df['T'], self.df['x'], filetype='core', rocky=self.df['wf'] < wf_cutoff)
        elif callable(meth):
            if args:
                if type(args) == str:
                    args = [args]
                self.df['rc'] = meth(*self.df[args].values)
            else:
                self.df['rc'] = meth()

    def calc_total_radii(self, meth='LF14', **kwargs):
        self.df['rpo'] = total_rad_generic(meth, self.df['m'], self.df['x'], self.df['T'], self.df['rc'], self.df['wf'], **kwargs)
        self.df['rp'] = self.df['rpo'] + 0

    def calc_impact_loss_by_mass_ratio(self, tgasdisp=0, mass_ratio_cutoff=0.1, X_low=1e-4):
        imp_mr_max = self.df['ims'].apply(lambda row: max_imp_mass_ratio(row, tgasdisp))
        self.df['rp'][imp_mr_max >= mass_ratio_cutoff] = self.df['rc'][imp_mr_max >= mass_ratio_cutoff].values
        self.df['x'][imp_mr_max >= mass_ratio_cutoff] = X_low

    def calc_photev_loss_by_energy_ratio(self, phi_cutoff=1, eta=0.1, X_low=1e-4):
        if callable(eta):
            eta = calc_eta(eta, self.df['m'], self.df['rp'])
        PHI = self.df['x']/3.3e-3 * (5.2e45/self.star['Eout']) * 0.1/eta * (self.df['m']/3)**(5/4) * (1000/self.df['T'])**4 * (self.df['rc']/self.df['rp'])**2
        self.df['rp'][PHI < phi_cutoff] = self.df['rc'][PHI < phi_cutoff].values
        self.df['x'][PHI < phi_cutoff] = X_low

    def calc_photevap_loss_by_timescale(self, eta=0.1, tmax=3000, tnum=500, rp_calc_method='LF14', **kwargs):
        if callable(eta):
            eta = calc_eta(eta, self.df['m'], self.df['rp'])
        self.df['rp'], self.df['x'] = atm_structure_photevap(self.df['m'], self.df['x'], self.df['T'], self.df['a'], self.star['ms'], self.df['rc'], self.df['rp'], eta, tmax, tnum, rp_calc_meth=rp_calc_method, **kwargs)




