import os.path
import numpy as np
from gps.source import *
import scipy.stats as st
from matplotlib.ticker import ScalarFormatter

path2data = lambda *args: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', *args)


def evolve_genesis_data(models, star=None, X=None, Xparametric=True, imp_mx_cutoff=None, tgasdisp=None, photevap_eta=None, rpcalc='LF14', rpmax=5, start_age=200, end_age=5000):
    sev = StrucEvol(path2data('genesis_all_wfs.pk'), star=star)
    if isinstance(models, (list, tuple)):
        sev.df = sev.df[sev.df['sys'].apply(lambda x: x[0] in models)].reset_index(drop=True)
    elif callable(models):
        sev.df = sev.df[sev.df['sys'].apply(lambda x: models(x[0]))].reset_index(drop=True)
    if X:
        if not Xparametric:
            sev.add_atmosphere(X)
        else:
            sev.add_atmosphere(X, 'm', 'T')
    sev.calc_core_radii()
    if imp_mx_cutoff is not None:
        sev.calc_total_radii(rpcalc, wf_cutoff=0.3, age=end_age)
        sev.calc_impact_loss_by_mass_ratio(tgasdisp, imp_mx_cutoff)
    if photevap_eta is not None:
        sev.calc_photevap_loss_by_timescale(photevap_eta, tmax=end_age, tnum=200, rp_calc_method=rpcalc, age=start_age, age_update=True, wf_cutoff=0.3)
    if rpmax:
        sev.df = sev.df[sev.df['rp'] < rpmax]
    return sev

def bulk_comp_stat_model(df, kind='', wflim=0.4, xlim=2e-4, mlim=None, atmkey='x'):
    if type(kind) == str:
        mask = df['m'] > 0
        if 'b' in kind or 'g' in kind:
            maskbg = (df[atmkey] < xlim)  # for 'b' i.e. bare this is the mask
            if mlim:
                maskbg &= df['m'] < mlim
            if 'g' in kind:  # for 'g' i.e. gas-rich the mask is then inversed
                maskbg = ~maskbg
            mask &= maskbg
        if 'r' in kind:
            mask &= (df['wf'] < wflim)
        elif 'w' in kind:
            mask &= (df['wf'] >= wflim)
        return mask, sum(mask)
    masks = []
    Nkinds = []
    for k in kind:
        res = bulk_comp_stat_model(df, kind=k, wflim=wflim, xlim=xlim, mlim=mlim)
        masks.append(res[0])
        Nkinds.append(res[1])
    return masks, Nkinds

class period_F_KDE:
    
    def __init__(self, df, wflim=(0.4, 0.7), mlim=10):
        self.df = df
        self.wflim = wflim
        self.mlim = mlim
        self.comps = 'br', 'bw', 'gr', 'gw'
        self.atmkinds = 'b', 'g'
        self.corecomps = 'r', 'w'
        self.calc_kde()
        
    def calc_kde(self):
        masks, Ns = bulk_comp_stat_model(self.df, self.comps, wflim=self.wflim, mlim=self.mlim)
        self.occ_prob = {self.comps[i]: Ns[i] / sum(Ns) for i in range(len(self.comps))}
        self.kernel = {self.comps[i]: st.gaussian_kde(np.vstack([np.log10(self.df[masks[i]]['p']), self.df[masks[i]]['fp']])) for i in range(len(self.comps))}
        
    def dist_pdf(self, kind, p, fp):
        assert kind in self.comps
        return self.kernel[kind]((np.log10(p), fp))

    def kde(self, kind, p, fp):
        return self.dist_pdf(kind, p, fp)

    def occ_pdf(self, kind, p, fp):
        if kind in self.comps:
            return self.kde(kind, p, fp) * self.occ_prob[kind]
        if kind in ('r', 'w'):
            return sum(self.kde(atmkind+kind, p, fp) * self.occ_prob[atmkind+kind] for atmkind in self.atmkinds)
        if kind in ('b', 'g'):
            return sum(self.kde(kind+corecomp, p, fp) * self.occ_prob[kind+corecomp] for corecomp in self.corecomps)
        if kind == 'any':
            # print((self.kde('br', p, fp) * self.occ_prob['br']) / sum([self.kde(comp, p, fp) * self.occ_prob[comp] for comp in self.comps]))
            return sum(self.kde(comp, p, fp) * self.occ_prob[comp] for comp in self.comps)
        if kind == 'all':
            return 0

    def comp_pdf_given_detected(self, kind, p, fp, detected='any'):
        if kind == detected or kind == 'any':
            return 1
        if all(k in kind+detected for k in self.atmkinds):
            return 0
        if all(k in kind+detected for k in self.corecomps):
            return 0
        if len(kind) < len(detected) and detected != 'any':
            return 1
        if len(kind) == len(detected) == 1:
            if kind in self.atmkinds:
                kind = kind + detected
            else:
                kind = detected + kind
        return self.occ_pdf(kind, p, fp) / self.occ_pdf(detected, p, fp)

def get_wwprob_using_kde(df, startype, wflim=0.4, mlim=20):
    masks = bulk_comp_stat_model(df, ('br', 'bw'), wflim=wflim, mlim=mlim)[0]
    Pb = [sum(masks[0]) / (sum(masks[0]) + sum(masks[1])), 0]
    Pb[1] = 1 - Pb[0]
    if startype == 'M':
        pmin, pmax = np.log10(0.5), np.log10(50)
    if startype == 'G':
        pmin, pmax = np.log10(0.35), np.log10(100)
    for m, mask in enumerate(masks):
        if Pb[m] == 0 or sum(mask) < 4:
            continue
        if (df[mask]['fp'] == df[mask]['fp'].iloc[0]).all():
            kern = st.gaussian_kde(np.log10(df[mask]['p']))
            Pb[m] *= kern.integrate_box_1d(pmin, pmax)
        else:
            try:
                kern = st.gaussian_kde(np.vstack([np.log10(df[mask]['p']), df[mask]['fp']]))
            except:
                # print(df)
                print(np.vstack([np.log10(df[mask]['p']), df[mask]['fp']]))
                raise
            Pb[m] *= kern.integrate_box([pmin, 0.9], [pmax, 1.35])
    return Pb[1] / sum(Pb) * 100

def show_ww_prob2d(df, pgrid, y='', ygrid=[], wflim=0.4, ax=None, cax=False, gridlines=True, plottype='colormesh', 
                   label=True, **kwargs):
    if type(pgrid) == int:
        pgrid = np.logspace(*np.log10(extreme(df['p'].values)), pgrid)
    p = (pgrid[1:] * pgrid[:-1]) ** 0.5
    if type(ygrid) == int:
        ygrid = np.linspace(*np.log10(extreme(df[y].values)), ygrid)
    elif isinstance(ygrid, (tuple, list)):
        ygrid = getattr(np, ygrid[0] + 'space')(*np.log10(extreme(df[y].values)), ygrid[1])
    Y = (ygrid[1:] + ygrid[:-1]) / 2
    if gridlines:
        mark_pgrid = kwargs.pop('mark_pgrid', None)
        mark_ygrid = kwargs.pop('mark_ygrid', None)
    kernel = {}
    Npl = {}
    for comp in ('br', 'bw', 'gr', 'gw'):
        mask, N = bulk_comp_stat_model(df, comp, wflim=wflim, mlim=20)
        kernel[comp] = st.gaussian_kde(np.vstack([np.log10(df[mask]['p']), df[mask]['fp']]))
        Npl[comp] = N
    wwprob = np.ones((len(Y), len(p))) * np.nan
    for i in range(len(pgrid)-1):
        for j in range(len(ygrid)-1):
            rect = [np.log10(pgrid[i]), ygrid[j]], [np.log10(pgrid[i+1]), ygrid[j+1]]
            wp = kernel['bw'].integrate_box(*rect) * Npl['bw'] / sum(Npl.values()) + kernel['gw'].integrate_box(*rect) * Npl['gw'] / sum(Npl.values())
            rp = kernel['br'].integrate_box(*rect) * Npl['br'] / sum(Npl.values()) + kernel['gr'].integrate_box(*rect) * Npl['gr'] / sum(Npl.values())
            wwprob[j, i] = wp / (wp + rp)
    if ax:
        ylabel = kwargs.pop('ylabel', '')
        if plottype == 'contour':
            contour = ax.contourf(p, Y, wwprob, **kwargs)
        elif plottype == 'colormesh':
            contour = ax.pcolormesh(pgrid, ygrid, wwprob, vmax=1, **kwargs)
        if label:
            ax.set_xlabel('Period (days)')
            ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        if gridlines:
            if mark_pgrid is None:
                pgrid_plot = pgrid
            elif isinstance(mark_pgrid, slice):
                pgrid_plot = pgrid[mark_pgrid]
            elif hasattr('mark_pgrid', '__len__'):
                pgrid_plot = mark_pgrid
            if mark_ygrid is None:
                ygrid_plot = ygrid
            elif isinstance(mark_ygrid, slice):
                ygrid_plot = ygrid[mark_ygrid]
            elif hasattr('mark_ygrid', '__len__'):
                ygrid_plot = mark_ygrid
            for pg in pgrid_plot:
                ax.axvline(pg, alpha=0.1, c='k')
            for yg in ygrid_plot:
                ax.axhline(yg, alpha=0.1, c='k')
        if cax:
            cbar = plt.colorbar(contour, ax=ax)
            if label:
                cbar.ax.set_ylabel('Probability of Water world', size=20)
            cbar.ax.tick_params(labelsize=14)
            return {'wwprobgrid': wwprob, 'pgrid': pgrid, 'ygrid': ygrid}, contour, cbar
        return {'wwprobgrid': wwprob, 'pgrid': pgrid, 'ygrid': ygrid}, contour
    return {'wwprobgrid': wwprob, 'pgrid': pgrid, 'ygrid': ygrid}

def completeness_filter(df, rs='rs', plim=0.5):
    weights = P_completeness(df, rs)
    kernel = st.gaussian_kde(np.vstack([df['p'], df['rp']]), weights=weights)
    pdf = kernel(np.vstack([df['p'], df['rp']]))
    pdfsrtd = np.sort(pdf)[::-1]
    acc = np.in1d(pdf, pdfsrtd[np.cumsum(pdfsrtd) > plim])
    return df.iloc[acc,:]

def quantile_to_level(pdf, quantile):
    """Return pdf levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(pdf)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels