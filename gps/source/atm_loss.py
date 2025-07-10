import numpy as np

from .planet_radius_calc_methods import *
from .constants import year_to_sec

# Lxuv_decay = lambda t, ms: 1.2e30*ms*(t/100)**-1.5 * Myr2sec
# Lxuv = lambda t, ms:  1.2e30*Myr2sec*ms if t<100 else Lxuv_decay(t, ms)

def Lxuv(t, ms, tsat=100, a0=0.5):
    ''' Follows Owen (2017) '''
    Lsat = 1.2e30*year_to_sec*1e6*ms
    return Lsat if t < tsat else Lsat * (t/tsat)**(-1-a0)


def calc_eta(func, Mp, Rp):
    vesc = (2 * G * Mp / Rp) ** 0.5
    return func(vesc)


def xdot_photevap(t, X, Mp, Rp, a, ms, eta=0.1, **kwargs):
    if np.all(X==0):
        return np.zeros(max([np.shape(v) for v in locals().values()]))
    # print(Lxuv(t, ms))
    if callable(eta):
        eta = calc_eta(eta, Mp, Rp)
    return -eta*np.pi*Rp**3*Lxuv(t, ms, **kwargs)/(4*np.pi*a**2*G*Mp**2)


def atm_structure_photevap(mc, rc, rp, X, Teq, a, ms=1, eta=0.1, tmax=1000, tnum=200, t=None, rp_calc_meth=0, snapshot=False, progress=False, **kwargs):
    if np.all(X == 0):
        return rp, X
    mc = np.atleast_1d(mc)
    rc = np.atleast_1d(rc)
    rp = np.atleast_1d(rp)
    X = np.atleast_1d(X)
    Teq = np.atleast_1d(Teq)
    a = np.atleast_1d(a)
    Xf = X.copy()
    rpf = rp.copy()
    ms = ms * np.ones(len(mc))
    mask = Xf > 0.0001
    # print('mask0', sum(mask))
    # rpf[~mask] = rc[~mask]
    if t is None:
        t = np.linspace(0, tmax, tnum)
    dt = np.diff(t)
    # print(len(Xf[Xf < 0.001]) / len(Xf))
    if snapshot:
        rpss = []
        Xss = []
    tsat = kwargs.pop('tsat', 100)
    a0 = kwargs.pop('a0', 0.5)
    # print('X0 org', X)
    # print('org')
    for i in range(0, len(t) - 1):
        if progress and (i % 10 == 0 or i == len(t) - 2):
            print(f'Runs: {i}/{len(t) - 1}', end=' ')
        # print('ms', ms)
        # df = pd.DataFrame({'x': X, 'mc': mc, 'rp': rp, 'a': a, 'rc': rc, 'Teq': Teq, 'per': sma2per(a), 'xdot': xdot_photevap(t[i], X, mc*ME, rp*RE, a*AU, ms, eta)})
        # Xfold = Xf[mask].copy()
        Xf[mask] = Xf[mask] + xdot_photevap(t[i], Xf[mask], mc[mask]*ME, rpf[mask]*RE, a[mask]*AU, ms[mask], eta, tsat=tsat, a0=a0) * dt[i]
        # print('before')
        # print(t[i], sum(mask))
        # print(np.c_[mc[mask], a[mask], Xf[mask], rpf[mask]])
        mask = Xf > 0.0001
        # rpf[~mask] = rc[~mask]
        # rpf[Xf<0.0001] = rpf[Xf]
        if sum(mask) == 0:
            break
        mask1 = mask
        if rp_calc_meth == 0:
            rocky = kwargs.pop('rocky', True)
            if hasattr(rocky, '__len__'):
                rocky = rocky[mask1]
            rpf[mask1] = rad_from_zeng_grid(mc[mask1], Teq[mask1], Xf[mask1], 'core+env', rocky=rocky)
        elif rp_calc_meth == 1:
            Tkh_Myr = kwargs.get('Tkh_Myr', 100)
            Tfac = kwargs.get('Tfac', 1)
            if kwargs.get('Tkh_update', False):
                Tkh_Myr = 100 if t[i] <= 100 else t[i]
                if progress and (i % 10 == 0 or i == len(t) - 2):
                    print(Tkh_Myr, end='\r')
            rpf[mask1] = rad_using_evapmass(mc[mask1], rc[mask1], Teq[mask1], Xf[mask1], Tkh_Myr=Tkh_Myr, Tfac=Tfac)
        elif rp_calc_meth == 2:
            # print(kwargs['age'])
            age = kwargs.get('age', 5)
            age0 = age
            age_update = kwargs.get('age_update', True)
            if age_update:
                age = age0 + t[i+1]
                if progress and (i % 10 == 0 or i == len(t) - 2):
                    print(age, end='\r')
                # if i == len(t) - 2:
                #     print(mc[mask1], rc[mask1], Xf[mask1], Teq[mask1], age)
            # print(np.c_[mc[mask1], rc[mask1], Xf[mask1], Teq[mask1]])
            rpf[mask1] = rad_lopezfortney_analytic(mc[mask1], rc[mask1], Xf[mask1], Teq[mask1], age)
        elif rp_calc_meth == 3:
            ifrac = kwargs.get('ifrac', 0)
            if np.ndim(ifrac) == 0:
                ifrac = np.ones(len(mc)) * ifrac
            rc_calc_meth = kwargs.get('rc_calc_meth', 0)
            age = kwargs.get('age', 5)
            age0 = age
            age_update = kwargs.get('age_update', True)
            if age_update:
                age = age0 + t[i+1]
            rpf[mask1] = rad_lopezfortney_analytic_ifcorr(mc[mask1], ifrac[mask1], Xf[mask1], Teq[mask1], age, rc_calc_meth=rc_calc_meth)
        else:
            raise NotImplemented('rp_calc_meth > 3')
        if snapshot:
            rpss.append(rpf.copy())
            Xss.append(Xf.copy())
            tss = t + kwargs.get('age', 0)
    if progress:
        print()
    if not snapshot:
        rpf[~mask] = rc[~mask]
        Xf[~mask] = 0.0001
        # print(len(Xf[Xf<0.001])/len(Xf))
        return rpf, Xf
    for i in range(len(rpss)):
        mask = Xss[i] > 0.0001
        rpss[i][~mask] = rc[~mask]
        Xss[i][~mask] = 0.0001
    return tss, rpss, Xss

