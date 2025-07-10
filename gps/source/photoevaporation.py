import numpy as np
from .constants import *
from .planet_radius_calc_methods import total_rad_generic

Myr2sec = 86400 * 365.25 * 1e6
Lxuv_decay = lambda t, ms: 1.2e30*ms*(t/100)**-1.5 * Myr2sec
Lxuv = lambda t, ms:  1.2e30*Myr2sec*ms if t<100 else Lxuv_decay(t, ms)
Lsun = 1.2e30*10**3.5


def calc_eta(func, m, rp):
    vesc = (2 * G * m * ME / rp / RE) ** 0.5
    return func(vesc)


def xdot_photevap(t, X, Mp, Rp, a, ms, eta=0.1):
    if np.all(X==0):
        return np.zeros(max([np.shape(v) for v in locals().values()]))
    return -eta*np.pi*Rp**3*Lxuv(t, ms)/(4*np.pi*a**2*G*Mp**2)


def atm_structure_photevap(mc, X, Teq, a, ms=1, rc=None, rp=None, eta=0.1, t=None, rp_calc_meth=0, Xmin=0.00001, snapshot=False, **kwargs):
    # tstart, tend, and t are in Myr

    if np.all(X < Xmin):
        return rp, X
    mc = np.atleast_1d(mc)
    wf = kwargs.pop('wf', None)
    mc, rc, Teq, X, a, ms, wf, eta = np.broadcast_arrays(mc, rc, Teq, X, a, ms, wf, eta)

    if t is None:
        t = np.linspace(10, 5000, 100)  # t[0] is not zero as some functions may collapse at t=0
    dt = np.diff(t)

    if rp is not None:
        rp = np.broadcast_to(rp, mc.shape)
    else:
        rp = total_rad_generic(rp_calc_meth, mc, X, Teq, rc, wf, age=t[0], **kwargs)

    Xf = X.copy()
    rpf = rp.copy()
    mask = Xf > Xmin
    rpss = []
    Xss = []

    progress = kwargs.pop('progress', False)
    for i in range(0, len(t) - 1):
        if progress and (i==0 or i % 10 == 9 or i == len(t) - 2):
            print(f'time-step: {i+1}/{len(t)-1}', end='\r')
        Xf[mask] = Xf[mask] + xdot_photevap(t[i], Xf[mask], mc[mask]*ME, rpf[mask]*RE, a[mask]*AU, ms[mask], eta[mask]) * dt[i]
        mask = Xf > Xmin
        if sum(mask) == 0:
            break
        rpf[mask] = total_rad_generic(rp_calc_meth, mc[mask], Xf[mask], Teq[mask], rc[mask], wf[mask], age=t[i+1], **kwargs)
        if snapshot:
            rpss.append(rpf.copy())
            Xss.append(Xf.copy())

    if progress:
        print()

    if not snapshot:
        rpf[~mask] = rc[~mask]
        Xf[~mask] = Xmin
        return rpf, Xf

    for i in range(len(rpss)):
        mask = Xss[i] > Xmin
        rpss[i][~mask] = rc[~mask]
        Xss[i][~mask] = Xmin
    return rpss, Xss