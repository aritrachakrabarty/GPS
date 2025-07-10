import os.path
import numpy as np
from scipy.interpolate import interp1d
from .constants import *
from .utils import datapath
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interpn

mr_file = {
    "core+env": {True: r'..\data\massradius_zeng\Earthlike{0}h{1}K1mbar.txt', False: r'..\data\massradius_zeng\halfh2o{0}h{1}K1mbar.txt'},
    "core": {True: r'..\data\massradius_zeng\EarthlikeRocky.txt', False: r'..\data\massradius_zeng\50percentH2O_{1}K_1mbar.txt'}
}
Teqgrid = {True: np.array([300, 500, 700, 1000, 2000]), False: np.array([300, 500, 700, 1000])}

f_mass_radius = lambda ifr: (1 + 0.55 * ifr - 0.14 * ifr ** 2)

def icefrac_from_fmr(fmr):
    a, b, c = -0.14, 0.55, 1 - fmr
    if b**2 >= 4 * a * c:
        det = (b**2 - 4*a*c)**0.5
        return (det - b) / 2 / a

X_zeng = np.array([0.001, 0.003, 0.01, 0.02, 0.05])

def rad_from_zeng_grid(m, Teq=None, X=None, filetype='core', rocky=True):
    if np.ndim(m) == 0 and np.ndim(Teq) == 0 and np.ndim(X) == 0:
        if X in X_zeng:
            xid = str(X)[3:] if X else None
            if X == 0:
                filetype = 'core'
            else:
                filetype = 'core+env'
            if (filetype=='core' and rocky) or Teq in Teqgrid[rocky]:
                mz, rz = np.loadtxt(mr_file[filetype][rocky].format(xid, int(Teq)), unpack=True)
                return interp1d(mz, rz, fill_value='extrapolate')(m)
            Tgs = Teqgrid[rocky][np.argsort(abs(Teqgrid[rocky]-Teq))[:2]]
            rz = []
            for Tg in Tgs:
                _mz, _rz = np.loadtxt(mr_file[filetype][rocky].format(xid, Tg), unpack=True)
                rz.append(interp1d(_mz, _rz, fill_value='extrapolate')(m))
            return interp1d(Tgs, rz, fill_value='extrapolate')(Teq)
        Xgs = X_zeng[np.argsort(abs(X_zeng-X))[:2]]
        rgs = []
        for Xg in Xgs:
            rgs.append(rad_from_zeng_grid(m, Teq, Xg, filetype, rocky))
        return interp1d(Xgs, rgs, fill_value='extrapolate')(X)
    r = np.array([])
    for i in range(max(np.shape(m), np.shape(Teq), np.shape(X), np.shape(filetype), np.shape(rocky))[0]):
        try:
            r = np.append(r, rad_from_zeng_grid(m[i] if np.ndim(m) != 0 else m, Teq[i] if np.ndim(Teq) != 0 else Teq, X[i] if np.ndim(X) != 0 else X,
                                                filetype[i] if np.ndim(filetype)!=0 else filetype, rocky[i] if np.ndim(rocky)!=0 else rocky))
        except:
            print(np.shape(m), np.shape(Teq), np.shape(X), np.shape(filetype), np.shape(rocky))
            print(m, np.shape(m))
            raise
    return r


def rad_lopezfortney_analytic(mp, rc, X, Teq, age):
    renv = 2.06 * mp**-0.21 * (X/0.05)**0.59 * (Teq/279)**(0.044*4) * (age/5000)**-0.18
    g = G * mp * ME / (rc+renv)**2 / RE**2
    ratm = 9*kb*Teq/g/2.3/amu/RE
    return rc + renv + ratm


def total_rad_generic(meth, mp=None, X=None, Teq=None, rc=None, wf=None, **kwargs):
    if meth == 'Z19-grid':
        wf_cutoff = kwargs.pop('wf_cutoff', 0.3)
        return rad_from_zeng_grid(mp, Teq, X, filetype='core+env', rocky=wf < wf_cutoff)
    elif meth == 'LF14':
        return rad_lopezfortney_analytic(mp, rc, X, Teq, kwargs.pop('age', 100))
    elif callable(meth):
        args = kwargs.pop('args', ())
        locals().update(**kwargs)
        args = list(args)
        for i in range(len(args)):
            args[i] = locals()[args[i]]
        return meth(*args)
