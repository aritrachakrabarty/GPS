import matplotlib.pyplot as plt
import numpy as np

from init import *
from gps.source.atm_loss import atm_structure_photevap as atm_structure_photevap_org, xdot_photevap as xdot_photevap_org
# from gps.source.evapmass import solve_structure_org
sys.path.append('../../EvapMass-master')
sys.path.append('../../')
from planet_structure import solve_structure

args_for_func = ('mp', 'X', 'Teq')
def calc_radii_using_evapmass(mc, X, Teq, Tkh_Myr=100, Xiron=0.3, Xice=0):
    mc = np.atleast_1d(mc)
    mc, X, Teq, Xiron, Xice = np.broadcast_arrays(mc, X, Teq, Xiron, Xice)
    Tkh_Myr_new = Tkh_Myr  # if age < 100 else age
    rp = np.zeros_like(mc)
    for i in range(len(mc)):
        rp[i] = solve_structure(X[i], Teq[i], mc[i], Tkh_Myr_new, Xiron[i], Xice[i])[2][0] / RE
    return rp



N = 2000
np.random.seed(1)
m = np.random.lognormal(np.log(3.72), 0.44, N)
np.random.seed(2)
xo = np.random.lognormal(np.log(0.02), 0.71, N)
np.random.seed(3)
p = broken_power_per_sampler(5.75, 2.31, 0.08, N)
a = per2sma(p)
T = 279/a**0.5
rc = m**(1/3.7)
rpo = rad_lopezfortney_analytic(m, rc, xo, T, age=100)
rpo1 = calc_radii_using_evapmass(m, xo, T)
print(pd.DataFrame({'m': m, 'xo': xo, 'p': p, 'rc': rc}))
t = np.geomspace(100, 5000, 100)


rp, x = atm_structure_photevap(m, xo, T, a, rc=rc, eta=0.1, tstart=100, tend=1000, tnum=200, rp_calc_meth='LF14', )
rp1, x1 = atm_structure_photevap_org(m, rc, rpo, xo, T, a, eta=0.1, rp_calc_meth=2, age=100, age_update=True, Tkh_update=True, Tkh_Myr=100, tnum=200, progress=True)
rp2, x1 = atm_structure_photevap(m, xo, T, a, rc=rc, t=t, tstart=50, tend=5000, tnum=50, rp_calc_meth=calc_radii_using_evapmass, args=args_for_func, progress=True)

bins = fulton_redges(0.9, 6)
# plt.hist(rpo, bins, histtype='step')
plt.hist(rp, bins, histtype='step')
plt.hist(rp1, bins, histtype='step')
plt.hist(rp2, bins, histtype='step')

plt.show()
