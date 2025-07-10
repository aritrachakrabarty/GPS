import matplotlib.pyplot as plt

from init import *
from gps.source.atm_loss import atm_structure_photevap as atm_structure_photevap_org, xdot_photevap as xdot_photevap_org
sys.path.append('../../EvapMass-master')
sys.path.append('../../')
from planet_structure import solve_structure


N = 2000
np.random.seed(1)
m = np.random.lognormal(np.log(3.72), 0.44, N)
np.random.seed(2)
xo = np.random.lognormal(np.log(0.02), 0.71, N)
np.random.seed(3)
p = broken_power_per_sampler(p0=5.75, k1=2.31, k2=0.08, N=N)
a = per2sma(p)
T = 279/a**0.5
rc = m**(1/3.7)
# print(rc)
# rpo = total_rad_generic('LF14', m, xo, T, rc, age=100)
rpo = rad_lopezfortney_analytic(m, rc, xo, T, age=100)

# xdot = xdot_photevap(0, xo, m*ME, rpo*RE, a*AU, 1, 0.1)
# xdotorg = xdot_photevap_org(0, xo, m*ME, rpo*RE, a*AU, 1, 0.1)

# print(np.array_equal(xdot, xdotorg))

# rp, x = atm_structure_photevap(m, rc, None, T, xo, a, rp_calc_meth='LF14', age=100, tmax=5000, age_update=True)
# print(xo)
rp, x = atm_structure_photevap(m, xo, T, a, rc=rc, rp=rpo, eta=0.1, rp_calc_meth='LF14', tstart=10, tnum=200)
rp1, x1 = atm_structure_photevap_org(m+0, rc+0, rpo+0, xo+0, T+0, a+0, eta=0.1, rp_calc_meth=2, age=100, age_update=True, Tkh_update=True, Tkh_Myr=100, tnum=200, snapshot=False)
# print(rp, rpo)
bins = fulton_redges(0.9, 6)
# plt.hist(rpo, bins)
# plt.hist(rp, bins, histtype='step')
# plt.hist(rp1, bins, histtype='step')

Rcen = (bins[:-1] * bins[1:]) ** 0.5
fig, ax = plt.subplots(1, 2)
ax[0].hist(rp, bins=bins, weights=np.ones(N) / N, histtype='step', label='OW17')
ax[0].hist(rp1, bins=bins, weights=np.ones(N) / N, histtype='step', label='LF14')
ax[0].set_xlabel(f'Radius ({runit})')
ax[0].set_ylabel('Occurrence per star')
legend_colortxt_nosym(ax[0])

bins = np.geomspace(np.min([xo, x, x1]), np.max([xo, x, x1]), 20)
ax[1].hist(xo, bins=bins, histtype='step', label='Before loss')
ax[1].hist(x, bins=bins, histtype='step', label='After loss: OW17')
ax[1].hist(x1, bins=bins, histtype='step', label='After loss: LF14')
ax[1].set_xlabel(f'Atm mass-frac')
ax[1].set_ylabel('Count')

# print(np.min([xo, x, x1]), np.max([xo, x, x1]))
# print(xo)
# print(x1)
plt.show()