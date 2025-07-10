import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from .constants import *

## opacity from form kappa = kappa0*P^alpha*T^beta
alpha = 0.68 # pressure dependence of opacity
beta = 0.45 # temperature dependence of opacity
kappa0 = 10**(-7.32) # opacity constant

mu = 2.35 * mh # solar metallicity gas
gamma = 5./3. # ratio of specific rho_earth_cgs

grad_ab = (gamma-1.)/gamma

def integrand1(x,gamma):

    return x*(1./x-1.)**(1./(gamma-1.))

def integrand2(x,gamma):

    return x**2.*(1./x-1.)**(1./(gamma-1.))

def get_I2_I1(DR_Rc,gamma):

    ratio = np.zeros(np.size(DR_Rc))

    for i in range(np.size(DR_Rc)):

        I2 = quad(integrand2,1./(DR_Rc[i]+1.),1.,args=gamma)
        I1 = quad(integrand1,1./(DR_Rc[i]+1.),1.,args=gamma)

        ratio[i] = I2[0]/I1[0]

    return ratio

def get_I2(DR_Rc,gamma):

    I2 = np.zeros(np.size(DR_Rc))

    for i in range(np.size(DR_Rc)):

        I2[i] = quad(integrand2,1./(DR_Rc[i]+1.),1.,args=gamma)[0]

    return I2


def solve_structure(X,Teq,Mcore,Tkh_Myr,Xiron,Xice,Rcore):

    # this function solves for the structure of the adiabatic envelope of the planet
    # given an envelope mass fraction, i.e. we wish to find the radius of the
    # radiative-convective boundary
    Rcore *= RE

    # use rough analytic formula from Owen & Wu (2017) to guess

    Delta_R_guess = 2.*Rcore * (X/0.027)**(1./1.31) * (Mcore/5.)**(-0.17)

    lg_Delta_Rrcb_guess = np.log10(Delta_R_guess)

    input_args = [X,Teq,Mcore,Rcore,Tkh_Myr]

    # use log Delta_Rrcb_guess to prevent negative solutions

    lg_D_Rrcb_sol = fsolve(Rrcb_function,lg_Delta_Rrcb_guess,args=input_args)[0]

    Rrcb_sol = 10.**lg_D_Rrcb_sol + Rcore

    # now find f-factor to compute planet radius

    rho_rcb = get_rho_rcb(lg_D_Rrcb_sol,X,Mcore,Rcore,Teq,Tkh_Myr)
    pressure_rcb = rho_rcb * kb * Teq / mu

    # now calculate the densities at the photosphere
    Pressure_phot = (2./3. * (G*Mcore*ME/ (Rrcb_sol**2.*kappa0*Teq**beta)))**(1./(1.+alpha))
    rho_phot_calc = (mu/kb) * Pressure_phot / Teq

    # now find f factor
    H = kb * Teq * Rrcb_sol ** 2. / (mu * G * Mcore*ME)

    f = 1. + (H/Rrcb_sol)* np.log(rho_rcb/rho_phot_calc)
    # Rrcb_sol *= 0.9
    # f = (H / Rrcb_sol) * np.log(pressure_rcb*10000000 / 1000)

    Rplanet = f*Rrcb_sol

    return Rrcb_sol, f, Rplanet, Rrcb_sol-Rcore

def Rrcb_function(lg_D_Rrcb,input_args):

    # we combine equation 4 and 13 from Owen & Wu (2017) to produce a function to solve

    # first evaluate the density at the radiative convective boundary

    # unpack-input arguments
    X=input_args[0]
    Teq = input_args[1]
    Mcore = input_args[2]
    Rcore = input_args[3]
    Tkh_Myr = input_args[4]

    # Rcore = mass_to_radius_solid(Mcore,Xiron,Xice) * earth_radius_to_cm

    Rrcb = 10.**lg_D_Rrcb + Rcore

    Delta_R_Rc = 10.**lg_D_Rrcb / Rcore

    rho_core = Mcore * ME / (4./3.*np.pi*Rcore**3.)

    rho_rcb = get_rho_rcb(lg_D_Rrcb,X,Mcore,Rcore,Teq,Tkh_Myr)


    I2 = get_I2(np.array([Delta_R_Rc]),gamma)

    cs2 = kb * Teq / mu

    Xguess=3.*(Rrcb/Rcore)**3.*(rho_rcb/rho_core)*(grad_ab * \
              (G * Mcore * ME)/(cs2 * Rrcb))**(1./(gamma-1.))*I2

    return Xguess - X

def get_rho_rcb(lg_D_Rrcb,X,Mcore,Rcore,Teq,Tkh_Myr):

    # evaluate the density at the radiative convective boundary - equation
    # 13 from owen & wu

    # Rcore = mass_to_radius_solid(Mcore,Xiron,Xice) * earth_radius_to_cm

    Rrcb = 10.**lg_D_Rrcb + Rcore

    Delta_R_Rc = 10.**lg_D_Rrcb / Rcore

    I2_I1 = get_I2_I1(np.array([Delta_R_Rc]),gamma)

    TKh_sec = Tkh_Myr * 1.e6 * year_to_sec

    rho_rcb = (mu / kb) *(I2_I1 * 64. * np.pi * sigma_sb * Teq ** (3. - alpha - beta) \
                          * Rrcb * TKh_sec / (3.*kappa0*Mcore*ME*X))**(1./(1.+alpha))

    return rho_rcb