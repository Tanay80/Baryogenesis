import numpy as np
from scipy.integrate import solve_ivp, romb, cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count

DEFAULT_ZMAX = 0.4
DEFAULT_NZ = 257
DEFAULT_NALPHA = 181

d_w = 0.0

def ang_sep(ls, bs, la, ba):                                                                                                   
    c1 = SkyCoord(l=ls, b=bs, unit="deg", frame="galactic")                                                                     
    c2 = SkyCoord(l=la, b=ba, unit="deg", frame="galactic")                                                                       
    alpha = c1.separation(c2).rad                                                                                                
    return alpha

#Evolution equations
def evol_equ(z, var, alpha, wbar): 
    A, h, sigma, e2, om_im, om_k = var

    A = (1 / (1+z)) * ((np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2)) / (np.abs(1 - e2) ** (1/3))) 
    dAdz = ((A*A) * (np.abs(1-e2) ** (1/3)) * (np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2))) / ((sigma + 1) * e2 * np.sin(alpha) * np.sin(alpha) + (2 - 3 * np.sin(alpha) * np.sin(alpha)) * sigma - 1)
    
    dhdz = dAdz * (-3 * h / 2) * (1 + sigma * sigma - om_k/3 + (1 - sigma * sigma - om_k) * wbar) / A
    dsigmadz = dAdz * ((3 * sigma * wbar + 2 * d_w) * (1 - sigma * sigma - om_k) / 2 + (1 + sigma) * (sigma * sigma - sigma + 1) + (2 - sigma) * (-om_k - (1 + sigma) * (1 + sigma)) / 2 - om_k * (1 + sigma)) / A
    de2dz = dAdz * (6 * sigma * (1 - e2)) / A
    dom_imdz = dAdz * (-om_im * (2 * sigma * d_w + 3 * (wbar - 1) * sigma * sigma + om_k * (1 + 3 * wbar))) / A
    dom_kdz = dAdz * (-2 * om_k * (1 - 2 * sigma + (-3/2)*(1 + sigma * sigma - om_k/3 + (1 - sigma * sigma - om_k) * wbar))) / A
                                                                                                                                                                             
    return [dAdz, dhdz, dsigmadz, de2dz, dom_imdz, dom_kdz]                                                                                

#Cosmological integral in 'z'
def integralFunc(z, A, h, sigma, e2, alpha):

    A = (1 / (1+z)) * ((np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2)) / (np.abs(1 - e2) ** (1/3))) 
    dAdz = ((A*A) * (np.abs(1-e2) ** (1/3)) * (np.abs(1 - e2 * np.sin(alpha) * np.sin(alpha)) ** (1/2))) / ((sigma + 1) * e2 * np.sin(alpha) * np.sin(alpha) + (2 - 3 * np.sin(alpha) * np.sin(alpha)) * sigma - 1)

    G = (1 - e2) ** (1/6)
    H = np.sqrt(1 - e2 * np.sin(alpha) * np.sin(alpha))
    f = G / H 
    
    integral = dAdz * f / (h * A**2) 
    return integral
    
def solve_one_alpha(
    alpha,
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    zmax=0.4,
    nz=257,
):
    """
    Solve the cosmological evolution once for one direction alpha.

    Returns
    -------
    z_grid
    A,h,sigma,e2,om_im,om_k
    """

    z_grid = np.linspace(0.0, zmax, nz)

    y0 = [
        1.0,
        h0,
        sigma0,
        e20,
        om_im0,
        om_k0,
    ]

    sol = solve_ivp(
        evol_equ,
        (0.0, zmax),
        y0,
        t_eval=z_grid,
        args=(alpha, wbar),
        rtol=1e-8,
        atol=1e-10,
    )

    return (
        z_grid,
        sol.y[0],
        sol.y[1],
        sol.y[2],
        sol.y[3],
        sol.y[4],
        sol.y[5],
    )

def build_distance_column(
    alpha,
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    zmax=0.4,
    nz=257,
):
    """
    Compute anisotropic distance for an entire redshift grid
    using only ONE ODE solve.
    """

    z, A, h, sigma, e2, om_im, om_k = solve_one_alpha(
        alpha,
        h0,
        sigma0,
        e20,
        om_im0,
        om_k0,
        wbar,
        zmax=zmax,
        nz=nz,
    )

    integrand = integralFunc(
        z,
        A,
        h,
        sigma,
        e2,
        alpha,
    )

    # Distance = -∫ integrand dz
    distance = -cumulative_trapezoid(
        integrand,
        z,
        initial=0.0,
    )

    distance *= 2997.92458

    return z, distance

def _distance_worker(args):
    """
    Worker function for multiprocessing.
    """

    alpha, h0, sigma0, e20, om_im0, om_k0, wbar = args

    _, distance = build_distance_column(
        alpha,
        h0,
        sigma0,
        e20,
        om_im0,
        om_k0,
        wbar,
        zmax=DEFAULT_ZMAX,
        nz=DEFAULT_NZ,
    )

    return distance

def build_distance_table(
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    zmax=DEFAULT_ZMAX,
    nz=DEFAULT_NZ,
    nalpha=DEFAULT_NALPHA,
):
    """
    Build the complete D(z, alpha) lookup table.
    """

    alpha_grid = np.linspace(0.0, np.pi, nalpha)

    distance_table = np.empty((nz, nalpha), dtype=float)

    z_grid = None

    for j, alpha in enumerate(alpha_grid):

        z, distance = build_distance_column(
            alpha,
            h0,
            sigma0,
            e20,
            om_im0,
            om_k0,
            wbar,
            zmax=zmax,
            nz=nz,
        )

        if z_grid is None:
            z_grid = z

        distance_table[:, j] = distance

        if (j + 1) % 20 == 0 or j == nalpha - 1:
            print(f"{j+1}/{nalpha} alpha values completed")

    return z_grid, alpha_grid, distance_table

def anisotropic_distance(
    z,
    alpha,
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    nsteps=65,
):
    """
    Compute the anisotropic comoving distance
    for a single redshift and a single angle alpha.
    """

    if z == 0:
        return 0.0

    z_grid = np.linspace(0.0, z, nsteps)
    dz = z_grid[1] - z_grid[0]

    y0 = [
        1.0,        # A
        h0,         # h
        sigma0,
        e20,
        om_im0,
        om_k0,
    ]

    sol = solve_ivp(
        evol_equ,
        (0.0, z),
        y0,
        t_eval=z_grid,
        args=(alpha, wbar),
    )

    I = integralFunc(
        z_grid,
        sol.y[0],
        sol.y[1],
        sol.y[2],
        sol.y[3],
        alpha,
    )

    distance_dimensionless = -romb(I, dx=dz)

    # Convert to h^{-1} Mpc
    distance_hmpc = 2997.92458 * distance_dimensionless

    return distance_hmpc
    
def anisotropic_distance_array(
    z_array,
    alpha_array,
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    nsteps=65,
):
    """
    Compute anisotropic distances for arrays of galaxies.
    """

    distances = np.empty_like(z_array, dtype=float)

    for i in range(len(z_array)):
        distances[i] = anisotropic_distance(
            z_array[i],
            alpha_array[i],
            h0,
            sigma0,
            e20,
            om_im0,
            om_k0,
            wbar,
            nsteps=nsteps,
        )

    return distances
    
# -------------------------------------------------------------------------
# Build interpolation table
# -------------------------------------------------------------------------

def build_distance_interpolator(
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    zmax=0.8,
    nz=80,
    nalpha=60,
):

    z_grid = np.linspace(0.0, zmax, nz)

    alpha_grid = np.linspace(0.0, np.pi, nalpha)

    table = np.empty((nz, nalpha))

    for ia, alpha in enumerate(alpha_grid):

        print(f"alpha {ia+1}/{nalpha}")

        for iz, z in enumerate(z_grid):

            table[iz, ia] = anisotropic_distance(
                z,
                alpha,
                h0,
                sigma0,
                e20,
                om_im0,
                om_k0,
                wbar,
            )

    interp = RegularGridInterpolator(
        (z_grid, alpha_grid),
        table,
        bounds_error=False,
        fill_value=None,
    )

    return interp
    
def anisotropic_distance_array_fast(
    z_array,
    alpha_array,
    interpolator,
):

    pts = np.column_stack((z_array, alpha_array))

    return interpolator(pts)
    
def build_distance_interpolator(
    h0,
    sigma0,
    e20,
    om_im0,
    om_k0,
    wbar,
    zmax=DEFAULT_ZMAX,
    nz=DEFAULT_NZ,
    nalpha=DEFAULT_NALPHA,
):

    z_grid, alpha_grid, distance_table = build_distance_table(
        h0,
        sigma0,
        e20,
        om_im0,
        om_k0,
        wbar,
        zmax=zmax,
        nz=nz,
        nalpha=nalpha,
    )

    interpolator = RegularGridInterpolator(
        (z_grid, alpha_grid),
        distance_table,
        bounds_error=False,
        fill_value=None,
    )

    return interpolator
    
def anisotropic_distance_array_fast(
    z_array,
    alpha_array,
    interpolator,
):

    points = np.column_stack((z_array, alpha_array))

    return interpolator(points)
