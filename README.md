# Baryo

## Anisotropic Cosmology and DESI ADPD Analysis

This repository contains the numerical pipeline used to study homogeneous but anisotropic cosmological models using the DESI Angular Density Probability Distribution (ADPD) statistic.

The code generates anisotropic mock catalogues, computes the ADPD statistic, evaluates the likelihood against DESI observations, scans the anisotropy parameter space, and produces the final comparison figures.

---

## Repository structure
anisotropic_cosmology.py Background cosmology
DESI_Anisotropic.py Generate anisotropic mock catalogues
DESI_run.py Run ADPD pipeline
likelihood.py χ² likelihood
scan_omk.py Scan Ω_k
fit_sigma.py Optimise σ₀ and E₂₀
optimize.py Optimisation utilities
make_1sigma_band.py Generate 1σ confidence band
plot_comparison.py Produce final comparison figures
DESI_ADPD_Likelihood.py Cobaya likelihood
desi_adpd.yaml Cobaya configuration

covariance.npy
inverse_covariance.npy
mean_vector.npy Fixed covariance products

data/ Input DESI catalogues
Comparison/ Model ADPD curves
scan_curves/ χ² scan outputs
chains/ Cobaya chains

---

## Required external data

The DESI DR1 clustering catalogues are **not included** because of their size.

Download them from

https://data.desi.lbl.gov/public/dr1/

Required files include, for example,

- BGS_BRIGHT_NGC_clustering.dat.fits
- BGS_BRIGHT_SGC_clustering.dat.fits

Place all downloaded FITS catalogues inside data/.

---

## Installation

Python ≥3.11

Required packages
numpy
scipy
matplotlib
astropy
healpy
cobaya

---

## Typical workflow

1. Download DESI catalogues into `data/`
2. Generate anisotropic mock catalogues
python DESI_Anisotropic.py
3. Compute ADPD curves
python DESI_run.py
4. Scan Ω_k
python scan_omk.py
5. Generate the 1σ confidence region
python make_1sigma_band.py
6. Produce the final comparison figures
python plot_comparison.py

---

## Notes

The covariance matrix (`covariance.npy`) is fixed and is computed from FLAMINGO mock catalogues following the methodology adopted in the reference ADPD analysis.

The anisotropic model changes only the theoretical prediction; the covariance matrix is assumed unchanged.

---

## Citation

If you use this code, please cite

- DESI Collaboration DR1
- The FLAMINGO simulation papers
- Our accompanying anisotropic cosmology publication (to appear)


