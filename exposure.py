# ---------------------------------------------------------------
# Functions to compute exposure
# ----------------------------------------------------------------


import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle as pk
from scipy import interpolate
import regionmask
import glob

from settings import *




# function to compute extreme event exposure across a person's lifetime (mf_exposure.m)
def calc_life_exposure(df_life_expectancy_5, df_countries, df_birthyears, d_exposure_peryear):

    # to store exposure per life for every run
    exposure_birthyears = []

    for j, country in enumerate(df_countries['name']):

        # initialise birth years 
        exposure_birthyears_percountry = np.empty(len(df_birthyears))

        for i, birth_year in enumerate(df_birthyears.index): 

            # ugly solution to deal with similar years of life expectancy - to be solved more elegantly. 
            life_expectancy_5 = df_life_expectancy_5.loc[birth_year, country] 
            if np.size(life_expectancy_5) > 1: 
                life_expectancy_5 = life_expectancy_5.iloc[0]

            # define death year based on life expectancy
            death_year = birth_year + np.floor(life_expectancy_5)

            # integrate exposure over full years lived
            exposure_birthyears_percountry[i] = d_exposure_peryear[country].sel(time=slice(birth_year,death_year)).sum().values
            # add exposure during last (partial) year
            exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + d_exposure_peryear[country].sel(time=death_year+1).sum().values * (life_expectancy_5 - np.floor(life_expectancy_5))
        

        if j == 0: # ugly - solve better!
            exposure_birthyears = exposure_birthyears_percountry
        else: 
            exposure_birthyears = np.vstack([exposure_birthyears, exposure_birthyears_percountry])

    df_exposure_perlife = pd.DataFrame(exposure_birthyears.transpose(),index=df_birthyears.index, columns=df_countries.index)
    return df_exposure_perlife



# calculated weighted fieldmean per country mask
def calc_weighted_fldmean_country(da, weights, countries_mask, ind_country):
    da_masked = da.where(countries_mask == ind_country)
    da_weighted_fldmean = da_masked.weighted(weights).mean(dim=("lat", "lon"))
    return da_weighted_fldmean