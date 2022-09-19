#!/usr/bin/env python3
# ---------------------------------------------------------------
# Main script to postprocess and visualise lifetime exposure data
#
# Python translation of the MATLAB scripts of Thiery et al. (2021)
# https://github.com/VUB-HYDR/2021_Thiery_etal_Science
# ----------------------------------------------------------------

#%% ----------------------------------------------------------------
# Summary and notes

# to save the enironment used (with your path to the env directory): 
# conda env export -p C:\Users\ivand\anaconda3\envs\exposure_env > exposure_env.yml


# Data types are defined in the variable names starting with:  
#     df_     : DataFrame    (pandas)
#     gdf_    : GeoDataFrame (geopandas)
#     da_     : DataArray    (xarray)
#     d_      : dictionary  
#     sf_     : shapefile
#     ...dir  : directory

# TODO
# - not yet masked for small countries
# - how to handle South-Sudan and Palestina? Now manually filtered out in load
# - south sudan and palestine not dealt with (we only have 176 countries instead of 178 from matlab, these are missing in end mmm values)
# - for import to hydra, the file loading and dealing with lower case letters in hadgem will need to be addressed (way forward is to test line 236 of load_manip on hydra to see if the gcm key string .upper() method works even though the directory has lower case "a" in hadgem)


#               
#%%  ----------------------------------------------------------------
# IMPORT AND PATH 
# ----------------------------------------------------------------

import xarray as xr
import pickle as pk
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import os
import matplotlib.pyplot as plt
scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# FLAGS
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'heatwavedarea'   # 0: all
                                    # 1: burntarea
                                    # 2: cropfailedarea
                                    # 3: driedarea
                                    # 4: floodedarea
                                    # 5: heatwavedarea
                                    # 6: tropicalcyclonedarea
                                    # 7: waterscarcity
flags['runs'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0         # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure'] = 0       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute exposure (i.e. produce and save exposure as pickle)
flags['exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)


# TODO: add rest of flags


#%% ----------------------------------------------------------------
# INITIALISE
# ----------------------------------------------------------------
from settings import *

# set global variables
init()


# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)

#%% ----------------------------------------------------------------
# LOAD AND MANIPULATE DATA
# ----------------------------------------------------------------

# TODO: when regions added, make this one function returning dict! 
from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC

df_GMT_15, df_GMT_20, df_GMT_NDC = load_GMT(
    year_start,
    year_end,
) 

# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    # load worldbank and unwpp data
    meta, worldbank, unwpp = load_worldbank_unwpp_data()

    # unpack values
    df_countries, df_regions = meta
    df_worldbank_country, df_worldbank_region = worldbank
    df_unwpp_country, df_unwpp_region = unwpp

    # manipulate worldbank and unwpp data to get birth year and life expectancy values
    df_birthyears, df_life_expectancy_5 = get_life_expectancies(
        df_worldbank_country, 
        df_unwpp_country,
    )

    # load population size per age cohort data
    wcde = load_wcde_data() 

    # interpolate population size per age cohort data to our ages (0-60)
    d_cohort_size = get_cohortsize_countries(
        wcde, 
        df_countries, 
        df_GMT_15,
    )
    
    # interpolate pop sizes per age cohort for all ages (0-104)
    d_all_cohorts = get_all_cohorts(
        wcde, 
        df_countries, 
        df_GMT_15,
    )

    # do the same for the regions; get life expectancy, birth years and cohort weights per region, as well as countries per region
    d_region_countries, df_birthyears_regions, df_life_expectancy_5_regions, d_cohort_weights_regions = get_regions_data(
        df_countries, 
        df_regions, 
        df_worldbank_region, 
        df_unwpp_region, 
        d_cohort_size,
    )

    # --------------------------------------------------------------------
    # Load population and country masks, and mask population per country
    
    # Load SSP population totals 
    da_population = load_population(
        year_start,
        year_end,
    )
    
    gdf_country_borders = gpd.read_file('./data/natural_earth/Cultural_10m/Countries/ne_10m_admin_0_countries.shp'); 

    # mask population totals per country  and save country regions object and countries mask
    df_countries, countries_regions, countries_mask, gdf_country_borders = get_mask_population(
        da_population, 
        gdf_country_borders, 
        df_countries,
    ) 

    # pack country information
    d_countries = {
        'info_pop' : df_countries, 
        'borders' : gdf_country_borders,
        'population_map' : da_population,
        'birth_years' : df_birthyears,
        'life_expectancy_5': df_life_expectancy_5, 
        'cohort_size' : d_cohort_size, 
        'all_cohorts' : d_all_cohorts,
        'mask' : (countries_regions,countries_mask),
    }

    # pack region information
    d_regions = {
        'birth_years' : df_birthyears_regions,
        'life_expectancy_5': df_life_expectancy_5_regions, 
        'cohort_size' : d_cohort_weights_regions,
    }

    # save metadata dictionary as a pickle
    print('Saving country and region data')
    
    if not os.path.isdir('./data/pickles'):
        os.mkdir('./data/pickles')
    with open('./data/pickles/country_info.pkl', 'wb') as f: # note; 'with' handles file stream closing
        pk.dump(d_countries,f)
    with open('./data/pickles/region_info.pkl', 'wb') as f:
        pk.dump(d_regions,f)

else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/pickles/country_info.pkl', 'rb'))

    # unpack country information
    df_countries = d_countries['info_pop']
    gdf_country_borders = d_countries['borders']
    da_population = d_countries['population_map']
    df_birthyears = d_countries['birth_years']
    df_life_expectancy_5 = d_countries['life_expectancy_5']
    d_cohort_size = d_countries['cohort_size']
    d_all_cohorts = d_countries['all_cohorts']
    countries_regions, countries_mask = d_countries['mask']

    # load regions pickle
    d_regions = pk.load(open('./data/pickles/region_info.pkl', 'rb'))

    # unpack region information
    df_birthyears_regions = d_regions['birth_years']
    df_life_expectancy_5_regions = d_regions['life_expectancy_5']
    d_cohort_weights_regions = d_regions['cohort_size']
 
# --------------------------------------------------------------------
# Load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta,d_pic_meta = load_isimip(
    flags['runs'], 
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,    
)

#%% ----------------------------------------------------------------
# COMPUTE EXPOSURE PER LIFETIME
# ------------------------------------------------------------------

from exposure import * 

# --------------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span

if flags['exposure']: 
    
    start_time = time.time()
    
    # calculate exposure per country and per region and save data (takes 23 mins)
    d_exposure_perrun_RCP, d_exposure_perregion_perrun_RCP, d_exposure_perrun_15, d_exposure_perrun_20, d_exposure_perrun_NDC, da_exposure_cohort = calc_exposure(
        grid_area,
        d_regions,
        d_isimip_meta, 
        df_birthyears_regions, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5,
        d_all_cohorts,
    )
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )

else: # load processed country data

    print('Loading processed exposures')

    # load country pickle
    with open('./data/pickles/exposure_{}.pkl'.format(d_isimip_meta[list(d_isimip_meta.keys())[0]]['extreme']), 'rb') as f:
        d_exposure = pk.load(f)

    # unpack country information
    d_exposure_perrun_RCP = d_exposure['exposure_perrun_RCP']
    d_exposure_perrun_15 = d_exposure['exposure_perrun_15']
    d_exposure_perrun_20 = d_exposure['exposure_perrun_20']
    d_exposure_perrun_NDC = d_exposure['exposure_perrun_NDC']
    da_exposure_cohort = d_exposure['exposure_per_cohort']

    # unpack region information
    d_exposure_perregion_perrun_RCP = d_exposure['exposure_perregion_perrun_RCP']
    d_landfrac_peryear_perregion = d_exposure['landfrac_peryear_perregion']

# --------------------------------------------------------------------
# process picontrol data

if flags['exposure_pic']:
    
    start_time = time.time()
    
    # takes 38 mins crop failure
    d_exposure_perrun_pic, d_exposure_perregion_perrun_pic, = calc_exposure_pic(
        grid_area,
        d_regions,
        d_pic_meta, 
        df_birthyears_regions, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5,
    )
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )    
    
else: # load processed country data
    
    print('Loading processed exposures')

    with open('./data/pickles/exposure_pic_{}.pkl'.format(d_pic_meta[list(d_pic_meta.keys())[0]]['extreme']), 'rb') as f:
        d_exposure_pic = pk.load(f)
    
    # unpack pic country information
    d_exposure_perrun_pic = d_exposure_pic['exposure_perrun']
    
    # unpack pic regional information
    d_exposure_perregion_perrun_pic = d_exposure_pic['exposure_perregion_perrun']
    d_landfrac_peryear_perregion_pic = d_exposure_pic['landfrac_peryear_perregion']
    
#%% --------------------------------------------------------------------
# compile hist+RCP and pic for EMF and emergence analysis

# call function to compute mmm, std, qntl for exposure (also 99.99 % of pic as "ext")
ds_exposure_RCP = calc_exposure_mmm_xr(
    d_exposure_perrun_RCP,
    'country',
    'RCP',
)
ds_exposure_15 = calc_exposure_mmm_xr(
    d_exposure_perrun_15,
    'country',
    '15',
)
ds_exposure_20 = calc_exposure_mmm_xr(
    d_exposure_perrun_20,
    'country',
    '20',
)
ds_exposure_NDC = calc_exposure_mmm_xr(
    d_exposure_perrun_NDC,
    'country',
    'NDC',
)
ds_exposure_perregion = calc_exposure_mmm_xr(
    d_exposure_perregion_perrun_RCP,
    'region',
    'RCP',
)
ds_exposure_pic = calc_exposure_mmm_pic_xr(
    d_exposure_perrun_pic,
    'country',
    'pic',
)
ds_exposure_pic_perregion = calc_exposure_mmm_pic_xr(
    d_exposure_perregion_perrun_pic,
    'region',
    'pic',
)

# pool all datasets for different trajectories
ds_exposure = xr.merge([
    ds_exposure_RCP,
    ds_exposure_15,
    ds_exposure_20,
    ds_exposure_NDC,
])

#%% ----------------------------------------------------------------
# COMPUTE EMERGENCE PER LIFETIME
# ------------------------------------------------------------------

from emergence import *

# ADD FLAG OPTION FOR COMPUTE OR LOAD PICKLES, AND THEN WRITE LOAD COMMANDS

# # emergence calculations
# gdf_exposure_emergence_birth_year = calc_exposure_emergence(
#     ds_exposure,
#     ds_exposure_pic,
#     gdf_country_borders,
# )

# # plot emergence
# emergence_plot(
#     gdf_exposure_emergence_birth_year,
# )

# cohort exposure
ds_exposure_cohort = calc_cohort_emergence(
    da_exposure_cohort,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
)

# population experiencing normal vs unprecedented exposure
ds_pop_frac = calc_unprec_exposure(
    ds_exposure_cohort,
    ds_exposure_pic,
)

ds_pop_frac['mean_unprec'] = ds_pop_frac['unprec'].mean(dim='runs')
ds_pop_frac['max_unprec'] = ds_pop_frac['unprec'].max(dim='runs')
ds_pop_frac['min_unprec'] = ds_pop_frac['unprec'].min(dim='runs')

ds_pop_frac['mean_normal'] = ds_pop_frac['normal'].mean(dim='runs')
ds_pop_frac['max_normal'] = ds_pop_frac['normal'].max(dim='runs')
ds_pop_frac['min_normal'] = ds_pop_frac['normal'].min(dim='runs')

# pack exposure information
d_unprecedented_exposure = {
    'exposure_dataset' : ds_exposure_cohort, 
    'population_fraction_dataset' : ds_pop_frac,
}

with open('./data/pickles/cohort_exposure_{}.pkl'.format(flags['extr']), 'wb') as f:
    pk.dump(d_exposure,f)

x=12
y=9
lw_mean=1
lw_fill=0.1
ub_alpha = 0.5
title_font = 14
tick_font = 12
axis_font = 14
legend_font = 14
impactyr_font =  11
col_grid = '0.8'     # color background grid
style_grid = 'dashed'     # style background grid
lw_grid = 0.5     # lineweight background grid
col_unprec = 'darkred'       # unprec mean color
col_unprec_fill = '#F08080'     # unprec fill color
col_normal = 'steelblue'       # normal mean color
col_normal_fill = 'lightsteelblue'     # normal fill color
col_bis = 'black'     # color bisector
style_bis = '--'     # style bisector
lw_bis = 1     # lineweight bisector
time = ds_pop_frac.time.values
xmin = np.min(time)
xmax = np.max(time)
f,ax = plt.subplots(figsize=(x,y))

# plot unprecedented
ax.plot(
    time,
    ds_pop_frac['mean_unprec'].values,
    lw=lw_mean,
    color=col_unprec,
    label='Population unprecedented',
    zorder=1,
)
ax.fill_between(
    time,
    ds_pop_frac['max_unprec'].values,
    ds_pop_frac['min_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_unprec_fill,
    zorder=1,
)

# plot normal
ax.plot(
    time,
    ds_pop_frac['mean_normal'].values,
    lw=lw_mean,
    color=col_normal,
    label='Population normal',
    zorder=2,
)
ax.fill_between(
    time,
    ds_pop_frac['max_normal'].values,
    ds_pop_frac['min_normal'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_normal_fill,
    zorder=2,
)


# %%
