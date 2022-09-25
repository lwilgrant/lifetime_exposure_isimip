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
flags['extr'] = 'heatwavedarea'     # 0: all
                                    # 1: burntarea
                                    # 2: cropfailedarea
                                    # 3: driedarea
                                    # 4: floodedarea
                                    # 5: heatwavedarea
                                    # 6: tropicalcyclonedarea
                                    # 7: waterscarcity
flags['runs'] = 0           # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure'] = 0       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute exposure (i.e. produce and save exposure as pickle)
flags['exposure_cohort'] = 0       # 0: do not process ISIMIP runs to compute exposure across cohorts (i.e. load exposure pickle)
                                   # 1: process ISIMIP runs to compute exposure across cohorts (i.e. produce and save exposure as pickle)                            
flags['exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)
flags['emergence'] = 0      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)

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

    # -------------------------------------------------------------------------------------------------------
    
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
    exposures = calc_exposure(
        grid_area,
        d_regions,
        d_isimip_meta, 
        df_birthyears_regions, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5,
    )
    
    d_exposure_perrun_RCP,\
    d_exposure_perregion_perrun_RCP,\
    d_exposure_perrun_15,\
    d_exposure_perrun_20,\
    d_exposure_perrun_NDC = exposures
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )

else: # load processed country data

    print('Loading processed exposures')

    # load country pickle
    with open('./data/pickles/exposure_{}.pkl'.format(flags['extr']), 'rb') as f:
        d_exposure = pk.load(f)

    # unpack country information
    d_exposure_perrun_RCP = d_exposure['exposure_perrun_RCP']
    d_exposure_perrun_15 = d_exposure['exposure_perrun_15']
    d_exposure_perrun_20 = d_exposure['exposure_perrun_20']
    d_exposure_perrun_NDC = d_exposure['exposure_perrun_NDC']

    # unpack region information
    d_exposure_perregion_perrun_RCP = d_exposure['exposure_perregion_perrun_RCP']
    d_landfrac_peryear_perregion = d_exposure['landfrac_peryear_perregion']

# --------------------------------------------------------------------
# process exposure across cohorts

if flags['exposure_cohort']:
    
    start_time = time.time()
    
    # calculate exposure per country and per cohort
    exposure_cohort = calc_cohort_exposure(
        d_isimip_meta,
        df_countries,
        countries_regions,
        countries_mask,
        da_population,
        d_all_cohorts,
    )
    
    da_exposure_cohort_RCP,\
    da_exposure_cohort_15,\
    da_exposure_cohort_20,\
    da_exposure_cohort_NDC,\
    da_exposure_peryear_perage_percountry_RCP,\
    da_exposure_peryear_perage_percountry_15,\
    da_exposure_peryear_perage_percountry_20,\
    da_exposure_peryear_perage_percountry_NDC = exposure_cohort
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )
    
else:
    
    print('Loading processed exposures')

    # pickle cohort exposures
    with open('./data/pickles/exposure_cohort_RCP_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_cohort_RCP = pk.load(f)
    with open('./data/pickles/exposure_cohort_15_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_cohort_15 = pk.load(f)
    with open('./data/pickles/exposure_cohort_20_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_cohort_20 = pk.load(f)
    with open('./data/pickles/exposure_cohort_NDC_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_cohort_NDC = pk.load(f)
        
    # pickle exposure peryear perage percountry
    with open('./data/pickles/exposure_peryear_perage_percountry_RCP_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_RCP = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_15_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_15 = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_20_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_20 = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_NDC_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_NDC = pk.load(f)

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
    
    print('Loading processed pic exposures')

    with open('./data/pickles/exposure_pic_{}.pkl'.format(d_pic_meta[list(d_pic_meta.keys())[0]]['extreme']), 'rb') as f:
        d_exposure_pic = pk.load(f)
    
    # unpack pic country information
    d_exposure_perrun_pic = d_exposure_pic['exposure_perrun']
    
    # unpack pic regional information
    d_exposure_perregion_perrun_pic = d_exposure_pic['exposure_perregion_perrun']
    d_landfrac_peryear_perregion_pic = d_exposure_pic['landfrac_peryear_perregion']
    
#%% --------------------------------------------------------------------
# compile hist+RCP and pic for EMF

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

#%% ----------------------------------------------------------------
# COMPUTE EMERGENCE PER LIFETIME
# ------------------------------------------------------------------

from emergence import *

# --------------------------------------------------------------------
# process exposures to sum cumulatively across life expectancies, for comparison against cohort exposures for pop frac analysis

# RCP unprecedented exposure
# ds_exposure_mask_RCP = calc_cohort_emergence(
ds_exposure_mask_RCP = calc_cohort_emergence(
    da_exposure_peryear_perage_percountry_RCP,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
)

# RCP mask for unprecedented exposure and age of emergence
ds_exposure_mask_RCP,age_emergence_RCP = exposure_pic_masking(
    ds_exposure_mask_RCP,
    ds_exposure_pic,
)

# 1.5 deg unprecedented exposure
ds_exposure_mask_15 = calc_cohort_emergence(
    da_exposure_peryear_perage_percountry_15,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
)

# 1.5 deg mask for unprecedented exposure and age of emergence
ds_exposure_mask_15,age_emergence_15 = exposure_pic_masking(
    ds_exposure_mask_15,
    ds_exposure_pic,
)

# 2.0 deg unprecedented exposure
ds_exposure_mask_20 = calc_cohort_emergence(
    da_exposure_peryear_perage_percountry_20,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
)

# 2.0 deg mask for unprecedented exposure and age of emergence
ds_exposure_mask_20,age_emergence_20 = exposure_pic_masking(
    ds_exposure_mask_20,
    ds_exposure_pic,
)

# NDC unprecedented exposure
ds_exposure_mask_NDC = calc_cohort_emergence(
    da_exposure_peryear_perage_percountry_NDC,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
)

# NDC mask for unprecedented exposure and age of emergence
ds_exposure_mask_NDC,age_emergence_NDC = exposure_pic_masking(
    ds_exposure_mask_NDC,
    ds_exposure_pic,
)

# --------------------------------------------------------------------
# process cohort emergence 
if flags['emergence']:

    # cohort exposure RCP
    ds_exposure_cohort_RCP = calc_cohort_emergence(
        da_exposure_cohort_RCP,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    ) 
    
    # population experiencing normal vs unprecedented exposure
    ds_pop_frac_RCP = calc_unprec_exposure(
        ds_exposure_cohort_RCP,
        ds_exposure_mask_RCP,
        d_all_cohorts,
        year_range,
        df_countries,
    )    
    
    # cohort exposure 1.5 deg
    ds_exposure_cohort_15 = calc_cohort_emergence(
        da_exposure_cohort_15,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )

    # population experiencing normal vs unprecedented exposure
    ds_pop_frac_15 = calc_unprec_exposure(
        ds_exposure_cohort_15,
        ds_exposure_mask_15,
        d_all_cohorts,
        year_range,
        df_countries,        
    )
    
    # cohort exposure 2.0 deg
    ds_exposure_cohort_20 = calc_cohort_emergence(
        da_exposure_cohort_20,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )

    # population experiencing normal vs unprecedented exposure
    ds_pop_frac_20 = calc_unprec_exposure(
        ds_exposure_cohort_20,
        ds_exposure_mask_20,
        d_all_cohorts,
        year_range,
        df_countries,        
    )
    
    # cohort exposure NDCs
    ds_exposure_cohort_NDC = calc_cohort_emergence(
        da_exposure_cohort_NDC,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )

    # population experiencing normal vs unprecedented exposure
    ds_pop_frac_NDC = calc_unprec_exposure(
        ds_exposure_cohort_NDC,
        ds_exposure_mask_NDC,
        d_all_cohorts,
        year_range,
        df_countries,        
    )    

    # # pack exposure information DOESNT WORK, TOO MUCH DATA FOR PC (USING COHORT_EXPOSURE FOR NEW "calc_cohort_exposure()" function)
    # d_unprecedented_exposure = {
    #     'exposure_ds_RCP' : ds_exposure_cohort_RCP,
    #     'pop_frac_ds_RCP' : ds_pop_frac_RCP,
    #     'exposure_ds_15' : ds_exposure_cohort_15,
    #     'pop_frac_ds_15' : ds_pop_frac_15,
    #     'exposure_ds_20' : ds_exposure_cohort_20,
    #     'pop_frac_ds_20' : ds_pop_frac_20,
    #     'exposure_ds_NDC' : ds_exposure_cohort_NDC,
    #     'pop_frac_ds_NDC' : ds_pop_frac_NDC,        
    # }

    # with open('./data/pickles/cohort_exposure_{}.pkl'.format(flags['extr']), 'wb') as f:
    #     pk.dump(d_unprecedented_exposure,f)
    
    # # use sample ds to find age of emergence
    # unprec = ds_exposure_cohort_NDC['exposure'].where(ds_exposure_cohort_NDC['exposure_cumulative'] >= ds_exposure_pic['ext'])
        
else:
    
    pass
    
    # DOESN'T WORK, TOO MUCH DATA FOR PC TO STORE THIS PICKLE (WILL DO WITH ANOTHER NAME ON SERVER)
    # with open('./data/pickles/cohort_exposure_{}.pkl'.format(flags['extr']), 'rb') as f:
    #     d_unprecedented_exposure = pk.load(f)
        
    # ds_exposure_cohort_RCP = d_unprecedented_exposure['exposure_ds_RCP']
    # ds_pop_frac_RCP = d_unprecedented_exposure['pop_frac_ds_RCP']
    # ds_exposure_cohort_15 = d_unprecedented_exposure['exposure_ds_15']
    # ds_pop_frac_15 = d_unprecedented_exposure['pop_frac_ds_15']
    # ds_exposure_cohort_20 = d_unprecedented_exposure['exposure_ds_20']
    # ds_pop_frac_20 = d_unprecedented_exposure['pop_frac_ds_20']
    # ds_exposure_cohort_NDC = d_unprecedented_exposure['exposure_ds_NDC']
    # ds_pop_frac_NDC = d_unprecedented_exposure['pop_frac_ds_NDC']    

#%% ----------------------------------------------------------------
# plotting pop frac
# ------------------------------------------------------------------
# --------------------------------------------------------------------
# plotting utils
letters = ['a', 'b', 'c',\
            'd', 'e', 'f',\
            'g', 'h', 'i',\
            'j', 'k', 'l']
x=10
y=9
lw_mean=1
lw_fill=0.1
ub_alpha = 0.5
title_font = 14
tick_font = 12
axis_font = 11
legend_font = 14
impactyr_font =  11
col_grid = '0.8'     # color background grid
style_grid = 'dashed'     # style background grid
lw_grid = 0.5     # lineweight background grid
col_NDC = 'darkred'       # unprec mean color
col_NDC_fill = '#F08080'     # unprec fill color
col_15 = 'steelblue'       # normal mean color
col_15_fill = 'lightsteelblue'     # normal fill color
col_20 = 'darkgoldenrod'   # rcp60 mean color
col_20_fill = '#ffec80'     # rcp60 fill color
col_bis = 'black'     # color bisector
style_bis = '--'     # style bisector
lw_bis = 1     # lineweight bisector
time = year_range
# xmin = np.min(time)
# xmax = np.max(time)
xmin = 1960
xmax = 2100

ax1_ylab = 'Billions unprecendented'
ax2_ylab = 'Unprecedented/Total'
ax3_ylab = 'Unprecedented/Exposed'

f,(ax1,ax2,ax3) = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(x,y),
)

# --------------------------------------------------------------------
# plot unprecedented pop numbers

# NDC
ax1.plot(
    time,
    # ds_pop_frac['mean_unprec'].values * 1000,
    ds_pop_frac_NDC['mean_unprec'].values / 1e6,
    lw=lw_mean,
    color=col_NDC,
    label='Population unprecedented',
    zorder=1,
)
ax1.fill_between(
    time,
    # (ds_pop_frac['mean_unprec'].values * 1000) + (ds_pop_frac['std_unprec'].values * 1000),
    (ds_pop_frac_NDC['mean_unprec'].values / 1e6) + (ds_pop_frac_NDC['std_unprec'].values / 1e6),
    # (ds_pop_frac['mean_unprec'].values * 1000) - (ds_pop_frac['std_unprec'].values * 1000),
    (ds_pop_frac_NDC['mean_unprec'].values / 1e6) - (ds_pop_frac_NDC['std_unprec'].values / 1e6),
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_NDC_fill,
    zorder=1,
)

# 2.0 degrees
ax1.plot(
    time,
    # ds_pop_frac['mean_unprec'].values * 1000,
    ds_pop_frac_20['mean_unprec'].values / 1e6,
    lw=lw_mean,
    color=col_20,
    label='Population unprecedented',
    zorder=2,
)
ax1.fill_between(
    time,
    # (ds_pop_frac['mean_unprec'].values * 1000) + (ds_pop_frac['std_unprec'].values * 1000),
    (ds_pop_frac_20['mean_unprec'].values / 1e6) + (ds_pop_frac_20['std_unprec'].values / 1e6),
    # (ds_pop_frac['mean_unprec'].values * 1000) - (ds_pop_frac['std_unprec'].values * 1000),
    (ds_pop_frac_20['mean_unprec'].values / 1e6) - (ds_pop_frac_20['std_unprec'].values / 1e6),
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_20_fill,
    zorder=2,
)

# 1.5 degrees
ax1.plot(
    time,
    # ds_pop_frac['mean_unprec'].values * 1000,
    ds_pop_frac_15['mean_unprec'].values / 1e6,
    lw=lw_mean,
    color=col_15,
    label='Population unprecedented',
    zorder=3,
)
ax1.fill_between(
    time,
    # (ds_pop_frac['mean_unprec'].values * 1000) + (ds_pop_frac['std_unprec'].values * 1000),
    (ds_pop_frac_15['mean_unprec'].values / 1e6) + (ds_pop_frac_15['std_unprec'].values / 1e6),
    # (ds_pop_frac['mean_unprec'].values * 1000) - (ds_pop_frac['std_unprec'].values * 1000),
    (ds_pop_frac_15['mean_unprec'].values / 1e6) - (ds_pop_frac_15['std_unprec'].values / 1e6),
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_15_fill,
    zorder=3,
)

ax1.set_ylabel(
    ax1_ylab, 
    va='center', 
    rotation='vertical', 
    fontsize=axis_font, 
    labelpad=10,
)

# --------------------------------------------------------------------
# plot unprecedented frac of total pop

# NDC
ax2.plot(
    time,
    ds_pop_frac_NDC['mean_frac_all_unprec'].values,
    lw=lw_mean,
    color=col_NDC,
    label='Population unprecedented',
    zorder=1,
)
ax2.fill_between(
    time,
    ds_pop_frac_NDC['mean_frac_all_unprec'].values + ds_pop_frac_NDC['std_frac_all_unprec'].values,
    ds_pop_frac_NDC['mean_frac_all_unprec'].values - ds_pop_frac_NDC['std_frac_all_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_NDC_fill,
    zorder=1,
)

# 2.0 degrees
ax2.plot(
    time,
    ds_pop_frac_20['mean_frac_all_unprec'].values,
    lw=lw_mean,
    color=col_20,
    label='Population unprecedented',
    zorder=2,
)
ax2.fill_between(
    time,
    ds_pop_frac_20['mean_frac_all_unprec'].values + ds_pop_frac_20['std_frac_all_unprec'].values,
    ds_pop_frac_20['mean_frac_all_unprec'].values - ds_pop_frac_20['std_frac_all_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_20_fill,
    zorder=2,
)

# 1.5 degrees
ax2.plot(
    time,
    ds_pop_frac_15['mean_frac_all_unprec'].values,
    lw=lw_mean,
    color=col_15,
    label='Population unprecedented',
    zorder=3,
)
ax2.fill_between(
    time,
    ds_pop_frac_15['mean_frac_all_unprec'].values + ds_pop_frac_15['std_frac_all_unprec'].values,
    ds_pop_frac_15['mean_frac_all_unprec'].values - ds_pop_frac_15['std_frac_all_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_15_fill,
    zorder=3,
)

ax2.set_ylabel(
    ax2_ylab, 
    va='center', 
    rotation='vertical', 
    fontsize=axis_font, 
    labelpad=10,
)

# --------------------------------------------------------------------
# plot unprecedented frac of exposed pop

# NDC
ax3.plot(
    time,
    ds_pop_frac_NDC['mean_frac_exposed_unprec'].values,
    lw=lw_mean,
    color=col_NDC,
    label='Population unprecedented',
    zorder=1,
)
ax3.fill_between(
    time,
    ds_pop_frac_NDC['mean_frac_exposed_unprec'].values + ds_pop_frac_NDC['std_frac_exposed_unprec'].values,
    ds_pop_frac_NDC['mean_frac_exposed_unprec'].values - ds_pop_frac_NDC['std_frac_exposed_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_NDC_fill,
    zorder=1,
)

# 2.0 degrees
ax3.plot(
    time,
    ds_pop_frac_20['mean_frac_exposed_unprec'].values,
    lw=lw_mean,
    color=col_20,
    label='Population unprecedented',
    zorder=2,
)
ax3.fill_between(
    time,
    ds_pop_frac_20['mean_frac_exposed_unprec'].values + ds_pop_frac_20['std_frac_exposed_unprec'].values,
    ds_pop_frac_20['mean_frac_exposed_unprec'].values - ds_pop_frac_20['std_frac_exposed_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_20_fill,
    zorder=2,
)

# 1.5 degrees
ax3.plot(
    time,
    ds_pop_frac_15['mean_frac_exposed_unprec'].values,
    lw=lw_mean,
    color=col_15,
    label='Population unprecedented',
    zorder=3,
)
ax3.fill_between(
    time,
    ds_pop_frac_15['mean_frac_exposed_unprec'].values + ds_pop_frac_15['std_frac_exposed_unprec'].values,
    ds_pop_frac_15['mean_frac_exposed_unprec'].values - ds_pop_frac_15['std_frac_exposed_unprec'].values,
    lw=lw_fill,
    alpha=ub_alpha,
    color=col_15_fill,
    zorder=3,
)

ax3.set_ylabel(
    ax3_ylab, 
    va='center', 
    rotation='vertical', 
    fontsize=axis_font, 
    labelpad=10,
)

for i,ax in enumerate([ax1,ax2,ax3]):
    ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
    ax.set_xlim(xmin,xmax)
    # ax.xaxis.set_ticks(xticks_ts)
    # ax.xaxis.set_ticklabels(xtick_labels_ts)
    ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
    ax.tick_params(labelsize=tick_font,axis="y",direction="in")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
    ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
    ax.set_axisbelow(True) 
    if i < 2:
        ax.tick_params(labelbottom=False)

# %%
