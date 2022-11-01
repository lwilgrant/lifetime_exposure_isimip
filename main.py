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
# import and path
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
# flags
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
flags['emergence'] = 1      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)

# TODO: add rest of flags


#%% ----------------------------------------------------------------
# initialize
# ----------------------------------------------------------------
from settings import *

# set global variables
init()


# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)

#%% ----------------------------------------------------------------
# load and manipulate demographic, GMT and ISIMIP data
# ----------------------------------------------------------------

# TODO: when regions added, make this one function returning dict! 
from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC

df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj, ind_15, ind_20, ind_NDC = load_GMT(
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
# load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta,d_pic_meta = load_isimip(
    flags['runs'], 
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
)

#%% ----------------------------------------------------------------
# compute exposure per lifetime
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

else: # load processed exposure data

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
    da_exposure_cohort_strj,\
    da_exposure_peryear_perage_percountry_RCP,\
    da_exposure_peryear_perage_percountry_15,\
    da_exposure_peryear_perage_percountry_20,\
    da_exposure_peryear_perage_percountry_NDC,\
    da_exposure_peryear_perage_percountry_strj = exposure_cohort   
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )
    
else:  # load processed cohort exposure data
    
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
    with open('./data/pickles/exposure_cohort_strj_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_cohort_strj = pk.load(f)        
        
    # pickle exposure peryear perage percountry
    with open('./data/pickles/exposure_peryear_perage_percountry_RCP_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_RCP = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_15_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_15 = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_20_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_20 = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_NDC_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_NDC = pk.load(f)
    with open('./data/pickles/exposure_peryear_perage_percountry_strj_{}.pkl'.format(flags['extr']), 'rb') as f:
        da_exposure_peryear_perage_percountry_strj = pk.load(f)

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
    
else: # load processed pic data
    
    print('Loading processed pic exposures')

    with open('./data/pickles/exposure_pic_{}.pkl'.format(flags['extr']), 'rb') as f:
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

# pool all mmm exposures together
ds_exposure = xr.merge([
    ds_exposure_RCP,
    ds_exposure_15,
    ds_exposure_20,
    ds_exposure_NDC,
])

#%% ----------------------------------------------------------------
# compute lifetime emergence
# ------------------------------------------------------------------

from emergence import *

# --------------------------------------------------------------------
# process emergence of cumulative exposures, mask cohort exposures for time steps of emergence

if flags['emergence']:
    
    # cohort conversion to data array (again)
    da_cohort_size = xr.DataArray(
        np.asarray([v for k,v in d_all_cohorts.items() if k in list(df_countries['name'])]),
        coords={
            'country': ('country', list(df_countries['name'])),
            'time': ('time', year_range),
            'ages': ('ages', np.arange(104,-1,-1)),
        },
        dims=[
            'country',
            'time',
            'ages',
        ]
    )
    
    # need new cohort dataset that has total population per birth year (using life expectancy info; each country has a different end point)
    da_cohort_aligned = calc_birthyear_align(
        da_cohort_size,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )
    
    # convert to dataset and add weights
    ds_cohorts = ds_cohort_align(da_cohort_aligned)
    
    # pickle birth year aligned cohort sizes
    with open('./data/pickles/cohort_per_birthyear.pkl', 'wb') as f:
        pk.dump(ds_cohorts,f)    

    # ds_age_emergence_RCP, ds_pop_frac_RCP = all_emergence(
    #     da_exposure_peryear_perage_percountry_RCP,
    #     da_exposure_cohort_RCP,
    #     df_life_expectancy_5,
    #     year_start,
    #     year_end,
    #     year_ref,
    #     ds_exposure_pic,
    #     ds_cohorts,
    #     flags['extr'],
    #     'RCP',
    # )

    ds_age_emergence_15, ds_pop_frac_15 = all_emergence(
        da_exposure_peryear_perage_percountry_15,
        da_exposure_cohort_15,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
        ds_exposure_pic,
        ds_cohorts,
        flags['extr'],
        '15',
    )

    ds_age_emergence_20, ds_pop_frac_20 = all_emergence(
        da_exposure_peryear_perage_percountry_20,
        da_exposure_cohort_20,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
        ds_exposure_pic,
        ds_cohorts,
        flags['extr'],
        '20',
    )

    ds_age_emergence_NDC, ds_pop_frac_NDC = all_emergence(
        da_exposure_peryear_perage_percountry_NDC,
        da_exposure_cohort_NDC,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
        ds_exposure_pic,
        ds_cohorts,
        flags['extr'],
        'NDC',
    )
    
    ds_age_emergence_strj, ds_pop_frac_strj = strj_emergence(
        da_exposure_peryear_perage_percountry_strj,
        da_exposure_cohort_strj,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
        ds_exposure_pic,
        ds_cohorts,
        flags['extr'],
        'strj',
    )
        
else: # load pickles
    
    # birth year aligned population
    with open('./data/pickles/cohort_per_birthyear.pkl', 'rb') as f:
        ds_cohorts = pk.load(f)
    
    # pop frac
    # with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flags['extr'],'RCP'), 'rb') as f:
    #     ds_pop_frac_RCP = pk.load(f)
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flags['extr'],'15'), 'rb') as f:
        ds_pop_frac_15 = pk.load(f)
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flags['extr'],'20'), 'rb') as f:
        ds_pop_frac_20 = pk.load(f)
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flags['extr'],'NDC'), 'rb') as f:
        ds_pop_frac_NDC = pk.load(f)    
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flags['extr'],'strj'), 'rb') as f:
        ds_pop_frac_strj = pk.load(f)                
    
    # age emergence
    # with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flags['extr'],'RCP'), 'rb') as f:
    #     ds_age_emergence_RCP = pk.load(f)
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flags['extr'],'15'), 'rb') as f:
        ds_age_emergence_15 = pk.load(f)
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flags['extr'],'20'), 'rb') as f:
        ds_age_emergence_20 = pk.load(f)
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flags['extr'],'NDC'), 'rb') as f:
        ds_age_emergence_NDC = pk.load(f)                    
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flags['extr'],'strj'), 'rb') as f:
        ds_age_emergence_strj = pk.load(f)         
        
#%% ----------------------------------------------------------------
# plot emergence stuff
# ------------------------------------------------------------------

from plot import *   

# collect all data arrays for age of emergence into dataset for finding age per birth year
ds_age_emergence = xr.merge([
    # ds_age_emergence_RCP.rename({'age_emergence':'age_emergence_RCP'}),
    ds_age_emergence_15.rename({'age_emergence':'age_emergence_15'}),
    ds_age_emergence_20.rename({'age_emergence':'age_emergence_20'}),
    ds_age_emergence_NDC.rename({'age_emergence':'age_emergence_NDC'}),
])
        
# plot pop frac of 3 main GMT mapped scenarios across birth years
plot_pop_frac_birth_year(
    ds_pop_frac_NDC,
    ds_pop_frac_15,
    ds_pop_frac_20,
    year_range,
)

# plot pop frac for 0.8-3.5 degree stylized trajectories across birth years
plot_pop_frac_birth_year_strj(
    ds_pop_frac_strj,
    df_GMT_strj,
)

# plot pop frac and age emergence across GMT for stylized trajectories
plot_pop_frac_birth_year_GMT_strj(
    ds_pop_frac_strj,
    ds_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
    year_range,
)

# plot country mean age of emergence of 3 main GMT mapped scenarios across birth year
plot_age_emergence(
    ds_age_emergence_NDC,
    ds_age_emergence_15,
    ds_age_emergence_20,
    year_range,
)

# plot country mean age of emergence of stylized trajectories across birth years
plot_age_emergence_strj(
    ds_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
    year_range,
)

# calculate birth year emergence in simple approach
gdf_exposure_emergence_birth_year = calc_exposure_emergence(
    ds_exposure,
    ds_exposure_pic,
    ds_age_emergence,
    gdf_country_borders,
)

# country-scale spatial plot of birth and year emergence
emergence_plot(
    gdf_exposure_emergence_birth_year,
)

# plot stylized trajectories (GMT only)
plot_stylized_trajectories(
    df_GMT_strj,
    d_isimip_meta,
    year_range,
)

# plot pop frac across GMT for stylized trajectories; add points for 1.5, 2.0 and NDC from original analysis as test
plot_pop_frac_birth_year_GMT_strj_points(
    ds_pop_frac_strj,
    ds_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
    ds_age_emergence,
    ds_pop_frac_15,
    ds_pop_frac_20,
    ds_pop_frac_NDC,
    ind_15,
    ind_20,
    ind_NDC,
    year_range,
)

# %%
