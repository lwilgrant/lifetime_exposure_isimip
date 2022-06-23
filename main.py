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

# Luke's review:
# writing "# DONE" to mark as read


# TODO
# - not yet masked for small countries
# - how to handle South-Sudan and Palestina? Now manually filtered out in load
# - calc_weighted_fldmean: adding different masks is very time-inefficient. Find more efficient way of doing this


#               
#%%  ----------------------------------------------------------------
# IMPORT AND PATH 
# ----------------------------------------------------------------

import os
import requests
from zipfile import ZipFile
import io
import xarray as xr
import pickle as pk
import time

scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# FLAGS
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'cropfailedarea'   # 0: all
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
flags['exposure_ens'] = 0   # 0: do not perform sample statistics for exposure across all RCPs (separate mmm, std and iqr for 2.6 and 6.0)
                            # 1: sample exposure across all RCPs (lumped mmm, std and iqr for 2.6 and 6.0)


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

    # interpolate population size per age cohort data to our ages
    d_cohort_size = get_cohortsize_countries(
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
    d_exposure_perrun_RCP, d_exposure_perregion_perrun_RCP, = calc_exposure(
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

ds_exposure = xr.merge([
    ds_exposure_RCP,
    ds_exposure_15,
    ds_exposure_20,
    ds_exposure_NDC,
])

# emergence calculations
gdf_exposure_emergence_birth_year = calc_exposure_emergence(
    ds_exposure_RCP,
    ds_exposure_pic,
    gdf_country_borders,
)


def calc_exposure_emergence(
    ds_exposure,
    ds_exposure_pic,
    gdf_country_borders,
):

    mmm_subset = [
        'mmm_RCP',
        'mmm_15',
        'mmm_20',
        'mmm_NDC',
    ]

    EMF_subset = [
        'mmm_EMF_RCP',
        'mmm_EMF_15',
        'mmm_EMF_20',
        'mmm_EMF_NDC',
    ]

    ds_exposure_emergence = ds_exposure[mmm_subset].where(ds_exposure[mmm_subset] > ds_exposure_pic.ext_pic)
    ds_exposure_emergence_birth_year = ds_exposure_emergence.birth_year.where(ds_exposure_emergence.notnull()).min(dim='birth_year',skipna=True)
    da_exposure_emergence_birth_year_EMF = ds_exposure.mmm_EMF.where(ds_exposure.mmm_EMF.birth_year==da_exposure_emergence_birth_year).min(dim='birth_year',skipna=True)
    gdf_exposure_emergence_birth_year = da_exposure_emergence_birth_year.to_dataframe().join(gdf_country_borders)
    
    gdf_exposure_emergence_birth_year = gdf_exposure_emergence_birth_year.rename(columns={'birth_year':'emergence_year'})
    
    return gdf_exposure_emergence_birth_year

# --------------------------------------------------------------------
# compute averages across runs and sums across extremes 


# # call function computing the Exposure Multiplication Factor (EMF)
# # here I use multi-model mean as a reference
# d_EMF_mmm, d_EMF_q25, d_EMF_q75 = calc_exposure_EMF(
#     d_exposure_mmm, 
#     d_exposure_q25, 
#     d_exposure_q75, 
#     d_exposure_mmm,
# )


# maybe more values needed to return


# matlab parts not translated
# [exposure_15            , exposure_mms_15            , EMF_15                                                                                                                            ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perrun_15               , [], RCP2GMT_maxdiff_15    , RCP2GMT_maxdiff_threshold);
# [exposure_20            , exposure_mms_20            , EMF_20                                                                                                                            ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perrun_20               , [], RCP2GMT_maxdiff_20    , RCP2GMT_maxdiff_threshold);
# [exposure_NDC           , exposure_mms_NDC           , EMF_NDC                                                                                                                           ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perrun_NDC              , [], RCP2GMT_maxdiff_NDC   , RCP2GMT_maxdiff_threshold);
# [exposure_perregion_15  , exposure_perregion_mms_15  , EMF_perregion_15     , EMF_perregion_q25_15     , EMF_perregion_q75_15                                                            ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_15     , [], RCP2GMT_maxdiff_15    , RCP2GMT_maxdiff_threshold);
# [exposure_perregion_20  , exposure_perregion_mms_20  , EMF_perregion_20     , EMF_perregion_q25_20     , EMF_perregion_q75_20                                                            ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_20     , [], RCP2GMT_maxdiff_20    , RCP2GMT_maxdiff_threshold);
# [exposure_perregion_NDC , exposure_perregion_mms_NDC , EMF_perregion_NDC    , EMF_perregion_q25_NDC    , EMF_perregion_q75_NDC                                                           ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_NDC    , [], RCP2GMT_maxdiff_NDC   , RCP2GMT_maxdiff_threshold);
# [exposure_perregion_OS  , exposure_perregion_mms_OS  , EMF_perregion_OS     , EMF_perregion_q25_OS     , EMF_perregion_q75_OS  , exposure_perregion_q25_OS  , exposure_perregion_q75_OS  ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_OS     , [], RCP2GMT_maxdiff_OS    , RCP2GMT_maxdiff_threshold);
# [exposure_perregion_noOS, exposure_perregion_mms_noOS, EMF_perregion_noOS   , EMF_perregion_q25_noOS   , EMF_perregion_q75_noOS, exposure_perregion_q25_noOS, exposure_perregion_q75_noOS] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_noOS   , [], RCP2GMT_maxdiff_noOS  , RCP2GMT_maxdiff_threshold);


# [exposure_perregion_R26    , ~                          , EMF_perregion_R26    , EMF_perregion_q25_R26    , EMF_perregion_q75_R26                                                           ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_rcp26 & ind_gfdl, ages, age_ref, exposure_perregion_perrun_RCP    , [], []                     , RCP2GMT_maxdiff_threshold);
# [exposure_perregion_R26eval, ~                          , EMF_perregion_R26eval, EMF_perregion_q25_R26eval, EMF_perregion_q75_R26eval                                                       ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt   & ind_gfdl, ages, age_ref, exposure_perregion_perrun_R26eval, [], RCP2GMT_maxdiff_R26eval, RCP2GMT_maxdiff_threshold);


# [~                      , ~                          , EMF_perregion_15_young2pic , ~                              , ~                                                                   ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_15  , exposure_perregion_pic_mean, RCP2GMT_maxdiff_NDC , RCP2GMT_maxdiff_threshold);
# [~                      , ~                          , EMF_perregion_20_young2pic , ~                              , ~                                                                   ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_20  , exposure_perregion_pic_mean, RCP2GMT_maxdiff_NDC , RCP2GMT_maxdiff_threshold);
# [~                      , ~                          , EMF_perregion_NDC_young2pic, EMF_perregion_q25_NDC_young2pic, EMF_perregion_q75_NDC_young2pic                                     ] = mf_exposure_mmm(extremes, {isimip.extreme}, ind_gmt  , ages, age_ref, exposure_perregion_perrun_NDC , exposure_perregion_pic_mean, RCP2GMT_maxdiff_NDC , RCP2GMT_maxdiff_threshold);


# for ISIpedia article: country-level EMF youn2pic
# not translated

# loop over heatwave definitions - for heatwave sensitivity analysis requested by the reviewer
# not translated


#  calculations for Burning Embers diagram
# not translated


#%% ----------------------------------------------------------------
# PLOTTING - to be added
# ----------------------------------------------------------------


# %%
