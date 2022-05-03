# ---------------------------------------------------------------
# Main script to postprocess and visualise lifetime exposure data
#
# Python translation of the MATLAB scripts of Thiery et al. (2021)
# https://github.com/VUB-HYDR/2021_Thiery_etal_Science
# ----------------------------------------------------------------

#%% ----------------------------------------------------------------
# Summary and notes


# Data types are defined in the variable names starting with:  
#     df_     : DataFrame    (pandas)
#     gdf_    : GeoDataFrame (geopandas)
#     da_     : DataArray    (xarray)
#     d_   : dictionary  


# TODO
# - not yet masked for small countries
# - how to handle South-Sudan and Palestina? Now manually filtered out in load



#               
#%%  ----------------------------------------------------------------
# IMPORT AND PATH 
# ----------------------------------------------------------------

import os
import xarray as xr

scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# FLAGS
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr']  = 'floodedarea'      # 0: all
                                    # 1: burntarea
                                    # 2: cropfailedarea
                                    # 3: driedarea
                                    # 4: floodedarea
                                    # 5: heatwavedarea
                                    # 6: tropicalcyclonedarea
                                    # 7: waterscarcity

flags['runs']  = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask']  = 0          # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure'] = 1       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
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

df_GMT_15, df_GMT_20, df_GMT_NDC = load_GMT(year_start,year_end)



# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    # load worldbank and unwpp data
    meta, worldbank, unwpp = load_worldbank_unwpp_data() 

    # unpack values
    df_countries        , df_regions          = meta
    df_worldbank_country, df_worldbank_region = worldbank
    df_unwpp_country    , df_unwpp_region     = unwpp

    # manipulate worldbank and unwpp data to get birth year and life expectancy values
    df_birthyears, df_life_expectancy_5 = get_life_expectancies(df_worldbank_country, df_unwpp_country)


    # load population size per age cohort data
    wcde = load_wcde_data()

    # interpolate population size per age cohort data to our ages
    d_cohort_size         = get_cohortsize_countries(wcde, df_countries, df_GMT_15)


    # do the same for the regions
    # get life expectancy, birth years and cohort weights per region, as well as countries per region

    d_region_countries, df_birthyears_regions, df_life_expectancy_5_regions, d_cohort_weights_regions = get_regions_data(df_countries, df_regions, df_worldbank_region, df_unwpp_region, d_cohort_size)
    

    # TODO get regions mask


    # --------------------------------------------------------------------
    # Load population and country masks, and mask population per country
    

    # Load SSP population totals 
    da_population = load_population(year_start,year_end)


    # load country borders
    gdf_country_borders = gpd.read_file('./data/natural_earth/Cultural_10m/Countries/ne_10m_admin_0_countries.shp'); 


    # mask population totals per country  and save country regions object and countries mask
    df_countries, countries_regions, countries_mask = get_mask_population(da_population, gdf_country_borders, df_countries)

    # pack country information
    d_countries = {'info_pop'         : df_countries, 
                   'population_map'   : da_population,
                   'birth_years'      : df_birthyears,
                   'life_expectancy_5': df_life_expectancy_5, 
                   'cohort_size'      : d_cohort_size, 
                   'mask'             : (countries_regions,countries_mask)}


    # pack region information
    d_regions = {'birth_years'      : df_birthyears_regions,
                 'life_expectancy_5': df_life_expectancy_5_regions, 
                 'cohort_size'      : d_cohort_weights_regions }


    # save metadata dictionary as a pickle
    print('Saving country and region data')

    pk.dump(d_countries,open('./data/pickles/country_info.pkl', 'wb')  )
    pk.dump(d_regions,open('./data/pickles/region_info.pkl', 'wb')  )


else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/pickles/country_info.pkl', 'rb'))

    # unpack country information
    df_countries                      = d_countries['info_pop']
    da_population                     = d_countries['population_map']
    df_birthyears                     = d_countries['birth_years']
    df_life_expectancy_5              = d_countries['life_expectancy_5']
    d_cohort_size                     = d_countries['cohort_size']
    countries_regions, countries_mask = d_countries['mask']

    # load regions pickle
    d_regions = pk.load(open('./data/pickles/region_info.pkl', 'rb'))

    # unpack region information
    df_birthyears_regions             = d_regions['birth_years']
    df_life_expectancy_5_regions      = d_regions['life_expectancy_5']
    d_cohort_weights_regions          = d_regions['cohort_size']
 
# --------------------------------------------------------------------
# Load ISIMIP model data

grid_area                = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta = load_isimip(flags['runs'], extremes, model_names)


#%% ----------------------------------------------------------------
# COMPUTE EXPOSURE PER LIFETIME
# ------------------------------------------------------------------

from exposure import *

if flags['exposure']: 
    
    # initialise dicts
    d_RCP2GMT_maxdiff_15      = {}
    d_RCP2GMT_maxdiff_20      = {}
    d_RCP2GMT_maxdiff_NDC     = {}
    d_RCP2GMT_maxdiff_R26eval = {}

    d_exposure_perrun_RCP = {}
    d_exposure_perrun_15 = {}
    d_exposure_perrun_20 = {}
    d_exposure_perrun_NDC = {}
    d_exposure_perrun_R26eval = {} 


    # loop over simulations
    for i in d_isimip_meta: 


        print('simulation '+str(i)+ ' of '+str(len(d_isimip_meta)))

        # load AFA data of that run
        (da_AFA, da_AFA_pic) = pk.load(open('./data/pickles/isimip_AFA_'+str(i)+'_.pkl', 'rb'))


        # Get ISIMIP GMT indices closest to GMT trajectories        
        RCP2GMT_diff_15      = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=1)
        RCP2GMT_diff_20      = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=1)
        RCP2GMT_diff_NDC     = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=1)
        RCP2GMT_diff_R26eval = np.min(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=1)

        ind_RCP2GMT_15      = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=1)
        ind_RCP2GMT_20      = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=1)
        ind_RCP2GMT_NDC     = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=1)
        ind_RCP2GMT_R26eval = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=1)

        # Get maximum T difference between RCP and GMT trajectories (to remove rows later)
        d_RCP2GMT_maxdiff_15[i]       = np.nanmax(RCP2GMT_diff_15     )
        d_RCP2GMT_maxdiff_20[i]       = np.nanmax(RCP2GMT_diff_20     )
        d_RCP2GMT_maxdiff_NDC[i]      = np.nanmax(RCP2GMT_diff_NDC    )
        d_RCP2GMT_maxdiff_R26eval[i]  = np.nanmax(RCP2GMT_diff_R26eval)



        # --------------------------------------------------------------------
        # per country 

        # initialise dicts
        d_exposure_peryear_percountry_pic = {}
        d_exposure_peryear_percountry = {}

        # get spatial average
        for country in df_countries['name']:

            # calculate mean per country weighted by population
            ind_country = countries_regions.map_keys(country)
            # corresponding picontrol - assume constant 1960 population density (this line takes about 16h by itself)
            d_exposure_peryear_percountry_pic[country] = calc_weighted_fldmean_country(da_AFA_pic, da_population[0,:,:], countries_mask, ind_country)

            # historical + RCP simulations
            d_exposure_peryear_percountry[country]     = calc_weighted_fldmean_country(da_AFA,     da_population,        countries_mask, ind_country)


        # call function to compute extreme event exposure per country and per lifetime
        d_exposure_perrun_RCP[i] = calc_life_exposure(df_life_expectancy_5, df_countries, df_birthyears, d_exposure_peryear_percountry)

        # calculate exposure for GMTs, replacing d_exposure_perrun_RCP by indexed dictionary according to corresponding GMTs with ISIMIP. 
        d_exposure_perrun_15[i]      = calc_life_exposure(df_life_expectancy_5, df_countries, df_birthyears,  {country: da[ind_RCP2GMT_15] for country, da in d_exposure_peryear_percountry.items()});
        d_exposure_perrun_20[i]      = calc_life_exposure(df_life_expectancy_5, df_countries, df_birthyears,  {country: da[ind_RCP2GMT_20] for country, da in d_exposure_peryear_percountry.items()} );
        d_exposure_perrun_NDC[i]     = calc_life_exposure(df_life_expectancy_5, df_countries, df_birthyears,  {country: da[ind_RCP2GMT_NDC] for country, da in d_exposure_peryear_percountry.items()});
        d_exposure_perrun_R26eval[i] = calc_life_exposure(df_life_expectancy_5, df_countries, df_birthyears,  {country: da[ind_RCP2GMT_R26eval] for country, da in d_exposure_peryear_percountry.items()} );

            
        # --------------------------------------------------------------------
        # per region - to add
        #  

        # save pickles
        print('Saving processed exposures')

        # pack region information
        d_exposure = {'RCP2GMT_maxdiff_15'      : d_RCP2GMT_maxdiff_15,
                    'RCP2GMT_maxdiff_20'      : d_RCP2GMT_maxdiff_20, 
                    'RCP2GMT_maxdiff_NDC'     : d_RCP2GMT_maxdiff_NDC, 
                    'RCP2GMT_maxdiff_R26eval' : d_RCP2GMT_maxdiff_R26eval, 
                    'exposure_perrun_RCP'     : d_exposure_perrun_RCP, 
                    'exposure_perrun_15'      : d_exposure_perrun_15, 
                    'exposure_perrun_20'      : d_exposure_perrun_20, 
                    'exposure_perrun_NDC'     : d_exposure_perrun_NDC, 
                    'exposure_perrun_R26eval' : d_exposure_perrun_R26eval}

        pk.dump(d_exposure,open('./data/pickles/exposure.pkl', 'wb')  )

    else: # load processed country data

        print('Loading processed exposures')

        # load country pickle
        d_exposure = pk.load(open('./data/pickles/exposure.pkl', 'rb'))

        # unpack country information
        d_RCP2GMT_maxdiff_15        = d_exposure['RCP2GMT_maxdiff_15']
        d_RCP2GMT_maxdiff_20        = d_exposure['RCP2GMT_maxdiff_20']
        d_RCP2GMT_maxdiff_NDC       = d_exposure['RCP2GMT_maxdiff_NDC']
        d_RCP2GMT_maxdiff_R26eval   = d_exposure['RCP2GMT_maxdiff_R26eval']
        d_exposure_perrun_RCP       = d_exposure['exposure_perrun_RCP']
        d_exposure_perrun_15        = d_exposure['exposure_perrun_15']
        d_exposure_perrun_20        = d_exposure['exposure_perrun_20']
        d_exposure_perrun_NDC       = d_exposure['exposure_perrun_NDC']
        d_exposure_perrun_R26eval   = d_exposure['exposure_perrun_R26eval']


# %%
