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
flags['extr']  = 'all'  # 0: all
                        # 1: burntarea
                        # 2: cropfailedarea
                        # 3: driedarea
                        # 4: floodedarea
                        # 5: heatwavedarea
                        # 6: tropicalcyclonedarea
                        # 7: waterscarcity

flags['runs']  = 0      # 0: do not process ISIMIP runs (i.e. load runs workspace)
                        # 1: process ISIMIP runs (i.e. produce and save runs as workspace)
flags['mask'] = 0;      # 0: do not process country data (i.e. load masks workspace)
                        # 1: process country data (i.e. produce and save masks as workspace)
# TODO: add rest of flags


#%% ----------------------------------------------------------------
# INITIALISE
# ----------------------------------------------------------------
from settings import *

# set global variables
init()

# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
global extremes
extremes        = flags['extr']


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


    # --------------------------------------------------------------------
    # Load population and country masks, and mask population per country
    # Load SSP population totals 
    da_population = load_population(year_start,year_end)


    # load country borders
    gdf_country_borders = gpd.read_file('./data/natural_earth/Cultural_10m/Countries/ne_10m_admin_0_countries.shp'); 


    # mask population totals per country  and save country regions object and countries mask
    df_countries, countries_regions, countries_mask = get_mask_population(da_population, gdf_country_borders, df_countries)


    # pack country information
    d_countries = {'info_pop' : df_countries, 
                   'birth_years': df_birthyears,
                   'life_expectancy_5': df_life_expectancy_5, 
                   'cohort_size': d_cohort_size, 
                   'mask': (countries_regions,countries_mask)}

    # save metadata dictionary as a pickle
    print('Saving country info')

    pk.dump(d_countries,open('./data/pickles/country_info.pkl', 'wb')  )


else: # load processed country data

    print('Loading processed country info')

    d_countries = pk.load(open('./data/pickles/country_info.pkl', 'rb'))

    df_countries                      = d_countries['info_pop']
    df_birthyears                     = d_countries['birth_years']
    df_life_expectancy_5              = d_countries['life_expectancy_5']
    d_cohort_size                     = d_countries['cohort_size']
    countries_regions, countries_mask = d_countries['mask']




# --------------------------------------------------------------------
# Load ISIMIP model data

grid_area                = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta = load_isimip(flags['runs'], extremes, model_names)



#%%

#%% ----------------------------------------------------------------
# COMPUTE EXPOSURE PER LIFETIME
# ----------------------------------------------------------------