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
import cartopy.crs as ccrs
import cartopy as cr
import geopandas as gpd
import seaborn as sns
scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# flags
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'heatwavedarea' # 0: all
                                # 1: burntarea
                                # 2: cropfailedarea
                                # 3: driedarea
                                # 4: floodedarea
                                # 5: heatwavedarea
                                # 6: tropicalcyclonedarea
                                # 7: waterscarcity
flags['gmt'] = 'ar6'        # original: use Wim's stylized trajectory approach with max trajectory a linear increase to 3.5 deg                               
                            # ar6: substitute the linear max wth the highest IASA c7 scenario (increasing to ~4.0), new lower bound, and new 1.5, 2.0, NDC (2.8), 3.0
flags['rm'] = 'rm'       # no_rm: no smoothing of RCP GMTs before mapping
                         # rm: 21-year rolling mean on RCP GMTs 
flags['run'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure_trends'] = 0       # 0: do not run trend analysis on exposure for identifying regional trends (load pickle)
                                   # 1: run trend analysis
flags['lifetime_exposure'] = 0       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
                                     # 1: process ISIMIP runs to compute exposure (i.e. produce and save exposure as pickle)
flags['lifetime_exposure_cohort'] = 0       # 0: do not process ISIMIP runs to compute exposure across cohorts (i.e. load exposure pickle)
                                            # 1: process ISIMIP runs to compute exposure across cohorts (i.e. produce and save exposure as pickle)                            
flags['lifetime_exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                                     # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)
flags['emergence'] = 0      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)
flags['birthyear_emergence'] = 0    # 0: only run calc_birthyear_align with birth years from 1960-2020
                                    # 1: run calc_birthyear_align with birth years from 1960-2100                             
flags['gridscale'] = 0      # 0: do not process grid scale analysis, load pickles
                            # 1: process grid scale analysis
flags['gridscale_country_subset'] = 0      # 0: run gridscale analysis on all countries
                                           # 1: run gridscale analysis on subset of countries determined in "get_gridscale_regions" 
flags['gridscale_spatially_explicit'] = 0      # 0: do not load pickles for country lat/lon emergence (only for subset of GMTs and birth years)
                                               # 1: load those^ pickles
flags['gridscale_union'] = 0        # 0: do not process/load pickles for mean emergence and union of emergence across hazards
                                    # 1: process/load those^ pickles                                     
flags['testing'] = 0                           
flags['plot'] = 0

# TODO: add rest of flags


#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)

#%% ----------------------------------------------------------------
# load and manipulate demographic, GMT and ISIMIP data
# ----------------------------------------------------------------

from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj

df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj, GMT_indices = load_GMT(
    year_start,
    year_end,
    year_range,
    flags,
)

# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    d_countries = all_country_data()

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
da_cohort_size = d_countries['cohort_size']
countries_regions, countries_mask = d_countries['mask']    
 
# --------------------------------------------------------------------
# load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta,d_pic_meta = load_isimip(
    extremes,
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
    flags,
)

sims_per_step = {}
for step in GMT_labels:
    sims_per_step[step] = []
    print('step {}'.format(step))
    for i in list(d_isimip_meta.keys()):
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            sims_per_step[step].append(i)

#%% ----------------------------------------------------------------
# compute exposure per lifetime
# ------------------------------------------------------------------

from exposure import *

# --------------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span

if flags['exposure_trends']: 
    
    start_time = time.time()
    
    # calculate lifetime exposure per country and per region and save data
    ds_e = calc_exposure_trends(
        d_isimip_meta,
        grid_area,
        gdf_country_borders,
        flags,
    )
        
    print("--- {} minutes for original exosure computation ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )

else: # load processed exposure data

    print('Loading processed exposure trends')

    # load lifetime exposure pickle
    with open('./data/pickles/{}/exposure_trends_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_e = pk.load(f)  

# --------------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span

if flags['lifetime_exposure']: 
    
    start_time = time.time()
    
    # calculate lifetime exposure per country and per region and save data
    ds_le = calc_lifetime_exposure(
        d_isimip_meta, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5,
        flags,
    )
    
    print("--- {} minutes for original exosure computation ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )

else: # load processed exposure data

    print('Loading processed lifetime exposures')

    # load lifetime exposure pickle
    with open('./data/pickles/{}/lifetime_exposure_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_le = pk.load(f)

ds_le = calc_exposure_mmm_xr(ds_le)

# --------------------------------------------------------------------
# process lifetime exposure across cohorts

if flags['lifetime_exposure_cohort']:
    
    start_time = time.time()
    
    # calculate exposure per country and per cohort
    calc_cohort_lifetime_exposure(
        d_isimip_meta,
        df_countries,
        countries_regions,
        countries_mask,
        da_population,
        da_cohort_size,
        flags,
    )
    
    print("--- {} minutes to compute cohort exposure ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )
    
else:  # load processed cohort exposure data
    
    print('Processed exposures will be loaded in emergence calculation')

# --------------------------------------------------------------------
# process picontrol lifetime exposure

if flags['lifetime_exposure_pic']:
    
    start_time = time.time()
    
    # takes 38 mins crop failure
    d_exposure_perrun_pic = calc_lifetime_exposure_pic(
        d_pic_meta, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5, 
        flags,
    )
    
    print("--- {} minutes for PIC exposure ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )    
    
else: # load processed pic data
    
    print('Loading processed pic exposures')

    with open('./data/pickles/{}/exposure_pic_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        d_exposure_perrun_pic = pk.load(f)
    
ds_exposure_pic = calc_exposure_mmm_pic_xr(
    d_exposure_perrun_pic,
    'country',
    'pic',
)

#%% ----------------------------------------------------------------
# compute lifetime emergence
# ------------------------------------------------------------------

from emergence import *

# --------------------------------------------------------------------
# process emergence of cumulative exposures, mask cohort exposures for time steps of emergence

if flags['emergence']:
    
    if flags['birthyear_emergence']:
        
        by_emergence = np.arange(1960,2101)
        
    else:
        
        by_emergence = birth_years        
    
    if not os.path.isfile('./data/pickles/cohort_sizes.pkl'):
        
        # need new cohort dataset that has total population per birth year (using life expectancy info; each country has a different end point)
        da_cohort_aligned = calc_birthyear_align(
            da_cohort_size,
            df_life_expectancy_5,
            by_emergence,
        )
        
        # convert to dataset and add weights
        ds_cohorts = ds_cohort_align(
            da_cohort_size,
            da_cohort_aligned,
        )
        
        # pickle birth year aligned cohort sizes and global mean life expectancy
        with open('./data/pickles/cohort_sizes.pkl', 'wb') as f:
            pk.dump(ds_cohorts,f)  

    else:
        
        # load pickled birth year aligned cohort sizes and global mean life expectancy
        with open('./data/pickles/cohort_sizes.pkl', 'rb') as f:
            ds_cohorts = pk.load(f)                             
    
    ds_ae_strj, ds_pf_strj = strj_emergence(
        d_isimip_meta,
        df_life_expectancy_5,
        ds_exposure_pic,
        ds_cohorts,
        by_emergence,
        flags,
    )
        
else: # load pickles
    
    # birth year aligned population
    with open('./data/pickles/cohort_sizes.pkl', 'rb') as f:
        ds_cohorts = pk.load(f)
    
    # pop frac
    with open('./data/pickles/{}/pop_frac_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_pf_strj = pk.load(f)                
    
    # age emergence           
    with open('./data/pickles/{}/age_emergence_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_ae_strj = pk.load(f)    
                 
#%% ----------------------------------------------------------------
# grid scale
# ------------------------------------------------------------------

from gridscale import *

# list of countries to run gridscale analysis on (sometimes doing subsets across basiss/regions in floods/droughts)
gridscale_countries = get_gridscale_regions(
    grid_area,
    flags,
    gdf_country_borders,
)

# birth year aligned cohort sizes for gridscale analysis
if not os.path.isfile('./data/pickles/gs_cohort_sizes.pkl'):

    da_gs_popdenom = get_gridscale_popdenom(
        gridscale_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        da_population,
        df_life_expectancy_5,
    )

    # pickle birth year aligned cohort sizes for gridscale analysis
    with open('./data/pickles/gs_cohort_sizes.pkl', 'wb') as f:
        pk.dump(da_gs_popdenom,f)  
        
else:
    
    # load pickle birth year aligned cohort sizes for gridscale analysis
    with open('./data/pickles/gs_cohort_sizes.pkl', 'rb') as f:
        da_gs_popdenom = pk.load(f)               

if flags['gridscale']:
    
    ds_le_gs, ds_ae_gs, ds_pf_gs = gridscale_emergence(
        d_isimip_meta,
        d_pic_meta,
        flags,
        gridscale_countries,
        da_cohort_size,
        countries_regions,
        countries_mask,
        df_life_expectancy_5,
        GMT_indices,
        da_population,
    )
    
else:
    
    # load pickled aggregated lifetime exposure, age emergence and pop frac datasets
    with open('./data/pickles/{}/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_le_gs = pk.load(f)
    with open('./data/pickles/{}/gridscale_aggregated_age_emergence_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_ae_gs = pk.load(f)
    with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['extr'],flags['extr']), 'rb') as f:
        ds_pf_gs = pk.load(f)

# load spatially explicit datasets
if flags['gridscale_spatially_explicit']:
    
    d_gs_spatial = {}
    for cntry in gridscale_countries:
        with open('./data/pickles/{}/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
            d_gs_spatial[cntry] = pk.load(f)
            
# estimate union of all hazard emergences
if flags['gridscale_union']:
    
    da_emergence_mean, da_emergence_union = get_gridscale_union(
        da_population,
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )

#%% ----------------------------------------------------------------
# plot
# ------------------------------------------------------------------   

if flags['plot']:
    
    from plot import *
    
    plot_stylized_trajectories(
        df_GMT_strj,
        GMT_indices,
        d_isimip_meta,
    )    
    
    plot_sims_per_gmt(
        GMT_indices_plot,    
    )    
    
    plot_trend(
        ds_e,
        flags,
        gdf_country_borders,
        df_GMT_strj,
        GMT_indices,
        grid_area,
    )    
    
    plot_le_by_GMT_strj(
        ds_le,
        df_GMT_strj,
        ds_cohorts,
        flags
    )   
    
    plot_pf_ae_by_lines(
        ds_pf_strj,
        ds_ae_strj,
        df_GMT_strj,
        ds_cohorts,
        flags,
    )                  
    
    # heatmap of p, pf and ae for country-scale (gmt vs by)
    plot_p_pf_ae_cs_heatmap(
        ds_pf_strj,
        ds_ae_strj,
        df_GMT_strj,
        ds_cohorts,
        flags,
    )
    
    # same heatmap for for grid scale (gmt vs by)
    plot_p_pf_ae_gs_heatmap(
        ds_pf_gs,
        ds_ae_gs,
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )    
    
    # plotting unprecedented population totals between country level and gridscale level for given region 
    boxplot_cs_vs_gs_p(
        ds_pf_strj,
        ds_pf_gs,
        df_GMT_strj,
        flags,
        sims_per_step,
        gridscale_countries,
        ds_cohorts,
    )    
    
    # plotting unprecedented population fracs between country and gridscale levels for given region
    boxplot_cs_vs_gs_pf(
        ds_cohorts,
        da_gs_popdenom,
        ds_pf_strj,
        ds_pf_gs,
        df_GMT_strj,
        flags,
        sims_per_step,
        gridscale_countries,
    )    
    
    # plotting the pf and ae per sim for given GMT and birth year (using country scale)
    scatter_pf_ae_cs(
        ds_ae_strj,
        ds_cohorts,
        df_GMT_strj,
        ds_pf_strj,
        flags,
    )    
    
    # plotting the pf and ae per sim for given GMT and birth year (using grid scale)
    scatter_pf_ae_gs(
        ds_ae_gs,
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        flags,
    )        
    
    # plotting the number of simulations available per GMT step for flags['extr']    
    lineplot_simcounts(
        d_isimip_meta,
        flags,
    )
    
    # plot of heatwave heatmap, scatter plot and maps of 1, 2 and 3 degree pop unprecedented across countries
    combined_plot_hw_p(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
    )    
    
    # plot of heatwave heatmap, scatter plot and maps of 1, 2 and 3 degree pop frac unprecedented across countries - country scale
    combined_plot_hw_pf_cs(
        df_GMT_strj,
        ds_pf_strj,
        ds_cohorts,
        gdf_country_borders,
        sims_per_step,
        flags,
    )
        
    # plot of heatwave heatmap, scatter plot and maps of 1, 2 and 3 degree pop frac unprecedented across countries - gridscale
    combined_plot_hw_pf_gs(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
    )
    
    # plot change in emergence for 3 degrees between 1960 and 2020 birth cohorts
    emergence_union_plot(
        grid_area,
        da_emergence_union,
        da_emergence_mean,
    )    
    
    # combined heatmap plots of absolute pop and pop frac for all hazards computed @ grid scale
    plot_p_pf_gs_heatmap_combined(
        df_GMT_strj,
        da_gs_popdenom,
    )    
    
    # combined heatmap plots of absolute pop and pop frac for all hazards computed @ country scale
    plot_p_pf_cs_heatmap_combined(
        df_GMT_strj,
    )  
    
    # box plots of unprecedented population fracs for all hazards computed @ grid scale
    boxplot_combined_gs_pf(
        da_gs_popdenom,
        flags,   
    )
    
    # box plots of unprecedented population fracs for all hazards computed @ country scale
    boxplot_combined_cs_pf(
        ds_cohorts,
        gridscale_countries,
        flags,
    )    
    
    # box plots of unprecedented population totals for all hazards computed at grid scale
    boxplot_combined_gs_p(
        flags,   
    )   
    
    # box plots of unprecedented population totals for all hazards computed at country scale
    boxplot_combined_cs_p(
        flags,   
        gridscale_countries,
    )    
    
    # combined heatmaps across hazards of age emergence computed at grid scale
    plot_ae_gs_heatmap_combined(
        da_gs_popdenom,
    )    
    
    # combined heatmaps across hazards of age emergence computed at country scale
    plot_ae_cs_heatmap_combined(
        ds_cohorts,
    )
#%% ----------------------------------------------------------------
# concept figure
# ------------------------------------------------------------------   

# ------------------------------------------------------------------   
# get data
cntry='Belgium'
city_name='Brussels'
# cntry='Canada'
concept_bys = np.arange(1960,2021,30)
print(cntry)
da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
da_cntry = xr.DataArray(
    np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
    dims=countries_mask.dims,
    coords=countries_mask.coords,
)
da_cntry = da_cntry.where(da_cntry,drop=True)
# weights for latitude (probably won't use but will use population instead)
lat_weights = np.cos(np.deg2rad(da_cntry.lat))
lat_weights.name = "weights"   
# brussels coords  
city_lat = 50.8476
city_lon = 4.3572   
# saint john coords
# city_lat = 45.2733
# city_lon = 66.0633   

ds_spatial = xr.Dataset(
    data_vars={
        'cumulative_exposure': (
            ['run','GMT','birth_year','time','lat','lon'],
            np.full(
                (len(list(d_isimip_meta.keys())),
                 len(GMT_indices_plot),
                 len(concept_bys),
                 len(year_range),
                 len(da_cntry.lat.data),
                 len(da_cntry.lon.data)),
                fill_value=np.nan,
            ),
        ),
    },
    coords={
        'lat': ('lat', da_cntry.lat.data),
        'lon': ('lon', da_cntry.lon.data),
        'birth_year': ('birth_year', concept_bys),
        'time': ('time', year_range),
        'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
        'GMT': ('GMT', GMT_indices_plot)
    }
)

# load demography pickle
with open('./data/pickles/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
    ds_dmg = pk.load(f)                  

# loop over simulations
for i in list(d_isimip_meta.keys()): 

    print('simulation {} of {}'.format(i,len(d_isimip_meta)))

    # load AFA data of that run
    with open('./data/pickles/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
        da_AFA = pk.load(f)
        
    # mask to sample country and reduce spatial extent
    da_AFA = da_AFA.where(ds_dmg['country_extent']==1,drop=True)
    
    for step in GMT_indices_plot:
        
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            
            da_AFA_step = da_AFA.reindex(
                {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
            ).assign_coords({'time':year_range})                     
                                
            # simple lifetime exposure sum
            da_le = xr.concat(
                [(da_AFA_step.loc[{'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1)}].cumsum(dim='time') +\
                da_AFA_step.sel(time=ds_dmg['death_year'].sel(birth_year=by).item()) *\
                (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item()))\
                for by in concept_bys],
                dim='birth_year',
            ).assign_coords({'birth_year':concept_bys})
            
            da_le = da_le.reindex({'time':year_range})
            
            ds_spatial['cumulative_exposure'].loc[{
                'run':i,
                'GMT':step,
                'birth_year':concept_bys,
                'time':year_range,
                'lat':ds_dmg['country_extent'].lat.data,
                'lon':ds_dmg['country_extent'].lon.data,
            }] = da_le.loc[{
                'birth_year':concept_bys,
                'time':year_range,
                'lat':ds_dmg['country_extent'].lat.data,
                'lon':ds_dmg['country_extent'].lon.data,
            }]

# mean for brussels            
da_test_city = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').mean(dim='run')
da_test_city = da_test_city.rolling(time=5,min_periods=5).mean()

# standard deviation for brussels
da_test_city_std = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').std(dim='run')
da_test_city_std = da_test_city_std.rolling(time=5,min_periods=5).mean()

# fill in 1st 4 years with 1s
# first for mean
for by in da_test_city.birth_year.data:
    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5)}] = da_test_city.loc[{'birth_year':by,'GMT':step}].min(dim='time')
# then for std        
for by in da_test_city_std.birth_year.data:
    for step in GMT_indices_plot:
        da_test_city_std.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5)}] = da_test_city_std.loc[{'birth_year':by,'GMT':step}].min(dim='time')        

# da_test_kenya = ds_spatial['cumulative_exposure'].weighted(lat_weights).mean(('lat','lon')).mean(dim='run')
            
# load PIC pickle
with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
    ds_pic = pk.load(f)   

# plotting city lat/lon pixel doesn't give smooth kde
df_pic_city = ds_pic['lifetime_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').to_dataframe().drop(columns=['lat','lon','quantile'])         
da_pic_city_9999 = ds_pic['99.99'].sel({'lat':city_lat,'lon':city_lon},method='nearest')  


# plotting mean across kenya should be more smooth
# df_pic_kenya = ds_pic['lifetime_exposure'].weighted(lat_weights).mean(('lon','lat')).to_dataframe().drop(columns=['quantile'])   
# df_pic_kenya_9999 = ds_pic['99.99'].weighted(lat_weights).mean(('lon','lat'))       
# sns.displot(data=df_pic_kenya,kind='kde')

#%% ----------------------------------------------------------------
# concept figure
# ------------------------------------------------------------------   
 
# plot building
from mpl_toolkits.axes_grid1 import inset_locator as inset
plt.rcParams['patch.linewidth'] = 0.1
plt.rcParams['patch.edgecolor'] = 'k'
colors = dict(zip(GMT_indices_plot,['steelblue','darkgoldenrod','darkred']))
x=5
y=1

gmt_legend={
    GMT_indices_plot[0]:'1.5',
    GMT_indices_plot[1]:'2.5',
    GMT_indices_plot[2]:'3.5',
}

# ------------------------------------------------------------------   
# 1960 time series
f,ax = plt.subplots(
    figsize=(x,y)
)
for step in GMT_indices_plot:
    da_test_city.loc[{'birth_year':1960,'GMT':step}].plot.line(
        ax=ax,
        color=colors[step],
        linewidth=1
    )
end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
ax.set_title(None)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticks(np.arange(1960,2031,10))
ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
ax.annotate(
    'Born in 1960',
    (1962,ax.get_ylim()[-1]+1),
    xycoords=ax.transData,
    fontsize=10,
    rotation='horizontal',
    color='gray',
)
# ax.tick_params(colors='gray')
ax.set_xlim(
    1960,
    end_year,
)
ax.set_ylim(
    0,
    # np.round(da_test_city.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max())+1,
    da_pic_city_9999+1,
)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)    
ax.tick_params(colors='gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')
ax.hlines(
    y=da_pic_city_9999, 
    xmin=1960, 
    xmax=da_test_city.loc[{'birth_year':1960}].time.max()+10, 
    colors='grey', 
    linewidth=1, 
    linestyle='--', 
    label='99.99%', 
    zorder=1
)

# 1960 pdf
ax_pdf_l = end_year+5
ax_pdf_b = -2
ax_pdf_w = 20
ax_pdf_h = ax.get_ylim()[-1]+2
ax_pdf = ax.inset_axes(
    bounds=(ax_pdf_l, ax_pdf_b, ax_pdf_w, ax_pdf_h),
    transform=ax.transData,
)
# sns.kdeplot(
#     data=df_pic_city,
#     y='lifetime_exposure',
#     fill=True,
#     color='grey',
#     bw_adjust=5,
#     ax=ax_pdf
# )
sns.histplot(
    data=df_pic_city.round(),
    y='lifetime_exposure',
    # fill=True,
    color='lightgrey',
    discrete = True,
    ax=ax_pdf
)
ax_pdf.hlines(
    y=da_pic_city_9999, 
    xmin=0, 
    # xmax=0.4,
    xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    colors='grey', 
    linewidth=1, 
    linestyle='--', 
    label='99.99%', 
    zorder=1
)
for step in GMT_indices_plot:
    ax_pdf.hlines(
        y=da_test_city.loc[{'birth_year':1960,'GMT':step}].max(), 
        xmin=0, 
        # xmax=0.4,
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors=colors[step], 
        linewidth=1, 
        linestyle='-', 
        label=gmt_legend[step], 
        zorder=2
    )
ax_pdf.spines['right'].set_visible(False)
ax_pdf.spines['top'].set_visible(False)      
ax_pdf.set_ylabel(None)
ax_pdf.set_xlabel(None)
ax_pdf.set_ylim(-2,ax.get_ylim()[-1])
ax.tick_params(labelleft=False)    
ax_pdf.tick_params(colors='gray')
ax_pdf.spines['left'].set_color('gray')
ax_pdf.spines['bottom'].set_color('gray')
    
# ------------------------------------------------------------------       
# 1990 time series
ax2_l = 1960
# ax2_b = np.round(da_test_city.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max()) *2
ax2_b = da_pic_city_9999 *2
ax2_w = 1990-1960+np.floor(df_life_expectancy_5.loc[1990,cntry])
ax2_h = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())
ax2 = ax.inset_axes(
    bounds=(ax2_l, ax2_b, ax2_w, ax2_h),
    transform=ax.transData,
)

for step in GMT_indices_plot:
    da_test_city.loc[{'birth_year':1990,'GMT':step}].plot.line(
        ax=ax2,
        color=colors[step],
        linewidth=1,
    )
end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
ax2.set_title(None)
ax2.set_ylabel(None)
ax2.set_xlabel(None)
ax2.set_xlim(
    1960,
    end_year,
)
ax2.set_ylim(
    0,
    np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())+1,
)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)  
ax2.spines['left'].set_position(('data',1990))
ax2.tick_params(labelleft=False)    
ax2.tick_params(colors='gray')
ax2.spines['left'].set_color('gray')
ax2.spines['bottom'].set_color('gray')
ax2.hlines(
    y=da_pic_city_9999, 
    xmin=1990, 
    xmax=da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[0]}].time.\
        where(np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}])==np.round(da_pic_city_9999)).min()-3.1, 
    colors='grey', 
    linewidth=1, 
    linestyle='--', 
    label='99.99%', 
    zorder=1
)
ax2.annotate(
    'Born in 1990',
    (1992,ax2.get_ylim()[-1]-4),
    xycoords=ax2.transData,
    fontsize=10,
    rotation='horizontal',
    color='gray',
)

# 1990 pdf
ax2_pdf_l = end_year+5
ax2_pdf_b = -2
ax2_pdf_w = 20
# ax2_pdf_h = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()+2)
ax2_pdf_h = ax2.get_ylim()[-1]+2
ax2_pdf = ax2.inset_axes(
    bounds=(ax2_pdf_l, ax2_pdf_b, ax2_pdf_w, ax2_pdf_h),
    transform=ax2.transData,
)
# sns.kdeplot(
#     data=df_pic_city,
#     y='lifetime_exposure',
#     fill=True,
#     color='grey',
#     bw_adjust=5,
#     ax=ax2_pdf
# )
sns.histplot(
    data=df_pic_city.round(),
    y='lifetime_exposure',
    # fill=True,
    color='lightgrey',
    discrete = True,
    ax=ax2_pdf
)
ax2_pdf.hlines(
    y=da_pic_city_9999, 
    xmin=0, 
    # xmax=0.4,
    xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    colors='grey', 
    linewidth=1, 
    linestyle='--', 
    label='99.99%', 
    zorder=1
)
for step in GMT_indices_plot:
    ax2_pdf.hlines(
        y=da_test_city.loc[{'birth_year':1990,'GMT':step}].max(), 
        xmin=0, 
        # xmax=0.4, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors=colors[step], 
        linewidth=1, 
        linestyle='-', 
        label=gmt_legend[step], 
        zorder=2
    )
ax2_pdf.spines['right'].set_visible(False)
ax2_pdf.spines['top'].set_visible(False)      
ax2_pdf.set_ylabel(None)
ax2_pdf.set_xlabel(None)
ax2_pdf.set_ylim(-2,ax2.get_ylim()[-1])
ax2_pdf.tick_params(colors='gray')
ax2_pdf.spines['left'].set_color('gray')
ax2_pdf.spines['bottom'].set_color('gray')

# ------------------------------------------------------------------   
# 2020 time series
ax3_l = 1960
ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
ax3_w = 2020-1960+np.floor(df_life_expectancy_5.loc[2020,cntry])
ax3_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())
ax3 = ax2.inset_axes(
    bounds=(ax3_l, ax3_b, ax3_w, ax3_h),
    transform=ax2.transData,
)
# plot mean lines
for step in GMT_indices_plot:
    da_test_city.loc[{'birth_year':2020,'GMT':step}].plot.line(
        ax=ax3,
        color=colors[step],
        linewidth=1
    )
# plot std uncertainty bars
# for step in GMT_indices_plot:
#     ax3.fill_between(
#         x=da_test_city_std.loc[{'birth_year':2020,'GMT':step}].time.data,
#         y1=da_test_city.loc[{'birth_year':2020,'GMT':step}] + da_test_city_std.loc[{'birth_year':2020,'GMT':step}],
#         y2=da_test_city.loc[{'birth_year':2020,'GMT':step}] - da_test_city_std.loc[{'birth_year':2020,'GMT':step}],
#         color=colors[step],
#         alpha=0.2
#     )

end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])
ax3.set_title(None)
ax3.set_ylabel(None)
ax3.set_xlabel(None)
ax3.set_xlim(
    1960,
    end_year,
)
ax3.set_ylim(
    0,
    np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)  
ax3.spines['left'].set_position(('data',2020))
ax3.tick_params(labelleft=False)    
ax3.tick_params(colors='gray')
ax3.spines['left'].set_color('gray')
ax3.spines['bottom'].set_color('gray')
ax3.hlines(
    y=da_pic_city_9999, 
    xmin=2020, 
    # xmax=da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}].time.where(), 
    xmax=da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}].time.\
        where(np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}])==np.round(da_pic_city_9999)).min()+1,
    colors='grey', 
    linewidth=1, 
    linestyle='--', 
    label='99.99%', 
    zorder=1
)
ax3.annotate(
    'Born in 2020',
    (2022,ax3.get_ylim()[-1]-10),
    xycoords=ax3.transData,
    fontsize=10,
    rotation='horizontal',
    color='gray',
)

# 2020 pdf
ax3_pdf_l = end_year+5
ax3_pdf_b = -2
ax3_pdf_w = 20
# ax3_pdf_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max()+2)
ax3_pdf_h = ax3.get_ylim()[-1]+2
ax3_pdf = ax3.inset_axes(
    bounds=(ax3_pdf_l, ax3_pdf_b, ax3_pdf_w, ax3_pdf_h),
    transform=ax3.transData,
)
# sns.kdeplot(
#     data=df_pic_city,
#     y='lifetime_exposure',
#     fill=True,
#     color='grey',
#     bw_adjust=5,
#     cut=0,
#     ax=ax3_pdf
# )
sns.histplot(
    data=df_pic_city.round(),
    y='lifetime_exposure',
    # fill=True,
    color='lightgrey',
    discrete = True,
    ax=ax3_pdf
)
ax3_pdf.hlines(
    y=da_pic_city_9999, 
    xmin=0, 
    # xmax=0.4, 
    xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    colors='grey', 
    linewidth=1, 
    linestyle='--', 
    label='99.99%', 
    zorder=1
)
for step in GMT_indices_plot:
    ax3_pdf.hlines(
        y=da_test_city.loc[{'birth_year':2020,'GMT':step}].max(), 
        xmin=0, 
        # xmax=0.4, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors=colors[step], 
        linewidth=1, 
        linestyle='-', 
        label=gmt_legend[step], 
        zorder=2
    )
ax3_pdf.spines['right'].set_visible(False)
ax3_pdf.spines['top'].set_visible(False)      
ax3_pdf.set_ylabel(None)
ax3_pdf.set_xlabel(None)
ax3_pdf.set_ylim(-2,ax3.get_ylim()[-1])
ax3_pdf.tick_params(colors='gray')
ax3_pdf.spines['left'].set_color('gray')
ax3_pdf.spines['bottom'].set_color('gray')

# City name
ax3.annotate(
    '{}, \n{}'.format(city_name,cntry),
    (1980,15),
    xycoords=ax3.transData,
    fontsize=16,
    rotation='horizontal',
    color='gray',
)

# axis labels ===================================================================

# x axis label (time)
x_i=1950
y_i=-10
x_f=2040
y_f=y_i 
con = ConnectionPatch(
    xyA=(x_i,y_i),
    xyB=(x_f,y_f),
    coordsA=ax.transData,
    coordsB=ax.transData,
    color='gray',
)
ax.add_artist(con)   

con_arrow_top = ConnectionPatch(
    xyA=(x_f-2,y_f+1),
    xyB=(x_f,y_f),
    coordsA=ax.transData,
    coordsB=ax.transData,
    color='gray',
)
ax.add_artist(con_arrow_top)  

con_arrow_bottom = ConnectionPatch(
    xyA=(x_f-2,y_f-1),
    xyB=(x_f,y_f),
    coordsA=ax.transData,
    coordsB=ax.transData,
    color='gray',
)
ax.add_artist(con_arrow_bottom) 
ax.annotate(
    'Time',
    ((x_i+x_f)/2,y_f+1),
    xycoords=ax.transData,
    fontsize=12,
    color='gray',
)

# y axis label (Cumulative heatwave exposure since birth)
x_i=1950
y_i=-10
x_f=x_i
y_f=y_i + 50
con = ConnectionPatch(
    xyA=(x_i,y_i),
    xyB=(x_f,y_f),
    coordsA=ax.transData,
    coordsB=ax.transData,
    color='gray',
)
ax.add_artist(con)   

con_arrow_left = ConnectionPatch(
    xyA=(x_f-2,y_f-1),
    xyB=(x_f,y_f),
    coordsA=ax.transData,
    coordsB=ax.transData,
    color='gray',
)
ax.add_artist(con_arrow_left)  

con_arrow_right = ConnectionPatch(
    xyA=(x_f+2,y_f-1),
    xyB=(x_f,y_f),
    coordsA=ax.transData,
    coordsB=ax.transData,
    color='gray',
)
ax.add_artist(con_arrow_right) 

ax.annotate(
    'Cumulative heatwave exposure since birth',
    (x_i-10,(y_i+y_f)/5),
    xycoords=ax.transData,
    fontsize=12,
    rotation='vertical',
    color='gray',
)

# legend ===================================================================

# bbox
x0 = 1.5
y0 = -1.3
xlen = 0.5
ylen = 0.5

# space between entries
legend_entrypad = 0.5

# length per entry
legend_entrylen = 0.75

legend_font = 10
legend_lw=1
   
legendcols = list(colors.values())+['gray']+['lightgrey']
handles = [
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2]),
    Line2D([0],[0],linestyle='--',lw=legend_lw,color=legendcols[3]),
    Rectangle((0,0),1,1,color=legendcols[4]),
]
labels= [
    '1.5 °C GMT warming by 2100',
    '2.5 °C GMT warming by 2100',
    '3.5 °C GMT warming by 2100',
    '99.99% pre-industrial \n lifetime exposure',
    'pre-industrial lifetime \n exposure histogram'
]
ax.legend(
    handles, 
    labels, 
    bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
    loc=3,
    ncol=1,
    fontsize=legend_font, 
    mode="expand", 
    borderaxespad=0.,
    frameon=False, 
    columnspacing=0.05, 
)      

f.savefig('./figures/concept.png',dpi=900,bbox_inches='tight')


#%% ----------------------------------------------------------------
# age emergence & pop frac testing
# ------------------------------------------------------------------        

if flags['testing']:
    
    pass

                        

# %%
