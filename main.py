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

# To do
# Put all GDP/deprivation loading (GDP and GDP per capita for max time length of product in function under Emergence section)
    # run read in of isimip hist + rcp26 
    # pip install py-cdo; write CDO python lines to run remapcon2 if isimip grid Wang and Sun files not there
    # run read in of wang and sung;
    # pickle datasets
# Make lifetime gdp function (birth cohort choice in settings)
    # currently have sum option, need to add mean (don't worry about fractional last year of life span)
    # pickle datasets
# Cross lifetime gdp against emergence masks 
    # get pop estimates for sample of percentiles for each GDP product (2)
# Put all deprivation conversion/loading in function and run it in emergence section
    # pickle data
# Cross deprivation against emergence masks
    # get pop estimates for sample percentiles for deprivation
    
    
    


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
import seaborn as sns # must comment this out for things to work on the server
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
flags['gmt'] = 'ar6_new'    # original: use Wim's stylized trajectory approach with max trajectory a linear increase to 3.5 deg                               
                            # ar6: substitute the linear max wth the highest IASA c7 scenario (increasing to ~4.0), new lower bound, and new 1.5, 2.0, NDC (2.8), 3.0
                            # ar6_new: works off ar6, but ensures only 1.5-3.5 with perfect intervals of 0.1 degrees (less proc time and data volume)
flags['rm'] = 'rm'       # no_rm: no smoothing of RCP GMTs before mapping
                         # rm: 21-year rolling mean on RCP GMTs 
flags['version'] = 'pickles_v2'     # pickles: original version, submitted to Nature
                                        # inconsistent GMT steps (not perfect 0.1 degree intervals)
                                        # GMT steps ranging 1-4 (although study only shows ~1.5-3.5, so runs are inefficient)
                                        # only 99.99% percentile for PIC threshold
                                    # pickles_v2: version generated after submission to Nature in preparation for criticism/review
                                        # steps fixed in load_manip to be only 1.5-3.5, with clean 0.1 degree intervals
                                        # 5 percentiles for PIC threshold and emergence for each
                         # rm: 21-year rolling mean on RCP GMTs                          
flags['run'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
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
flags['gridscale_union'] = 0        # 0: do not process/load pickles for mean emergence and union of emergence across hazards
                                    # 1: process/load those^ pickles    
flags['global_emergence'] = 1       # 0: do not load pickles of global emergence masks
                                    # 1: load pickles                                                                               
flags['gdp_deprivation'] = 0        # 0: do not process/load lifetime GDP/GRDI average
                                    # 1: load lifetime GDP average analysis        
flags['vulnerability'] = 0          # 0: do not process subsets of d_collect_emergence vs gdp & deprivation quantiles
                                    # 1: process/load d_collect_emergence vs gdp & deprivation quantiles for vulnerability analysis
flags['plot_ms'] = 0 # 1 yes plot, 0 no plot
flags['plot_si'] = 0
flags['reporting'] = 0     



#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

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

    d_countries = all_country_data(flags)

else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/{}/country_info.pkl'.format(flags['version']), 'rb'))
    
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
    for i in list(d_isimip_meta.keys()):
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            sims_per_step[step].append(i)

#%% ----------------------------------------------------------------
# compute exposure per lifetime at country-scale
# ------------------------------------------------------------------

from exposure import *

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

#     with open('./data/{}/{}/exposure_pic_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
#         d_exposure_perrun_pic = pk.load(f)
    
# ds_exposure_pic = calc_exposure_mmm_pic_xr(
#     d_exposure_perrun_pic,
#     'country',
#     'pic',
# )

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
    
    if not os.path.isfile('./data/{}/cohort_sizes.pkl'.format(flags['version'])):
        
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
        with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_cohorts,f)  

    else:
        
        # load pickled birth year aligned cohort sizes and global mean life expectancy
        with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
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
    
    pass
    
    # # birth year aligned population
    # with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
    #     ds_cohorts = pk.load(f)
    
    # # pop frac
    # with open('./data/{}/{}/pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
    #     ds_pf_strj = pk.load(f)                
    
                 
#%% ----------------------------------------------------------------
# grid scale emergence
# ------------------------------------------------------------------

from gridscale import *

# list of countries to run gridscale analysis on (sometimes doing subsets across basiss/regions in floods/droughts)
gridscale_countries = get_gridscale_regions(
    grid_area,
    flags,
    gdf_country_borders,
)

# birth year aligned cohort sizes for gridscale analysis (summed over lat/lon per country)
if not os.path.isfile('./data/{}/gs_cohort_sizes.pkl'.format(flags['version'])):

    da_gs_popdenom = get_gridscale_popdenom(
        gridscale_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        da_population,
        df_life_expectancy_5,
    )

    # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
        pk.dump(da_gs_popdenom,f)  
        
else:
    
    # load pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
        da_gs_popdenom = pk.load(f)               

# run gridscale emergence analysis
if flags['gridscale']:
    
    ds_le_gs, ds_pf_gs = gridscale_emergence(
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
    with open('./data/{}/{}/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
        ds_le_gs = pk.load(f)
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
        ds_pf_gs = pk.load(f)
            
# # estimate union of all hazard emergences
# if flags['gridscale_union']:
    
#     da_emergence_mean, da_emergence_union = get_gridscale_union(
#         da_population,
#         flags,
#         gridscale_countries,
#         countries_mask,
#         countries_regions,
#     )

# read in global emergence masks
if flags['global_emergence']:
    
    d_global_emergence = collect_global_emergence(
        grid_area,
        flags,
        countries_mask,
        countries_regions,
        gridscale_countries,
        df_GMT_strj,
    )
    
# load/proc GDP and deprivation data
if flags['gdp_deprivation']:
    
    ds_gdp, ds_grdi = load_gdp_deprivation(
        flags,
        grid_area,
        gridscale_countries,
        df_life_expectancy_5,
    )
    
# vulnerability subsetting
if flags['vulnerability']:
    
    # first need global map of 2020 birth cohort sizes, spatially explicit
    if not os.path.isfile('./data/{}/2020_cohort_sizes.pkl'.format(flags['version'])):
        
        da_cohort_size_2020 = xr.full_like(countries_mask,fill_value=np.nan)
        
        for cntry in gridscale_countries:
            
            da_cntry = countries_mask.where(countries_mask == countries_regions.map_keys(cntry))
            cntry_cohort_frac = da_cohort_size.sel(country=cntry,time=2020,ages=0) / da_cohort_size.sel(country=cntry,time=2020).sum(dim='ages')
            da_cohort_size_2020 = xr.where(
                da_cntry.notnull(),
                da_population.sel(time=2020).where(da_cntry.notnull()) * cntry_cohort_frac,
                da_cohort_size_2020
            )
            
        with open('./data/{}/2020_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(da_cohort_size_2020,f)
        
    else:
        
        with open('./data/{}/2020_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
            da_cohort_size_2020 = pk.load(f)     
            
    # same thing but repeat for all years in birth cohort assessmet (1960-2020)
    if not os.path.isfile('./data/{}/1960-2020_cohort_sizes.pkl'.format(flags['version'])):
        
        da_cohort_size_1960_2020 = xr.concat(
            [xr.full_like(countries_mask,fill_value=np.nan) for by in birth_years],
            dim='birth_year'
        ).assign_coords({'birth_year':birth_years})
        
        for by in birth_years:
            
            for cntry in gridscale_countries:
                
                da_cntry = countries_mask.where(countries_mask == countries_regions.map_keys(cntry))
                cntry_cohort_frac = da_cohort_size.sel(country=cntry,time=by,ages=0) / da_cohort_size.sel(country=cntry,time=by).sum(dim='ages')
                da_cohort_size_1960_2020.loc[{'birth_year':by}] = xr.where(
                    da_cntry.notnull(),
                    da_population.sel(time=by).where(da_cntry.notnull()) * cntry_cohort_frac,
                    da_cohort_size_1960_2020.loc[{'birth_year':by}],
                )
            
        with open('./data/{}/1960-2020_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(da_cohort_size_1960_2020,f)
        
    else:
        
        with open('./data/{}/1960-2020_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
            da_cohort_size_1960_2020 = pk.load(f)     
            
# vulnerability dataset
extremes = [
    'burntarea', 
    'cropfailedarea', 
    'driedarea', 
    'floodedarea', 
    'heatwavedarea', 
    'tropicalcyclonedarea',
]
    # qntls_vulnerability = [0.1,0.25,0.49,0.51,0.75,0.9] # change to 10 quantiles of 10%


qntls_vulnerability = np.arange(0.,1.01,0.1)
# testing ranking of pixels for lifetime mean gdp
# how to rank order population pixels by gdp?
# mask of rank integers (by gdp)
    # get 
    # bin the ranks
xr.groupby_bins()
test.sortby(dim='lat')
test.rank(('lat'))
# for q in qntls_vulnerability:
np.reshape()
ds_gdp['gdp_isimip_rcp26_mean']
test = ds_gdp['gdp_isimip_rcp26_mean'].sel(birth_year=2020)
np.sort(test.values.flatten())
np.unique(np.sort(test.values.flatten()))
i = 0
f = i+0.1
q_i = test.quantile(i,dim=('lat','lon'),method='linear')
q_f = test.quantile(f,dim=('lat','lon'),method='linear')


# good ex of ranking:
arr = xr.DataArray([[4, 3, 6],[1,7,9]], dims=("y","x"))
print(arr)
arr2 = xr.DataArray(arr.values.flatten())
print(arr2)
print(arr2.rank(dim='dim_0'))
arr3 = xr.DataArray(np.reshape(arr2.rank(dim='dim_0').values,newshape=(len(arr.y),len(arr.x))), dims=("y","x"))
print(arr3)

# working ex on how to run a ranking; now need to figure how to 

# first make sure gdp and 
gdp = ds_gdp['gdp_isimip_rcp26_mean'].sel(birth_year=2020)
pop = da_cohort_size_1960_2020.sel(birth_year=2020)

gdp = gdp.where(pop.notnull())
pop = pop.where(gdp.notnull())

# check that this worked by seeing len of non-nans
if len(xr.DataArray(gdp.values.flatten())) == len(xr.DataArray(pop.values.flatten())):
    print('should only be using overlapping grid cells')

vulnerability = xr.DataArray(gdp.values.flatten())
vulnerability_ranks = xr.DataArray(gdp.values.flatten()).rank(dim='dim_0').round()
vulnerability_indices = vulnerability_ranks.dim_0
sorted_vi = vulnerability_indices.sortby(vulnerability_ranks) # puts nans at back
sorted_vr = vulnerability_ranks.sortby(vulnerability_ranks) # puts nans at back

pop_flat = xr.DataArray(pop.values.flatten())
sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because gdp and pop need common mask
sorted_pop_cumsum = sorted_pop.cumsum()
sorted_pop_cumsum_pct = sorted_pop_cumsum / sorted_pop.sum()
vulnerability_binned = vulnerability.groupby_bins(sorted_pop_cumsum_pct,bins=10)
testpop_binned = sorted_pop.groupby_bins(sorted_pop_cumsum_pct,bins=10) # sums here are pretty close!
# now, how to go from binned vulnerbility from flat array back to map shape

np.nanmin(vulnerability_ranks)

test = ds_gdp['gdp_isimip_rcp26_mean'].sel(birth_year=2020)
test2 = xr.DataArray(test.values.flatten()).rank(dim='dim_0') # 1D array of ranks of lifetime gdp means
test3 = xr.DataArray( # reshape rank array
    np.reshape(test2.values,newshape=(len(test.lat),len(test.lon))),
    coords={
        'lat': ('lat', test.lat.data),
        'lon': ('lon', test.lon.data),
    }
)

testp2 = xr.DataArray(da_cohort_size_1960_2020.sel(birth_year=2020).values.flatten())
test4 = test3.groupby_bins(
    group=da_cohort_size_1960_2020.sel(birth_year=2020),
    bins=10,
    right=True
)



test_qif = xr.where(
    (test < q_f) & (test >=q_i),
    test,
    np.nan
)
test_qif.sel(birth_year=2020).plot()

for q in qntls_vulnerability:    
    
    indices_vulnerability = ['gdp_isimip_rcp26_mean','grdi'] # instead just rcp26 and grdi
    # for v in ds_gdp.data_vars:
    #     indices_vulnerability.append(v)
    # for v in ds_grdi.data_vars:
    #     indices_vulnerability.append(v)
    all_runs=np.arange(1,87)
    ds_vulnerability = xr.Dataset(
        data_vars={
            'heatwavedarea': (
                ['run','qntl','vulnerability_index', 'birth_year'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'cropfailedarea': (
                ['run','qntl','vulnerability_index'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),            
            'floodedarea': (
                ['run','qntl','vulnerability_index'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'burntarea': (
                ['run','qntl','vulnerability_index'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'driedarea': (
                ['run','qntl','vulnerability_index'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),            
            'floodedarea': (
                ['run','qntl','vulnerability_index'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),             
            'tropicalcyclonedarea': (
                ['run','qntl','vulnerability_index'],
                np.full(
                    (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),                                           
        },
        coords={
            'run': ('run', all_runs),
            'qntl': ('qntl', qntls_vulnerability),
            'vulnerability_index': ('vulnerability_index', indices_vulnerability),
            'birth_year': ('birth_year', birth_years)
        }
    )
    # test = d_global_emergence['heatwavedarea']['2.7']['emergence_per_run_heatwavedarea'].sel(qntl='99.9',birth_year=2020).copy(deep=True)
    # ds_test = test.to_dataset()
    for e in extremes:
        for v in list(ds_gdp.data_vars):
            for q in qntls_vulnerability:
                for by in birth_years:
                    qntl_gdp = ds_gdp[v].loc[{'birth_year':by}].quantile(
                        q,
                        dim=('lat','lon'),
                        method='closest_observation',
                    )         
                    if q > 0.5:   
                        da_mask_gdp_group = xr.where(ds_gdp[v]>=qntl_gdp.item(),1,np.nan)
                    elif q < 0.5:
                        da_mask_gdp_group = xr.where(ds_gdp[v]<=qntl_gdp.item(),1,np.nan)
                    for r in d_global_emergence[e]['2.7']['emergence_per_run_{}'.format(e)].run.data:
                        da_emerge = d_global_emergence[e]['2.7']['emergence_per_run_{}'.format(e)].sel(qntl='99.9',birth_year=2020,run=r)
                        da_emerge_constrained = da_emerge.where(da_mask_gdp_group.notnull())
                        ds_vulnerability['{}'.format(e)].loc[{'run':r,'qntl':q,'vulnerability_index':v}] = xr.where(
                            da_emerge_constrained == 1,
                            da_cohort_size_2020,
                            0
                        ).sum(dim=('lat','lon'))
        for v in list(ds_grdi.data_vars):
            for q in qntls_vulnerability:
                qntl_grdi = ds_grdi[v].quantile(
                    q,
                    dim=('lat','lon'),
                    method='closest_observation',
                )    
                if q > 0.5:   
                    da_mask_grdi_group = xr.where(ds_grdi[v]>=qntl_grdi.item(),1,np.nan)
                elif q < 0.5:
                    da_mask_grdi_group = xr.where(ds_grdi[v]<=qntl_grdi.item(),1,np.nan)
                for r in d_global_emergence[e]['2.7']['emergence_per_run_{}'.format(e)].run.data:
                    da_emerge = d_global_emergence[e]['2.7']['emergence_per_run_{}'.format(e)].sel(qntl='99.9',birth_year=2020,run=r)
                    da_emerge_constrained = da_emerge.where(da_mask_grdi_group.notnull())
                    ds_vulnerability['{}'.format(e)].loc[{'run':r,'qntl':q,'vulnerability_index':v}] = xr.where(
                        da_emerge_constrained == 1,
                        da_cohort_size_2020,
                        0
                    ).sum(dim=('lat','lon'))                                  
            
# convert to millions people
# 6 rows for extremes
# 7 columns for vulnerability index
df_vulnerability = ds_vulnerability.to_dataframe().reset_index()      
for e in extremes:
    print(e)
    f,axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(16,14)
    )
    df_vulnerability_e = df_vulnerability.loc[:,['run','qntl','vulnerability_index',e]]
    df_vulnerability_e.loc[:,e] = df_vulnerability_e.loc[:,e] / 10**6 # convert to millions of people
    for ax,v in zip(axes.flatten(),indices_vulnerability):
        # ax.set_title()
        df_vulnerability_v = df_vulnerability_e[df_vulnerability_e['vulnerability_index']==v]
        df_vulnerability_v.boxplot(column=e,by='qntl',ax=ax)
        ax.set_title(None)
        ax.set_title(v)
        ax.set_xlabel(None)
    f.delaxes(axes[-1][-1])
    plt.show()
    




 

#%% ----------------------------------------------------------------
# gdp analysis; Wang & Sun 
# ------------------------------------------------------------------    


#%% ----------------------------------------------------------------
# main text plots
# ------------------------------------------------------------------       

if flags['plot_ms']:

    from plot_ms import *

    # f1 of ms, conceptual figure of city grid cell
    plot_conceptual(
        da_cohort_size,
        countries_mask,
        countries_regions,
        d_isimip_meta,
        flags,
        df_life_expectancy_5,
    )

    # f2 of ms, combined heatwave plot
    plot_combined_piechart(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
        df_countries,
    )
    
    # # f2 alternative with both absolute pops below box plots and pie charts
    # plot_combined_population_piechart(
    #     df_GMT_strj,
    #     ds_pf_gs,
    #     da_gs_popdenom,
    #     gdf_country_borders,
    #     sims_per_step,
    #     flags,
    #     df_countries,
    # )    
    
    # # f2 alternative with absolute pops below box plots and no pie charts
    # plot_combined_population(
    #     df_GMT_strj,
    #     ds_pf_gs,
    #     da_gs_popdenom,
    #     gdf_country_borders,
    #     sims_per_step,
    #     flags,
    #     df_countries,
    # )        

    # f3 of heatmaps across all hazards
    plot_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )

    # f4 of emergence union plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_emergence_union(
        grid_area,
        da_emergence_mean,
    )

    # f4 alternative for hexagons and multiple thresholds
    plot_hexagon_multithreshold(
        d_global_emergence,
    )    

#%% ----------------------------------------------------------------
# supplementary text plots
# ------------------------------------------------------------------  

if flags['plot_si']:

    from plot_si import *

    # pf time series for 2020 birth year across GMTs
    plot_pf_gmt_tseries_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )
    
    # pf time series for 2.7 degree world across birth years
    plot_pf_by_tseries_allhazards(
        flags,
        df_GMT_strj,
        da_gs_popdenom,
    )   
    
    # pf box plots for 1.5, 2.5 and 3.5 degree world across birth years
    plot_boxplots_allhazards(
        da_gs_popdenom,
        df_GMT_strj,
        flags,
    )     
    
    # pf maps for 1..5, 2.5, 3.5 for all hazards
    plot_pf_maps_allhazards(
        da_gs_popdenom,
        gdf_country_borders,
        flags,
    )    
    
    # emergence fraction plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_emergence_fracs(
        grid_area,
        da_emergence_mean,
    )    
    
    # plot locations where exposure occurs at all in our dataset
    plot_exposure_locations(
        grid_area,
        countries_mask,
        flags,
    )    
    
    # plot tseries box plots for 1.5, 2.5 and 3.5 when denominator contrained by exposure extent
    plot_geoconstrained_boxplots(
        flags,
    )    

    # plot gmt pathways of rcps and ar6
    plot_gmt_pathways(
        df_GMT_strj,
        d_isimip_meta,
    )
    
    # plot heatmaps of pf for country level emergence
    plot_heatmaps_allhazards_countryemergence(
        df_GMT_strj,
        flags,
    )     
    
    # plot pie charts of all hazards
    plot_allhazards_piecharts(
        da_gs_popdenom,
        df_countries,
        flags,
    )
    
    # plot cohort sizes in stacked bar chart
    plot_cohort_sizes(
        df_countries,
        da_gs_popdenom,
    )    
    
    # plot hexagon landfracs (will change to only show landfracs for SI)
    plot_hexagon_landfrac(
        d_global_emergence,
    )    
    

#%% ----------------------------------------------------------------
# sample analytics for paper
# ------------------------------------------------------------------

if flags['reporting']:
    
    from reporting import *
    
    # estimates of land area and (potential) pf for 1960 and 2020 emergencve of multiple hazards
    multi_hazard_emergence(
        grid_area,
        da_emergence_mean,
        da_gs_popdenom,
    )
    
    # get birth year cohort sizes at grid scale
    gridscale_cohort_sizes(
        da_population,
        gridscale_countries,   
    )    
    
    # per hazard, locations where exposure occurs across whole ensemble
    exposure_locs(
        grid_area,
    )
    
    # per run for 1.5, 2.5, 2.7 and 3.5, collect maps of emergence locations to be used in geographically constrained pf estimates
    emergence_locs_perrun(
        flags,
        grid_area,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )    
    
    # compute geographically constrained pf
    pf_geoconstrained()
    
    # print geographically constrained pf vs regular pf
    print_pf_geoconstrained(
        flags,
        da_gs_popdenom,
    )    

    # checking for signifiance of change in means between 1960 and 2020 pf per event and for a GMT level
    paired_ttest(
        flags,
        da_gs_popdenom,
    )
    
    # print latex table on ensemble members per hazard
    print_latex_table_ensemble_sizes(
        flags,
        df_GMT_strj,
    )   
    
    # children living unprec exposure between 1.5 and 2.7 degrees warming
    print_millions_excess(
        flags,
        df_GMT_strj,
    )     

    # print pf info    
    print_pf_ratios(
        df_GMT_strj,
        da_gs_popdenom,
    )    
    
    # get number of million people unprecedented: (will change this stuff to run for all extremes and birth years for table in paper)
    print_absolute_unprecedented(
        ds_pf_gs,
    )
 
    # get cities that work for 
    find_valid_cities(
        df_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        d_isimip_meta,
        flags,
    )
    
    # latex tables of CF per extr and GMT pathway
    print_latex_table_unprecedented(
        flags,
        da_gs_popdenom,
    )    
    
    print_latex_table_unprecedented_sideways(
        flags,
        da_gs_popdenom,
    )    
