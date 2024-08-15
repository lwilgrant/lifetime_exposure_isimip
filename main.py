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

#               
#%%  ----------------------------------------------------------------
# import and path
# ----------------------------------------------------------------

import xarray as xr
import pickle as pk
import time
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cr
import geopandas as gpd
# import seaborn as sns # must comment this out for things to work on the server
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
                                    # pickles_v3: version generated after the 2021 toolchains were taken away from hydra. could not longer use old pickles effectively
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
flags['gridscale'] = 0     # 0: do not process grid scale analysis, load pickles
                            # 1: process grid scale analysis
flags['gridscale_le_test'] = 0      # 0: do not process the grid scale analysis testing diff versions of constant life expectancy
                                    # 1: process grid scale analysis testing diff versions of constant life expectancy                             
flags['gridscale_country_subset'] = 0      # 0: run gridscale analysis on all countries
                                           # 1: run gridscale analysis on subset of countries determined in "get_gridscale_regions" 
flags['global_emergence_recollect'] = 0        # 0: do not process or load pickles of global emergence masks
                                    # 1: process or load pickles if they're present (note that pickles are huge on hydra)
flags['global_avg_emergence'] = 0                                                                                                
flags['gdp_deprivation'] = 1        # 0: do not process/load lifetime GDP/GRDI average
                                    # 1: load lifetime GDP average analysis        
flags['vulnerability'] = 1          # 0: do not process subsets of d_collect_emergence vs gdp & deprivation quantiles
                                    # 1: process/load d_collect_emergence vs gdp & deprivation quantiles for vulnerability analysis
flags['plot_ms'] = 0 # 1 yes plot, 0 no plot
flags['plot_si'] = 0
flags['reporting'] = 0  
flags['testing'] = 0   



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

df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj = load_GMT(
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

# from exposure import *

# # --------------------------------------------------------------------
# # process lifetime exposure across cohorts

# if flags['lifetime_exposure_cohort']:
    
#     start_time = time.time()
    
#     # calculate exposure per country and per cohort
#     calc_cohort_lifetime_exposure(
#         d_isimip_meta,
#         df_countries,
#         countries_regions,
#         countries_mask,
#         da_population,
#         da_cohort_size,
#         flags,
#     )
    
#     print("--- {} minutes to compute cohort exposure ---".format(
#         np.floor((time.time() - start_time) / 60),
#         )
#           )
    
# else:  # load processed cohort exposure data
    
#     print('Processed exposures will be loaded in emergence calculation')

# --------------------------------------------------------------------
# process picontrol lifetime exposure

# if flags['lifetime_exposure_pic']:
    
#     start_time = time.time()
    
#     # takes 38 mins crop failure
#     d_exposure_perrun_pic = calc_lifetime_exposure_pic(
#         d_pic_meta, 
#         df_countries, 
#         countries_regions, 
#         countries_mask, 
#         da_population, 
#         df_life_expectancy_5, 
#         flags,
#     )
    
#     print("--- {} minutes for PIC exposure ---".format(
#         np.floor((time.time() - start_time) / 60),
#         )
#           )    
    
# else: # load processed pic data
    
#     print('Loading processed pic exposures')

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

# from emergence import *

# # --------------------------------------------------------------------
# # process emergence of cumulative exposures, mask cohort exposures for time steps of emergence

# if flags['emergence']:
    
#     if flags['birthyear_emergence']:
        
#         by_emergence = np.arange(1960,2101)
        
#     else:
        
#         by_emergence = birth_years        
    
#     if not os.path.isfile('./data/{}/cohort_sizes.pkl'.format(flags['version'])):
        
#         # need new cohort dataset that has total population per birth year (using life expectancy info; each country has a different end point)
#         da_cohort_aligned = calc_birthyear_align(
#             da_cohort_size,
#             df_life_expectancy_5,
#             by_emergence,
#         )
        
#         # convert to dataset and add weights
#         ds_cohorts = ds_cohort_align(
#             da_cohort_size,
#             da_cohort_aligned,
#         )
        
#         # pickle birth year aligned cohort sizes and global mean life expectancy
#         with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
#             pk.dump(ds_cohorts,f)  

#     else:
        
#         # load pickled birth year aligned cohort sizes and global mean life expectancy
#         with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
#             ds_cohorts = pk.load(f)                             
    
#     ds_ae_strj, ds_pf_strj = strj_emergence(
#         d_isimip_meta,
#         df_life_expectancy_5,
#         ds_exposure_pic,
#         ds_cohorts,
#         by_emergence,
#         flags,
#     )
        
# else: # load pickles
    
#     pass
    
#     # # birth year aligned population
#     # with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
#     #     ds_cohorts = pk.load(f)
    
#     # # pop frac
#     # with open('./data/{}/{}/pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
#     #     ds_pf_strj = pk.load(f)                
    
                 
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
    
    # load pickle birth year aligned cohort sizes for gridscale analysis (summed per country, i.e. not lat/lon explicit)
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
        da_gs_popdenom = pk.load(f)               

# run gridscale emergence analysis
if flags['gridscale']:
    
    ds_pf_gs = gridscale_emergence(
        d_isimip_meta,
        d_pic_meta,
        flags,
        gridscale_countries,
        da_cohort_size,
        countries_regions,
        countries_mask,
        df_life_expectancy_5,
        da_population,
    )    
    
else:
    
    # # load pickled aggregated lifetime exposure, age emergence and pop frac datasets
    # with open('./data/{}/{}/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
    #     ds_le_gs = pk.load(f)
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
        ds_pf_gs = pk.load(f)
        
if flags['gridscale_le_test']:
    
    ds_pf_gs_le_test = gridscale_emergence_life_expectancy_constant(
        d_isimip_meta,
        d_pic_meta,
        flags,
        gridscale_countries,
        da_cohort_size,
        countries_regions,
        countries_mask,
        df_life_expectancy_5,
        da_population,
    )        
    
else:
    
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_le_test_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',flags['extr']), 'rb') as f:
        ds_pf_gs_le_test = pk.load(f)    
            
# estimate union of all hazard emergences (probably removing this because I don't focus on it in the paper anymore)
# if flags['gridscale_union']:
    
#     da_emergence_mean, da_emergence_union = get_gridscale_union(
#         da_population,
#         flags,
#         gridscale_countries,
#         countries_mask,
#         countries_regions,
#     )

# read in all global emergence masks (d_global_emergence is then used for vulnerability assessment, but only possible on hpc because it is large for some hazards)
if flags['global_emergence_recollect']:

    # temporarily commented out extremes in this function outside heatwaved area to test new means extraction below
    d_global_emergence = collect_global_emergence(
        grid_area,
        flags,
        countries_mask,
        countries_regions,
        gridscale_countries,
        df_GMT_strj,
    )
    
    # temporarily commented out extremes in this function outside heatwaved area to test new means extraction below
    d_global_pic_qntls = collect_pic_qntls(
        grid_area,
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )  
    
    d_global_pic_qntls_extra = collect_pic_qntls_extra(
        grid_area,
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )    
    

if flags['global_avg_emergence']:
      
    # run averaging on d_global_emergence to produce SI figure of emergence fractions
    ds_emergence_mean = get_mean_emergence(
        df_GMT_strj,
        flags,
        da_population,
        d_global_emergence,
    )    
    
# load/proc GDP and deprivation data
if flags['gdp_deprivation']:
    
    ds_gdp, ds_grdi = load_gdp_deprivation(
        flags,
        grid_area,
        da_population,
        countries_mask,
        countries_regions,
        gridscale_countries,
        df_life_expectancy_5,
    )
    
# vulnerability subsetting
if flags['vulnerability']:  

    # get spatially explicit cohort sizes for all birth years in analysis
    da_cohort_size_1960_2020 = get_spatially_explicit_cohorts_1960_2020(
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
        da_cohort_size,
        da_population,
    )
    
    # adds data arrays to ds_gdp and ds_grdi with ranked vulnerability binned by population (i.e. ranges of ranked vulnerability, physically distributed, grouped/binned by population size)            
    ds_gdp_qntls, ds_grdi_qntls = get_vulnerability_quantiles(
        flags,
        grid_area,
        da_cohort_size_1960_2020,
        ds_gdp,
        ds_grdi,
    )
        
    # just a dummy d_global_emergence to run emergence_by_vulnerability
    try:
        d_global_emergence
    except NameError:
        print('to save memory on my laptop, d_global_emergence is not unpickled. defining a dummy var for emergence_by_vulnerability')
        d_global_emergence={}
    else:
        pass

    # dataset of emergence numbers selected by quantiles of vulnerability, both with grdi and gdp
    ds_vulnerability = emergence_by_vulnerability(
        flags,
        df_GMT_strj,
        ds_gdp_qntls,
        ds_grdi_qntls,
        da_cohort_size_1960_2020,
        d_global_emergence,
    )
    
if flags['testing']:
    
    # here we quickly plot quantiles by qth value
    for q in range(10):
        if q == 0:
            da = xr.where(
                ds_gdp_qntls['gdp_q_by_p'].loc[{'qntl':q,'birth_year':2020}].notnull(),
                q+1,
                0
            ).squeeze()
        else:
            da = da + xr.where(
                ds_gdp_qntls['gdp_q_by_p'].loc[{'qntl':q,'birth_year':2020}].notnull(),
                q+1,
                0
            ).squeeze()
    da = da.where(countries_mask.notnull())
    da = da.where(da!=0)
    da.plot(
        levels=np.arange(0.5,10.6),
        cmap='viridis',
        cbar_kwargs={
            'ticks':np.arange(1,11).astype(int)
        }
    )
    # da.plot(levels=range(10))
        # da_cohort_size_1960_2020_q = da_cohort_size_1960_2020.loc[{'birth_year':2020}].where(
        #     ds_gdp_qntls['gdp_q_by_p'].loc[{'qntl':q,'birth_year':2020}].notnull()
        # )        
        
    for q in range(10):
        print('population of {}th quantile:'.format(q))
        # ds_gdp_qntls['gdp_q_by_p'].loc[{'qntl':q,'birth_year':2020}].plot()
        # plt.show()
        da_cohort_size_1960_2020_q = da_cohort_size_1960_2020.loc[{'birth_year':2020}].where(
            ds_gdp_qntls['gdp_q_by_p'].loc[{'qntl':q,'birth_year':2020}].notnull()
        )
        # print('population')
        da_cohort_size_1960_2020_q.plot()
        print(da_cohort_size_1960_2020_q.sum(dim=('lat','lon')))
        plt.show()
        print("")
        print("scatter plot of {}th quantile with circles scaled by pop size:".format(q))
        
        da_cohort_size_1960_2020_q.plot.hist(
            bins=20,
        )
        plt.show()
        
        scatter_test_data = da_cohort_size_1960_2020_q.values.flatten()
        lon = da_cohort_size_1960_2020_q.lon.values
        lat = da_cohort_size_1960_2020_q.lat.values
        lat_grid, lon_grid = np.meshgrid(lat,lon,indexing='ij')
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        plt.scatter(
            x=lon_flat,
            y=lat_flat,
            # s=np.log10(scatter_test_data),
            s=np.sqrt(scatter_test_data),
            c=scatter_test_data,
        )
        plt.show()
    
    # below is copy of pyramid map code without saving
    vln_type = 'gdp'
    fontcolor='gray'
    if vln_type == 'grdi':
        qp_i = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=8,birth_year=2020) #"qp" for "quantile poor", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qp_i = xr.where(qp_i.notnull(),1,0)
        qp_ii = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=9,birth_year=2020)
        qp_ii = xr.where(qp_ii.notnull(),1,0)
        qp = qp_i + qp_ii
        qp = qp.where(qp!=0) # poor == 1

        qr_i = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=0,birth_year=2020) #"qr" for "quantile rich", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qr_i = xr.where(qr_i.notnull(),1,0)
        qr_ii = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=1,birth_year=2020)
        qr_ii = xr.where(qr_ii.notnull(),1,0).squeeze()
        qr = qr_i + qr_ii    
        qr = qr.where(qr!=0)*2 # rich == 2
    elif vln_type == 'gdp':
        qp_i = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=0,birth_year=2020) #"qp" for "quantile poor", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qp_i = xr.where(qp_i.notnull(),1,0)
        qp_ii = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=1,birth_year=2020)
        qp_ii = xr.where(qp_ii.notnull(),1,0)
        qp = qp_i + qp_ii
        qp = qp.where(qp!=0) # poor == 1

        qr_i = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=8,birth_year=2020) #"qr" for "quantile rich", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qr_i = xr.where(qr_i.notnull(),1,0)
        qr_ii = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=9,birth_year=2020)
        qr_ii = xr.where(qr_ii.notnull(),1,0).squeeze()
        qr = qr_i + qr_ii    
        qr = qr.where(qr!=0)*2 # rich == 2    

    # should convert pixels to points via geodataframe
    # first do for "poor"
    df_p = qp.to_dataframe().reset_index()
    # gdf_p = gpd.GeoDataFrame(
    #     df_p.grdi_q_by_p, geometry=gpd.points_from_xy(df_p.lon,df_p.lat)
    # )
    gdf_p = gpd.GeoDataFrame(
        df_p['{}_q_by_p'.format(vln_type)], geometry=gpd.points_from_xy(df_p.lon,df_p.lat)
    )
    gdf_p.set_crs(epsg = "4326",inplace=True)
    # then do for "rich"
    df_r = qr.to_dataframe().reset_index()
    # gdf_r = gpd.GeoDataFrame(
    #     df_r.grdi_q_by_p, geometry=gpd.points_from_xy(df_r.lon,df_r.lat)
    # )
    gdf_r = gpd.GeoDataFrame(
        df_r['{}_q_by_p'.format(vln_type)], geometry=gpd.points_from_xy(df_r.lon,df_r.lat)
    )
    gdf_r.set_crs(epsg = "4326",inplace=True)        
    # get bounds
    robinson = ccrs.Robinson().proj4_init
    gdf_robinson_bounds_v1 = gdf_p.to_crs(robinson).total_bounds # (minx,miny,maxx,maxy) will use this for xlim
    # gdf_robinson_bounds  # wil be read into function (take out of f2 function); use for y lim for antarctica consistency with other plots
    # get rid of nans so the dataframe is more plottable
    gdf_p = gdf_p.dropna()
    gdf_r = gdf_r.dropna()
    # plot
    f,ax = plt.subplots(
        ncols=1,
        nrows=1,
        subplot_kw={'projection':ccrs.Robinson()},
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    gdf_p.to_crs(robinson).plot(
        ax=ax,
        column='{}_q_by_p'.format(vln_type),
        color='darkgoldenrod',
        zorder=5,
        markersize=0.1,
    )    
    gdf_r.to_crs(robinson).plot(
        ax=ax,
        column='{}_q_by_p'.format(vln_type),
        color='forestgreen',
        zorder=4,
        markersize=0.1,
    )            
    ax.set_xlim(gdf_robinson_bounds_v1[0],gdf_robinson_bounds_v1[2])
    ax.set_ylim(gdf_robinson_bounds[1],gdf_robinson_bounds[3])      

    # gdf_robinson_bounds  

    # legend stuff
    cmap = ['darkgoldenrod','forestgreen']  

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75
    legend_font = 10
    legend_lw=3.5   

    legendcols = cmap
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),
        Rectangle((0,0),1,1,color=legendcols[1]),
    ]

    if vln_type == 'grdi':
        labels= [
            '20% highest deprivation',
            '20% lowest deprivation'
        ]
    elif vln_type == 'gdp':
        labels= [
            '20% lowest GDP',
            '20% highest GDP'
        ]
            
    x0 = 0.
    y0 = 1.0
    xlen = 0.2
    ylen = 0.3

    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=1,
        fontsize=legend_font, 
        labelcolor=fontcolor,
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )        

    # f.savefig(
    #     './figures/pyramid/inverted/vln_map_{}.png'.format(vln_type),
    #     dpi=1000,
    #     bbox_inches='tight',
    # )
    plt.show()      
    
    # for v in ds_pic_qntls.data_vars:
    #     print(v)
    #     ds_pic_qntls[v].plot()
    #     plt.show()
    #     print(ds_pic_qntls[v].mean(dim=('lat','lon')))    
    
    # d_global_pic_qntls
    # d_global_pic_qntls_extra
    # ds_pic_qntls = xr.merge([d_global_pic_qntls[flags['extr']],d_global_pic_qntls_extra[flags['extr']]]).drop_vars(['90.0','95.0','97.5'])
    # ds_pic_qntls['all_qntls'] = xr.concat(
    #     [ds_pic_qntls['99.0'],ds_pic_qntls['99.9'],ds_pic_qntls['99.99'],ds_pic_qntls['99.999'],ds_pic_qntls['99.9999'],ds_pic_qntls['99.99999']],
    #     dim='qntls'
    # ).assign_coords({'qntls':['99.0','99.9','99.99','99.999','99.9999','99.99999']})
    # ds_pic_qntls = ds_pic_qntls.rename({'all_qntls':'Bootstrapped pre-industrial \n lifetime exposure','qntls':'Percentiles'})
    # da_all_qntls = ds_pic_qntls['Bootstrapped pre-industrial \n lifetime exposure']
    # df_all_qntls = da_all_qntls.to_dataframe().reset_index()

    
    # box plots of global means
    # import seaborn as sns
    # sns.boxplot(
    #     data=df_all_qntls,
    #     x='Percentiles',
    #     y='Bootstrapped pre-industrial \n lifetime exposure',
    #     # showcaps=False,
    #     # showfliers=False,
    #     color='steelblue',
    # )
    
    # maps
    # da_all_qntls.plot(
    #     x='lon',
    #     y='lat',
    #     col='Percentiles',
    #     col_wrap=3,
    # )
    
        #     p = sns.boxplot(
        #     data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
        #     x='birth_year',
        #     y='pf',
        #     hue='GMT_label',
        #     palette=colors,
        #     showcaps=False,
        #     showfliers=False,
        #     boxprops={
        #         'linewidth':0,
        #         'alpha':0.5
        #     },        
        #     ax=ax,
        # )
    
    # # testing the quantile calculation in Belgium; indeed, we hit the last observation when we choose these higher precision levels
    # with open('./data/pickles_v3/heatwavedarea/gridscale_le_pic_heatwavedarea_Belgium.pkl', 'rb') as f:
    #     ds_pic_le_belgium = pk.load(f)    
        # pic extreme lifetime exposure definition (added more quantiles for v2)
    # test1 = ds_pic_le_belgium['lifetime_exposure'].quantile(
    #         q=0.99999,
    #         dim='lifetimes',
    #         method='closest_observation',
    #     )
    # test2 = ds_pic_le_belgium['lifetime_exposure'].quantile(
    #         q=0.999999,
    #         dim='lifetimes',
    #         method='closest_observation',
    #     )            
    # test3 = ds_pic_le_belgium['lifetime_exposure'].quantile(
    #         q=0.9999999,
    #         dim='lifetimes',
    #         method='closest_observation',
    #     )  
    
    pass    
    

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
    # plot_combined_piechart(
    #     df_GMT_strj,
    #     ds_pf_gs,
    #     da_gs_popdenom,
    #     gdf_country_borders,
    #     sims_per_step,
    #     flags,
    #     df_countries,
    # )
    
    # f2 alternative with absolute pops below box plots and no pie charts
    # further, returning robinson boundaries for use in pyramid plot maps for consistent map extents (that exclude antarctica)
    gdf_robinson_bounds = plot_combined_population(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
        df_countries,
    )        

    # f3 of heatmaps across all hazards
    plot_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )

    # # f4 of emergence union plot for hazards between 1960 and 2020 in a 2.7 degree world
    # plot_emergence_union(
    #     grid_area,
    #     da_emergence_mean,
    # )

    # # f4 alternative for hexagons and multiple thresholds
    # plot_hexagon_multithreshold(
    #     d_global_emergence,
    # )    

    # f4 pyramid plotting
    # first set up quantiles for plotting
    pyramid_setup(
        flags,
        ds_gdp,
        ds_grdi,
        da_cohort_size_1960_2020,
        ds_vulnerability,
    )
    # then run plots
    for vln_type in ('gdp','grdi'):
        print(vln_type)
        pyramid_plot( # this is plot of rich vs poor (brown & green) for current trajectories 
            flags,
            df_GMT_strj,
            vln_type,
        )
        pyramid_map( # map showing locations of top 20 vs bottom 20 quantiles
            vln_type,
            ds_grdi_qntls,
            ds_gdp_qntls,
            gdf_robinson_bounds,
        )
        pyramid_poor_lowhigh( # poor pop but high vs low GMT trajectories
            flags,
            df_GMT_strj,
        )
        pyramid_rich_lowhigh( # rich pop but high vs low GMT trajectories
            flags,
            df_GMT_strj,
        )
    
#%% ----------------------------------------------------------------
# supplementary text plots
# ------------------------------------------------------------------  

if flags['plot_si']:

    from plot_si import *
    
    # heatmaps but with simulations limited to common sims (to avoid dry GCM jumps)
    plot_sf1_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )    
    
    # pf box plots for 1.5, 2.5 and 3.5 degree world across birth years
    plot_sf2_boxplots_allhazards(
        da_gs_popdenom,
        df_GMT_strj,
        flags,
    )      
    
    # pf time series for 2.7 degree world across birth years
    plot_sf3_pf_by_tseries_allhazards(
        flags,
        df_GMT_strj,
        da_gs_popdenom,
    )          
    
    # pf maps for 1..5, 2.5, 3.5 for all hazards
    plot_sf4_pf_maps_allhazards(
        da_gs_popdenom,
        gdf_country_borders,
        flags,
    )        
    
    # emergence fraction plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_sf5_emergence_fracs(
        grid_area,
        ds_emergence_mean,
    )        
    
    # plot locations where exposure occurs at all in our dataset
    plot_sf6_exposure_locations(
        grid_area,
        countries_mask,
        flags,
    )        
    
    # plot heatmaps of pf for country level emergence
    plot_sf7_heatmaps_allhazards_countryemergence(
        df_GMT_strj,
        flags,
    )     
    
    # plot gmt time series for projections (rcp) and for which we map projections onto (ar6)
    plot_sf8_gmt_pathways(
        df_GMT_strj,
        d_isimip_meta,
    )    
        

    # pf time series for 2020 birth year across GMTs
    plot_pf_gmt_tseries_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )
    
    # plot tseries box plots for 1.5, 2.5 and 3.5 when denominator contrained by exposure extent
    plot_geoconstrained_boxplots(
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
    
    # plot heatmaps of delta CF between main text f3 (heatwavedarea panel) and 
    plot_life_expectancy_testing(
        df_GMT_strj,
        GMT_indices_plot,
        da_gs_popdenom,
        flags,
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
    
    # children (i.e. those born between 2003-2020) living unprec exposure between 1.5 and 2.7 degrees warming (for numbers in conclusion of paper)
    print_millions_excess(
        flags,
        df_GMT_strj,
    )     

    # print pf info    
    print_pf_ratios_and_abstract_numbers(
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
    
    # data for box plots of heatwaves (f2)
    print_f2_info(
        ds_pf_gs,
        flags,
        df_GMT_strj,
        da_gs_popdenom,
        gdf_country_borders,
    )
    
    # data for f3
    print_f3_info(
        flags,
    )
    
    # data for pyramid stuff (f4)
    print_pyramid_info(
        flags,
    )
    
                
                