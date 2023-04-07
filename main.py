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
flags['gridscale_union'] = 1        # 0: do not process/load pickles for mean emergence and union of emergence across hazards
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
# emergence across GMTs -----------------------------------    
x=8
y=9
markersize=10
lat = grid_area.lat.values
lon = grid_area.lon.values
mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
# tick_font = 12
# cbar stuff
col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors    

extremes = [
    'burntarea', 
    'cropfailedarea', 
    'driedarea', 
    'floodedarea', 
    'heatwavedarea', 
    'tropicalcyclonedarea',
]

colors=[
    mpl.colors.to_rgb('steelblue'),
    mpl.colors.to_rgb('darkgoldenrod'),
    mpl.colors.to_rgb('darkred'),
]
cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
levels = np.arange(0.5,3.6,1)

import matplotlib.gridspec as gridspec
f = plt.figure(figsize=(x,y))    
gs0 = gridspec.GridSpec(2,1)
gs0.update(wspace=0)
ax0 = f.add_subplot(gs0[0:1,:],projection=ccrs.Robinson()) # map of emergence union
gs00 = gridspec.GridSpecFromSubplotSpec(
    2,
    3,
    subplot_spec=gs0[1:3,:],
    wspace=0,
    hspace=0,
)
# gs00.update(wspace=0.2)
ax00 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
ax10 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
ax20 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 

ax01 = f.add_subplot(gs00[3],projection=ccrs.Robinson())
ax11 = f.add_subplot(gs00[4],projection=ccrs.Robinson())
ax21 = f.add_subplot(gs00[5],projection=ccrs.Robinson())        

ax00.set_title("ax00")
ax10.set_title("ax10")
ax20.set_title("ax20")
ax01.set_title("ax01")
ax11.set_title("ax11")
ax21.set_title("ax21")


for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):

    # 1 degree warming
    p1 = da_emergence_mean.loc[{
        'hazard':extr,
        'GMT':0,
    }]
    p1 = xr.where(p1>0,1,0)
    p1 = p1.where(p1).where(mask.notnull())    
    p1.plot(
        ax=ax,
        cmap=cmap_list,
        levels=levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    # 2 degree warming
    p2 = da_emergence_mean.loc[{
        'hazard':extr,
        'GMT':10,
    }]
    p2 = xr.where(p2>0,1,0)
    p2 = p2.where(p2).where(mask.notnull())*2
    p2.plot(
        ax=ax,
        cmap=cmap_list,
        levels=levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    
    # 3 degree warming
    p3 = da_emergence_mean.loc[{
        'hazard':extr,
        'GMT':19,
    }]
    p3 = xr.where(p3>0,1,0)
    p3 = p3.where(p3).where(mask.notnull())*3
    p3.plot(
        ax=ax,
        cmap=cmap_list,
        levels=levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=0
    )    
    ax.coastlines(linewidth=0.25)
    ax.set_title(extr)
union_levels = np.arange(0.5,6.5,1)

# p_u3 = da_emergence_union.loc[{'GMT':19,}].where(mask.notnull())
# xr.where(ds_emergence_union['emergence_mean']>0.5,1,0).sum(dim='hazard')
p_u3 = xr.where(da_emergence_mean.loc[{'GMT':19}]>0,1,0).sum(dim='hazard').where(mask.notnull())
p_u3.plot(
    ax=ax0,
    cmap='Reds',
    levels=union_levels,
    add_colorbar=False,
    add_labels=False,
    transform=ccrs.PlateCarree(),
    zorder=0
)
ax0.coastlines(linewidth=0.25)
  
  
#%% ----------------------------------------------------------------  
# emergence across birth years -----------------------------------    
x=8
y=12
markersize=10
lat = grid_area.lat.values
lon = grid_area.lon.values
mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
# tick_font = 12
# cbar stuff
col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors    

extremes = [
    'burntarea', 
    'cropfailedarea', 
    'driedarea', 
    'floodedarea', 
    'heatwavedarea', 
    'tropicalcyclonedarea',
]

extremes_labels = {
    'burntarea': 'Wild fires',
    'cropfailedarea': 'Crop failure',
    'driedarea': 'Droughts',
    'floodedarea': 'Floods',
    'heatwavedarea': 'Heatwaves',
    'tropicalcyclonedarea': 'Tropical cyclones'
}

colors=[
    mpl.colors.to_rgb('steelblue'),
    mpl.colors.to_rgb('darkgoldenrod'),
    mpl.colors.to_rgb('darkred'),
]
cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
levels = np.arange(0.5,3.6,1)

f = plt.figure(figsize=(x,y))    
gs0 = gridspec.GridSpec(3,1)
gs0.update(wspace=0)
ax0 = f.add_subplot(gs0[0:1,:],projection=ccrs.Robinson()) # map of emergence union
gs00 = gridspec.GridSpecFromSubplotSpec(
    3,
    2,
    subplot_spec=gs0[1:4,:],
    wspace=0,
    hspace=0,
)
# gs00.update(wspace=0.2)
ax00 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
ax10 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
ax20 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 

ax01 = f.add_subplot(gs00[3],projection=ccrs.Robinson())
ax11 = f.add_subplot(gs00[4],projection=ccrs.Robinson())
ax21 = f.add_subplot(gs00[5],projection=ccrs.Robinson())    

# union of emergence locations
i=0
union_levels = np.arange(0.5,6.5,1)
p_u3 = da_emergence_union.loc[{'GMT':19,'birth_year':2020}].where(mask.notnull())
import cartopy.feature as feature
ax0.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
ax0.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
p_u3.plot(
    ax=ax0,
    cmap='Reds',
    levels=union_levels,
    add_colorbar=False,
    add_labels=False,
    transform=ccrs.PlateCarree(),
    zorder=5
)
ax0.set_title(
    letters[i],
    loc='left',
    fontweight='bold',
)
for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
    
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
    p3 = da_emergence_mean.loc[{
        'hazard':extr,
        'GMT':19,
        'birth_year':2020,
    }]
    p3 = xr.where(p3>0,1,0)
    p3 = p3.where(p3).where(mask.notnull())*3
    p3.plot(
        ax=ax,
        cmap=cmap_list,
        levels=levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=5
    )    
    ax.set_title(
        extremes_labels[extr],
        loc='center',
        fontweight='bold',
    )
    ax.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
    )
    i+=1

#%% ----------------------------------------------------------------
x=18
y=12
markersize=10
lat = grid_area.lat.values
lon = grid_area.lon.values
mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors    
extremes = [
    'burntarea', 
    'cropfailedarea', 
    'driedarea', 
    'floodedarea', 
    'heatwavedarea', 
    'tropicalcyclonedarea',
]
extremes_labels = {
    'burntarea': 'Wildfires',
    'cropfailedarea': 'Crop failure',
    'driedarea': 'Droughts',
    'floodedarea': 'Floods',
    'heatwavedarea': 'Heatwaves',
    'tropicalcyclonedarea': 'Tropical cyclones'
}
colors=[
    mpl.colors.to_rgb('steelblue'),
    mpl.colors.to_rgb('darkgoldenrod'),
    mpl.colors.to_rgb('peru'),
]
cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

cmap_reds = plt.cm.get_cmap('Reds')
colors_union = [
    'white',
    cmap_reds(0.15),
    cmap_reds(0.3),
    cmap_reds(0.45),
    cmap_reds(0.6),
    cmap_reds(0.75),
    cmap_reds(0.9),
]
cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
levels = np.arange(0.5,3.6,1)
union_levels = np.arange(-0.5,6.6,1)
norm=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

f = plt.figure(figsize=(x,y))    
gs0 = gridspec.GridSpec(3,2)
gs0.update(wspace=0.25)

# left side for 1960
ax0 = f.add_subplot(gs0[0:1,0:1],projection=ccrs.Robinson()) # map of emergence union

pos0 = ax0.get_position()
cax = f.add_axes([
    pos0.x0+0.265,
    pos0.y0,
    pos0.width * 0.2,
    pos0.height*1
])

gsn0 = gridspec.GridSpecFromSubplotSpec(
    3,
    2,
    subplot_spec=gs0[1:4,0:1],
    wspace=0,
    hspace=0,
)
ax00 = f.add_subplot(gsn0[0],projection=ccrs.Robinson())
ax10 = f.add_subplot(gsn0[1],projection=ccrs.Robinson())
ax20 = f.add_subplot(gsn0[2],projection=ccrs.Robinson()) 

ax01 = f.add_subplot(gsn0[3],projection=ccrs.Robinson())
ax11 = f.add_subplot(gsn0[4],projection=ccrs.Robinson())
ax21 = f.add_subplot(gsn0[5],projection=ccrs.Robinson())    

# right side for 2020
ax1 = f.add_subplot(gs0[0:1,1:2],projection=ccrs.Robinson()) # map of emergence union
gsn1 = gridspec.GridSpecFromSubplotSpec(
    3,
    2,
    subplot_spec=gs0[1:4,1:2],
    wspace=0,
    hspace=0,
)
ax02 = f.add_subplot(gsn1[0],projection=ccrs.Robinson())
ax12 = f.add_subplot(gsn1[1],projection=ccrs.Robinson())
ax22 = f.add_subplot(gsn1[2],projection=ccrs.Robinson()) 

ax03 = f.add_subplot(gsn1[3],projection=ccrs.Robinson())
ax13 = f.add_subplot(gsn1[4],projection=ccrs.Robinson())
ax23 = f.add_subplot(gsn1[5],projection=ccrs.Robinson())    

# plot 1960
i=0
p_u3 = da_emergence_union.loc[{'GMT':19,'birth_year':1960}].where(mask.notnull())
import cartopy.feature as feature
ax0.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
ax0.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
p_u3.plot(
    ax=ax0,
    # cmap='Reds',
    cmap=cmap_list_union,
    levels=union_levels,
    add_colorbar=False,
    add_labels=False,
    transform=ccrs.PlateCarree(),
    zorder=5
)
ax0.set_title(
    letters[i],
    loc='left',
    fontweight='bold',
)
ax0.set_title(
    # 'Emergence in 1960 cohort lifetimes',
    'All hazards',
    loc='center',
    fontweight='bold',
)
i+=1
for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
    
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
    p3 = da_emergence_mean.loc[{
        'hazard':extr,
        'GMT':19,
        'birth_year':1960,
    }]
    p3 = xr.where(p3>0,1,0)
    p3 = p3.where(p3).where(mask.notnull())*3
    p3.plot(
        ax=ax,
        cmap=cmap_list,
        levels=levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=5
    )    
    ax.set_title(
        extremes_labels[extr],
        loc='center',
        fontweight='bold',
    )
    ax.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
    )
    i+=1
    
# plot 2020
# union_levels = np.arange(0.5,6.5,1)
p_u3 = da_emergence_union.loc[{'GMT':19,'birth_year':2020}].where(mask.notnull())
import cartopy.feature as feature
ax1.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
ax1.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
p_u3.plot(
    ax=ax1,
    # cmap='Reds',
    cmap=cmap_list_union,
    levels=union_levels,
    add_colorbar=False,
    add_labels=False,
    transform=ccrs.PlateCarree(),
    zorder=5
)
ax1.set_title(
    letters[i],
    loc='left',
    fontweight='bold',
)
ax1.set_title(
    # 'Emergence in 2020 cohort lifetimes',
    'All hazards',
    loc='center',
    fontweight='bold',
)
i+=1
for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
    
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
    p3 = da_emergence_mean.loc[{
        'hazard':extr,
        'GMT':19,
        'birth_year':2020,
    }]
    p3 = xr.where(p3>0,1,0)
    p3 = p3.where(p3).where(mask.notnull())*3
    p3.plot(
        ax=ax,
        cmap=cmap_list,
        levels=levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=5
    )    
    ax.set_title(
        extremes_labels[extr],
        loc='center',
        fontweight='bold',
    )
    ax.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
    )
    i+=1    
    
    cb = mpl.colorbar.ColorbarBase(
        ax=cax, 
        cmap=cmap_list_union,
        norm=norm,
        orientation='vertical',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )

    cb.set_label(
        'Number of emerged hazards'.format(flags['extr']),
        fontsize=12,
        labelpad=10,
    )
    cb.ax.yaxis.set_label_position('right')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    # cb.outline.set_edgecolor(col_cbedg)
    # cb.outline.set_linewidth(cb_edgthic)   
    # cax.xaxis.set_label_position('top')       

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
    
#%% ----------------------------------------------------------------
# age emergence & pop frac testing
# ------------------------------------------------------------------        

if flags['testing']:
    
    pass

                        

# %%
