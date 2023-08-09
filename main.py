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
flags['plot_ms'] = 1 # 1 yes plot, 0 no plot
flags['plot_si'] = 1
flags['testing'] = 0      

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
# grid scale emergence
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

    # f3 of heatmaps across all hazards
    plot_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
    )

    # f4 of emergence union plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_emergence_union(
        grid_area,
        da_emergence_mean,
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
    )     
    
    # plot pie charts of all hazards
    plot_allhazards_piecharts(
        da_gs_popdenom,
        df_countries,
    )
    
    # plot cohort sizes in stacked bar chart
    plot_cohort_sizes(
        df_countries,
        da_gs_popdenom,
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
    print_latex_table(
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
    
#%% ----------------------------------------------------------------
# testing for combined plot using cohort sizes instead of pie chart
# ------------------------------------------------------------------
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import cartopy as cr
import cartopy.feature as feature
x=12
y=7
markersize=10
# cbar stuff
col_cbticlbl = 'gray'   # colorbar color of tick labels
col_cbtic = 'gray'   # colorbar color of ticks
col_cbedg = 'gray'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors   
by=2020

gmt_legend={
    GMT_indices_plot[0]:'1.5',
    GMT_indices_plot[1]:'2.5',
    GMT_indices_plot[2]:'3.5',
}     

f = plt.figure(figsize=(x,y))    
gs0 = gridspec.GridSpec(10,3)
# gs0.update(hspace=0.8,wspace=0.8)

# box plots
ax0 = f.add_subplot(gs0[0:5,0:2]) 

# pop totals
# gs10 = gridspec.GridSpecFromSubplotSpec(
#     1,
#     1, 
#     subplot_spec=gs0[5:,0:2],
#     # top=0.8
# )
# ax01 = f.add_subplot(gs10[0],sharex=ax0)
gs0.update(hspace=0)
ax01 = f.add_subplot(gs0[5:,0:2],sharex=ax0)


# maps
gs01 = gridspec.GridSpecFromSubplotSpec(
    3,
    1, 
    subplot_spec=gs0[0:9,2:],
    # top=0.8
)
ax01 = f.add_subplot(gs01[0],projection=ccrs.Robinson())
ax11 = f.add_subplot(gs01[1],projection=ccrs.Robinson())
ax21 = f.add_subplot(gs01[2],projection=ccrs.Robinson()) 
pos00 = ax21.get_position()
cax00 = f.add_axes([
    pos00.x0-0.05,
    pos00.y0-0.1,
    pos00.width*1.95,
    pos00.height*0.2
])

l = 0 # letter indexing

# colorbar stuff ------------------------------------------------------------
cmap_whole = plt.cm.get_cmap('Reds')
levels = np.arange(0,1.01,0.05)
colors = [cmap_whole(i) for i in levels[:-1]]
cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
ticks = np.arange(0,101,10)
norm = mpl.colors.BoundaryNorm(levels,cmap_list_frac.N)   

# pop frac box plot ----------------------------------------------------------
GMT_indices_ticks=[6,12,18,24]
gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    

# levels = np.arange(0,1.01,0.05)
levels = np.arange(0,101,5)
norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)

# get data
df_list_gs = []
extr='heatwavedarea'
with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
    d_isimip_meta = pk.load(file)              
with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
    ds_pf_gs_plot = pk.load(file)
da_p_gs_plot = ds_pf_gs_plot['unprec'].loc[{
    'GMT':GMT_indices_plot,
    'birth_year':sample_birth_years,
}]
sims_per_step = {}
for step in GMT_labels:
    sims_per_step[step] = []
    for i in list(d_isimip_meta.keys()):
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            sims_per_step[step].append(i)    

for step in GMT_indices_plot:
    da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
    df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
    df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
    df_pf_gs_plot_step['hazard'] = extr
    df_list_gs.append(df_pf_gs_plot_step)
df_pf_gs_plot = pd.concat(df_list_gs)

# pf boxplot
colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
p = sns.boxplot(
    data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
    x='birth_year',
    y='pf',
    hue='GMT_label',
    palette=colors,
    showcaps=False,
    showfliers=False,
    boxprops={
        'linewidth':0,
        'alpha':0.5
    },        
    ax=ax0,
)
p.legend_.remove()                  
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)      
ax0.tick_params(colors='gray')
ax0.set_ylim(0,100)
ax0.spines['left'].set_color('gray')
ax0.spines['bottom'].set_color('gray')      
# ax0.set_ylabel('$\mathregular{PF_{Heatwaves}}$',color='gray',fontsize=14)
ax0.set_ylabel('$\mathregular{CF_{Heatwaves}}$ [%]',color='gray',fontsize=14)
ax0.set_xlabel('Birth year',color='gray',fontsize=14)       
ax0.set_title(
    letters[l],
    loc='left',
    fontweight='bold',
    fontsize=10
)    
l+=1 

# bbox
x0 = 0.075
y0 = 0.7
xlen = 0.2
ylen = 0.3

# space between entries
legend_entrypad = 0.5

# length per entry
legend_entrylen = 0.75

legend_font = 10
legend_lw=3.5   

legendcols = list(colors.values())
handles = [
    Rectangle((0,0),1,1,color=legendcols[0]),\
    Rectangle((0,0),1,1,color=legendcols[1]),\
    Rectangle((0,0),1,1,color=legendcols[2])
]

labels= [
    '1.5 °C GMT warming by 2100',
    '2.5 °C GMT warming by 2100',
    '3.5 °C GMT warming by 2100',    
]

ax0.legend(
    handles, 
    labels, 
    bbox_to_anchor=(x0, y0, xlen, ylen), 
    loc = 'upper left',
    ncol=1,
    fontsize=legend_font, 
    labelcolor='gray',
    mode="expand", 
    borderaxespad=0.,\
    frameon=False, 
    columnspacing=0.05, 
    handlelength=legend_entrylen, 
    handletextpad=legend_entrypad
)            
# maps of pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     

# gmt_indices_123 = [19,10,0]
gmt_indices_152535 = [24,15,6]
map_letters = {24:'g',15:'f',6:'e'}
da_p_gs_plot = ds_pf_gs['unprec'].loc[{
    'GMT':gmt_indices_152535,
    'birth_year':by,
}]

# since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
# so we only take sims or runs valid per GMT level and make sure nans are 0
df_list_gs = []
for step in gmt_indices_152535:
    da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
    da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
    df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
    df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
    df_list_gs.append(df_p_gs_plot_step)
df_p_gs_plot = pd.concat(df_list_gs)
df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
gdf = cp(gdf_country_borders.reset_index())
gdf_p = cp(gdf_country_borders.reset_index())
robinson = ccrs.Robinson().proj4_init

for ax,step in zip((ax01,ax11,ax21),gmt_indices_152535):
    gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    gdf_p.to_crs(robinson).plot(
        ax=ax,
        column='pf',
        cmap=cmap_list_frac,
        norm=norm,
        cax=cax00,
        zorder=2,
        rasterized=True,
    )

    gdf.to_crs(robinson).plot(
        ax=ax,
        color='none', 
        edgecolor='black',
        linewidth=0.25,
        zorder=3,
    )
    
    ax.set_title(
        '{} °C'.format(gmt_legend[step]),
        loc='center',
        fontweight='bold',
        fontsize=12,
        color='gray',       
    )
    
    ax.set_title(
        map_letters[step],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    # l+=1          
    
    # pointers connecting 2020, GMT step pixel in heatmap to map panels ------------------
    if step == gmt_indices_152535[0]:
        x_h=1 
    elif step == gmt_indices_152535[1]:
        x_h=0.95                      
    elif step == gmt_indices_152535[-1]:
        x_h=0.9
    y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
    x_m=0
    y_m=0.5
    con = ConnectionPatch(
        xyA=(x_h,y_h),
        xyB=(x_m,y_m),
        coordsA=ax0.transAxes,
        coordsB=ax.transAxes,
        color='gray'
    )
    ax0.add_artist(con)          
    
cb = mpl.colorbar.ColorbarBase(
    ax=cax00, 
    cmap=cmap_list_frac,
    norm=norm,
    orientation='horizontal',
    spacing='uniform',
    ticks=ticks,
    drawedges=False,
)

cb.set_label(
    # 'Population % living unprecedented exposure to heatwaves',
    # '$PF_HW$',
    # '$\mathregular{PF_{Heatwaves}}$ for 2020 birth cohort',
    '$\mathregular{CF_{Heatwaves}}$ for 2020 birth cohort [%]',
    fontsize=14,
    color='gray'
)
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(
    labelcolor=col_cbticlbl,
    color=col_cbtic,
    length=cb_ticlen,
    width=cb_ticwid,
    direction='out'
)   
cb.outline.set_edgecolor(col_cbedg)
cb.outline.set_linewidth(cb_edgthic)   
cax00.xaxis.set_label_position('top')   


gmt_indices_sample = [24,15,6]
gmt_legend={
    gmt_indices_sample[0]:'1.5',
    gmt_indices_sample[1]:'2.5',
    gmt_indices_sample[2]:'3.5',
}
colors = dict(zip([6,15,24],['steelblue','darkgoldenrod','darkred']))

by_sample = [1960,1990,2020]
incomegroups = df_countries['incomegroup'].unique()
income_countries = {}
for category in incomegroups:
    income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])
ig_dict = {
    'Low income':'LI',
    'Lower middle income': 'LMI',
    'Upper middle income': 'UMI',
    'High income': 'HI',
}


# %%
