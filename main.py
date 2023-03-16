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
flags['gridscale_spatially_explicit'] = 1      # 0: do not load pickles for country lat/lon emergence (only for subset of GMTs and birth years)
                                               # 1: load those^ pickles
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
    with open('./data/pickles/exposure_trends_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
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
    with open('./data/pickles/lifetime_exposure_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
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

    with open('./data/pickles/exposure_pic_{}.pkl'.format(flags['extr']), 'rb') as f:
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
    with open('./data/pickles/pop_frac_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
        ds_pf_strj = pk.load(f)                
    
    # age emergence           
    with open('./data/pickles/age_emergence_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
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
        pk.dump(ds_cohorts,f)  
        
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
    with open('./data/pickles/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['extr']), 'rb') as f:
        ds_le_gs = pk.load(f)
    with open('./data/pickles/gridscale_aggregated_age_emergence_{}.pkl'.format(flags['extr']), 'rb') as f:
        ds_ae_gs = pk.load(f)
    with open('./data/pickles/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['extr']), 'rb') as f:
        ds_pf_gs = pk.load(f)

# load spatially explicit datasets
if flags['gridscale_spatially_explicit']:
    
    d_gs_spatial = {}
    for cntry in gridscale_countries:
        with open('./data/pickles/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],cntry), 'rb') as f:
            d_gs_spatial[cntry] = pk.load(f)

#%% ----------------------------------------------------------------
# plot
# ------------------------------------------------------------------   

# not sure if this is at the correct place, but dictionary of valid runs per GMT step
sim_labels = {}
for step in GMT_labels:
    sim_labels[step] = []
    print('step {}'.format(step))
    for i in list(d_isimip_meta.keys()):
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            sim_labels[step].append(i)

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
    
    plot_pf_t_GMT_strj(
        ds_pf_strj,
        df_GMT_strj,
        flags,
    )        
    
    # plotting unprecedented population totals between country level and gridscale level for given region 
    boxplot_cs_vs_gs_p(
        ds_pf_strj,
        ds_pf_gs,
        df_GMT_strj,
        flags,
        sim_labels,
        gridscale_countries,
    )    
    
    # plotting unprecedented population fracs between country and gridscale levels for given region
    boxplot_cs_vs_gs_pf(
        ds_cohorts,
        da_gs_popdenom,
        ds_pf_strj,
        ds_pf_gs,
        df_GMT_strj,
        flags,
        sim_labels,
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
        sim_labels,
        flags,
    )    
    
    # plot of heatwave heatmap, scatter plot and maps of 1, 2 and 3 degree pop frac unprecedented across countries
    combined_plot_hw_pf(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sim_labels,
        flags,
    )        
    

    
    list_countries=gridscale_countries

#%% ----------------------------------------------------------------
# plot
# ------------------------------------------------------------------     
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

def combined_plot_hw(
    df_GMT_strj,
    ds_pf_gs,
    gdf_country_borders,
    flags,
):
    x=12
    y=10
    markersize=10
    tick_font = 12
    # cbar stuff
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(4,4)
    gs0.update(hspace=0.8,wspace=0.8)
    ax00 = f.add_subplot(gs0[0:2,0:2]) # heatmap
    ax10 = f.add_subplot(gs0[2:,0:2]) # scatterplot for 2020 by
    gs00 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[:4,2:],
    )
    ax01 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 
    pos00 = ax00.get_position()
    cax00 = f.add_axes([
        pos00.x0,
        pos00.y0+0.4,
        pos00.width,
        pos00.height*0.1
    ])
    pos01 = ax01.get_position()
    caxn1 = f.add_axes([
        pos01.x0-0.0775,
        pos00.y0+0.4,
        pos00.width,
        pos00.height*0.1
    ])    

    # pop frac heatmap ----------------------------------------------------------
    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)

    if flags['extr'] == 'heatwavedarea':
        levels = np.arange(0,1.01,0.1)
    else:
        levels = 10
        
    p2 = ds_pf_gs['unprec'].loc[{
        'birth_year':np.arange(1960,2021)
    }].sum(dim='country')
    p2 = p2.where(p2!=0).mean(dim='run') / da_cntry_pops.sum(dim='country')
    p2 = p2.plot(
        x='birth_year',
        y='GMT',
        ax=ax00,
        add_colorbar=True,
        levels=levels,
        cbar_kwargs={
            'label':'Population fraction',
            'cax':cax00,
            'orientation':'horizontal'
        }
    )
    p2.axes.set_yticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p2.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    p2.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p2.axes.set_xlabel('Birth year')
    cax00.xaxis.set_label_position('top')

    # add rectangle to 2020 series
    ax00.add_patch(Rectangle(
        (2020-0.5,0-0.5),1,29,
        facecolor='none',
        ec='k',
        lw=0.8
    ))
    # bracket connecting 2020 in heatmap to scatter plot panel

    # pop frac scatter ----------------------------------------------------------

    da_plt = ds_pf_gs['unprec'].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
    da_plt_gmt = da_plt.loc[{'birth_year':by}].where(da_plt.loc[{'birth_year':by}]!=0)
    da_plt_gmt = da_plt_gmt / da_cntry_pops.loc[{'birth_year':by}].sum(dim='country')
    p = da_plt_gmt.to_dataframe(name='pf').reset_index(level="run")
    x = p.index.values
    y = p['pf'].values
    ax10.scatter(
        x,
        y,
        s=markersize,
        c='steelblue'
    )
    ax10.plot(
        GMT_labels,
        da_plt_gmt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax10.set_ylabel(
        'Population fraction', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )          
    ax10.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
    )                                           
    ax10.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100,
    )    
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)    

    handles = [
        Line2D([0],[0],linestyle='None',marker='o',color='steelblue'),
        Line2D([0],[0],marker='_',color='r'),
            
    ]
    labels= [
        'Simulations',
        'Mean',     
    ]    
    x0 = 0.55 # bbox for legend
    y0 = 0.25
    xlen = 0.2
    ylen = 0.2    
    legend_font = 10        
    ax10.legend(
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

    # pop emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------

    cmap_whole = plt.cm.get_cmap('Reds')
    cmap55 = cmap_whole(0.01)
    cmap50 = cmap_whole(0.05)   # blue
    cmap45 = cmap_whole(0.1)
    cmap40 = cmap_whole(0.15)
    cmap35 = cmap_whole(0.2)
    cmap30 = cmap_whole(0.25)
    cmap25 = cmap_whole(0.3)
    cmap20 = cmap_whole(0.325)
    cmap10 = cmap_whole(0.4)
    cmap5 = cmap_whole(0.475)
    cmap0 = 'gray'
    cmap_5 = cmap_whole(0.525)
    cmap_10 = cmap_whole(0.6)
    cmap_20 = cmap_whole(0.625)
    cmap_25 = cmap_whole(0.7)
    cmap_30 = cmap_whole(0.75)
    cmap_35 = cmap_whole(0.8)
    cmap_40 = cmap_whole(0.85)
    cmap_45 = cmap_whole(0.9)
    cmap_50 = cmap_whole(0.95)  # red
    cmap_55 = cmap_whole(0.99)

    colors = [
        cmap0, # gray for 0 unprecedented
        cmap45,cmap35,cmap25,cmap20,cmap5, # 100,000s
        cmap_5,cmap_20,cmap_25,cmap_35,cmap_45,cmap_55, # millions
    ]

    # declare list of colors for discrete colormap of colorbar
    cmap_list_p = mpl.colors.ListedColormap(colors,N=len(colors))

    # colorbar args
    values_p = [
        -0.1, 0.1,
        2*10**5,4*10**5,6*10**5,8*10**5,
        10**6,2*10**6,4*10**6,6*10**6,8*10**6,
        10**7,2*10**7,
    ]
    tick_locs_p = [
        0,1,
        2*10**5,4*10**5,6*10**5,8*10**5,
        10**6,2*10**6,4*10**6,6*10**6,8*10**6,
        10**7,2*10**7,        
    ]
    tick_labels_p = [
        '0',None,
        '200,000','400,000','600,000','800,000',
        '10e6','2x10e6','4x10e6','6x10e6','8x10e6',
        '10e7','2x10e7',
    ]
    norm_p = mpl.colors.BoundaryNorm(values_p,cmap_list_p.N)         

    gmt_indices_123 = [0,10,19]
    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':gmt_indices_123,
        'birth_year':2020,
    }]
    df_list_gs = []
    for step in gmt_indices_123:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sim_labels[step],'GMT':step}].mean(dim='run')
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe().reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['unprec'] = df_p_gs_plot['unprec'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_123):
        gdf_p['unprec']=df_p_gs_plot['unprec'][df_p_gs_plot['GMT']==step].values
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='unprec',
            cmap=cmap_list_p,
            norm=norm_p,
            cax=caxn1,
        )           
        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
        ) 
        
        cb = mpl.colorbar.ColorbarBase(
            ax=caxn1, 
            cmap=cmap_list_p,
            norm=norm_p,
            orientation='horizontal',
            spacing='uniform',
            drawedges=False,
            ticks=tick_locs_p,
        )

    cb.set_label('Unprecedented population')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )
    cb.ax.set_xticklabels(
        tick_labels_p,
        rotation=45,
    )    
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)                         

    f.savefig('./figures/combined_heatmap_scatter_mapsofp_{}.png'.format(flags['extr']),dpi=900)
    plt.show()        


    
#%% ----------------------------------------------------------------
# age emergence & pop frac testing
# ------------------------------------------------------------------        

if flags['testing']:
    
    pass