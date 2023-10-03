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
flags['global_emergence'] = 1       # 0: do                                                                                
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

# read in global emergence masks
if flags['global_emergence']:
    
    d_global_emergence = collect_global_emergence(
        grid_area,
        flags,
        countries_mask,
        countries_regions,
        gridscale_countries,
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
    
    # f2 alternative with both absolute pops below box plots and pie charts
    plot_combined_population_piechart(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
        df_countries,
    )    
    
    # f2 alternative with absolute pops below box plots and noe pie charts
    plot_combined_population(
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
    
    # plot hexagon landfracs (will change to only show landfracs for SI)
    plot_hexagon_landfrac_union(
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

#%% ----------------------------------------------------------------
# plot ar6 hexagons with CF and multi extreme panels
# use da_gridscale_cohortsize with d_global_emergence
with open('./data/pickles/gridscale_cohort_global.pkl', 'rb') as file:
    da_gridscale_cohortsize = pk.load(file)   
extr='heatwavedarea'
emergence = d_global_emergence[extr]
step=17
tr=12
by=1960
testunprec = da_gridscale_cohortsize['cohort_size'].sel(birth_year=by).where(emergence['emergence_per_run_heatwavedarea'].sel(birth_year=by,run=tr)).sum(dim=('lat','lon'))
testunprec_frac = testunprec/da_gridscale_cohortsize['cohort_size'].sel(birth_year=by).sum(dim=('lat','lon'))

verify=ds_pf_gs['unprec'].sel(run=tr,GMT=step,birth_year=by).sum(dim='country')/da_gs_popdenom.sel(birth_year=by).sum(dim='country')


#%% ----------------------------------------------------------------
# plot ar6 hexagons with landfrac per extreme and multi extreme panels

gdf_ar6_hex = gpd.read_file('./data/shapefiles/zones.gpkg').rename(columns={'label': 'Acronym'})
gdf_ar6_hex = gdf_ar6_hex.set_index('Acronym').drop(['id','Continent','Name'],axis=1)
gdf_ar6_hex = gdf_ar6_hex.drop(labels=['GIC'],axis=0)

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

x=8
y=8.1
markersize=10
col_cbticlbl = 'gray'   # colorbar color of tick labels
col_cbtic = 'gray'   # colorbar color of ticks
col_cbedg = '0'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors    
density=6  # density of hatched lines showing frac of sims with emergence
landfrac_threshold = 10
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
    'cropfailedarea': 'Crop failures',
    'driedarea': 'Droughts',
    'floodedarea': 'Floods',
    'heatwavedarea': 'Heatwaves',
    'tropicalcyclonedarea': 'Tropical cyclones',
}

# colorbar stuff for union
cmap_reds = plt.cm.get_cmap('Reds')
colors_union = [
    'white',
    cmap_reds(0.25),
    cmap_reds(0.50),
    cmap_reds(0.75),
]
cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
cmap_list_union.set_over(cmap_reds(0.99))
levels = np.arange(0.5,3.6,1)
union_levels = np.arange(-0.5,3.6,1)
norm_union=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

f = plt.figure(figsize=(x,y))    
gs0 = gridspec.GridSpec(9,2)
gs0.update(wspace=0.25,hspace=0.2)

# left side 1960
ax0 = f.add_subplot(gs0[0:2,0:1],projection=ccrs.Robinson())
ax1 = f.add_subplot(gs0[3:5,0:1])
ax2 = f.add_subplot(gs0[5:7,0:1])
ax3 = f.add_subplot(gs0[7:9,0:1])

# colorbars
pos0 = ax3.get_position()

# colorbar for union
cax_union = f.add_axes([
    pos0.x0+0.15,
    pos0.y0-0.1,
    pos0.width*1.5,
    pos0.height*0.15
])

# right side for 2020
ax4 = f.add_subplot(gs0[0:2,1:2])
ax5 = f.add_subplot(gs0[3:5,1:2])
ax6 = f.add_subplot(gs0[5:7,1:2])
ax7 = f.add_subplot(gs0[7:9,1:2])   

# plot 1960
i=0
l=0

ax0.annotate(
    '1960 birth cohort',
    (0.15,1.3),
    xycoords=ax1.transAxes,
    fontsize=14,
    rotation='horizontal',
    color='gray',
    fontweight='bold',
)          

i+=1

# plot ar6 key
text_kws = dict(
    bbox=dict(color="none"),
    path_effects=[pe.withStroke(linewidth=2, foreground="w")],
    color="gray", 
    fontsize=4, 
)
line_kws=dict(lw=0.5)
ar6_polys = rm.defined_regions.ar6.land
ar6_polys = ar6_polys[np.arange(1,44).astype(int)]
ar6_polys.plot(ax=ax0,label='abbrev',line_kws=line_kws,text_kws=text_kws)
ax0.set_title(
    letters[l],
    loc='left',
    fontweight='bold',
    color='k',
    fontsize=10,
) 
l+=1

# plot blank hexagons
gdf_ar6_hex_blank = gdf_ar6_hex.copy()
gdf_ar6_hex_blank = gdf_ar6_hex_blank.drop(labels='PAC',axis=0)
gdf_ar6_hex_blank['label'] = list(gdf_ar6_hex_blank.index)
gdf_ar6_hex_blank.plot(
    ax=ax4,
    color='w',
    edgecolor='gray',
)
# ar6_polys.abbrevs NEED TO GET THESE IN HEXAGONS
gdf_ar6_hex_blank.apply(
    lambda x: ax4.annotate(
        text=x['label'], 
        xy=x.geometry.centroid.coords[0], 
        ha='center',
        color="gray", 
        fontsize=4, 
    ), 
    axis=1
)
ax4.set_axis_off()
ax4.set_title(
    letters[l],
    loc='left',
    fontweight='bold',
    color='k',
    fontsize=10,
) 
l+=1

for ax,thresh in zip((ax1,ax2,ax3),(10,20,30)):
    
    # new gdf for union
    gdf_ar6_hex_union_1960 = gdf_ar6_hex.copy()
    gdf_ar6_hex_union_1960['union'] = 0    
    
    for extr in extremes:
        
        ds_global_emergence = d_global_emergence[extr]
        gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
        gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
        gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==1960]
        gdf_hex_landrac_plottable_1960['landfrac'] = gdf_hex_landrac_plottable_1960['landfrac'] * 100 
        gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable_1960[['geometry','birth_year','region','names','landfrac']]
        gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'] + (gdf_hex_landrac_plottable_1960['landfrac'] > thresh)
        gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'].astype(float)
    
    gdf_ar6_hex_union_1960.plot(
        column='union',
        cmap=cmap_list_union,
        norm=norm_union,
        edgecolor='gray',
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    
    # label for threshold
    ax.annotate(
        'X = {}'.format(thresh),
        (-0.1,0.4),
        xycoords=ax.transAxes,
        fontsize=10,
        rotation='vertical',
        color='gray',
    )           
    
    l+=1                 
    i+=1

# 2020 birth cohort
ax4.annotate(
    '2020 birth cohort',
    (0.15,1.3),
    xycoords=ax5.transAxes,
    fontsize=14,
    rotation='horizontal',
    color='gray',
    fontweight='bold',
)       

# for ax,thresh in zip((ax4,ax5,ax6,ax7),(10,20,30,40)):
# for ax,thresh in zip((ax4,ax5,ax6),(10,20,30)):
for ax,thresh in zip((ax5,ax6,ax7),(10,20,30)):
    
    # new gdf for union
    gdf_ar6_hex_union_2020 = gdf_ar6_hex.copy()
    gdf_ar6_hex_union_2020['union'] = 0
    
    for extr in extremes:
    
        ds_global_emergence = d_global_emergence[extr]
        gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
        gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
        gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==2020]
        gdf_hex_landrac_plottable_2020['landfrac'] = gdf_hex_landrac_plottable_2020['landfrac'] * 100     
        gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable_2020[['geometry','birth_year','region','names','landfrac']]
        gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'] + (gdf_hex_landrac_plottable_2020['landfrac'] > thresh)
        gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'].astype(float)
    
    gdf_ar6_hex_union_2020.plot(
        column='union',
        cmap=cmap_list_union,
        norm=norm_union,
        edgecolor='gray',
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )        
    
    l+=1          
    i+=1  
    

# colorbar for union of emergence across extremes
cb_u = mpl.colorbar.ColorbarBase(
    ax=cax_union, 
    cmap=cmap_list_union,
    norm=norm_union,
    orientation='horizontal',
    extend='max',
    spacing='uniform',
    ticks=np.arange(0,7).astype('int'),
    drawedges=False,
)

cb_u.set_label(
    'Extremes with median emergence in >X% of region',
    fontsize=10,
    labelpad=8,
    color='gray',
)
cb_u.ax.xaxis.set_label_position('top')
cb_u.ax.tick_params(
    labelcolor=col_cbticlbl,
    labelsize=12,
    color=col_cbtic,
    length=cb_ticlen,
    width=cb_ticwid,
    direction='out'
)   
cb_u.outline.set_color('gray')

#%% ----------------------------------------------------------------
# plot ar6 hexagons for multi extremes across thresholds

def plot_hexagon_multithreshold(
    d_global_emergence,
):                   
    gdf_ar6_hex = gpd.read_file('./data/shapefiles/zones.gpkg').rename(columns={'label': 'Acronym'})
    gdf_ar6_hex = gdf_ar6_hex.set_index('Acronym').drop(['id','Continent','Name'],axis=1)
    gdf_ar6_hex = gdf_ar6_hex.drop(labels=['GIC'],axis=0)

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    from matplotlib.patches import ConnectionPatch
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection

    x=8
    y=8.1
    markersize=10
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = '0'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    
    density=6  # density of hatched lines showing frac of sims with emergence
    landfrac_threshold = 10
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
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }

    # colorbar stuff for union
    cmap_reds = plt.cm.get_cmap('Reds')
    colors_union = [
        'white',
        cmap_reds(0.25),
        cmap_reds(0.50),
        cmap_reds(0.75),
    ]
    cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
    cmap_list_union.set_over(cmap_reds(0.99))
    levels = np.arange(0.5,3.6,1)
    union_levels = np.arange(-0.5,3.6,1)
    norm_union=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(9,2)
    gs0.update(wspace=0.25,hspace=0.2)

    # left side 1960
    ax0 = f.add_subplot(gs0[0:2,0:1],projection=ccrs.Robinson())
    ax1 = f.add_subplot(gs0[3:5,0:1])
    ax2 = f.add_subplot(gs0[5:7,0:1])
    ax3 = f.add_subplot(gs0[7:9,0:1])

    # colorbars
    pos0 = ax3.get_position()

    # colorbar for union
    cax_union = f.add_axes([
        pos0.x0+0.15,
        pos0.y0-0.1,
        pos0.width*1.5,
        pos0.height*0.15
    ])

    # right side for 2020
    ax4 = f.add_subplot(gs0[0:2,1:2])
    ax5 = f.add_subplot(gs0[3:5,1:2])
    ax6 = f.add_subplot(gs0[5:7,1:2])
    ax7 = f.add_subplot(gs0[7:9,1:2])   
    
    # plot 1960
    i=0
    l=0

    ax0.annotate(
        '1960 birth cohort',
        (0.15,1.3),
        xycoords=ax1.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )          

    i+=1

    # plot ar6 key
    text_kws = dict(
        bbox=dict(color="none"),
        path_effects=[pe.withStroke(linewidth=2, foreground="w")],
        color="gray", 
        fontsize=4, 
    )
    line_kws=dict(lw=0.5)
    ar6_polys = rm.defined_regions.ar6.land
    ar6_polys = ar6_polys[np.arange(1,44).astype(int)]
    ar6_polys.plot(ax=ax0,label='abbrev',line_kws=line_kws,text_kws=text_kws)
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    ) 
    l+=1

    # plot blank hexagons
    gdf_ar6_hex_blank = gdf_ar6_hex.copy()
    gdf_ar6_hex_blank = gdf_ar6_hex_blank.drop(labels='PAC',axis=0)
    gdf_ar6_hex_blank['label'] = list(gdf_ar6_hex_blank.index)
    gdf_ar6_hex_blank.plot(
        ax=ax4,
        color='w',
        edgecolor='gray',
    )
    # ar6_polys.abbrevs NEED TO GET THESE IN HEXAGONS
    gdf_ar6_hex_blank.apply(
        lambda x: ax4.annotate(
            text=x['label'], 
            xy=x.geometry.centroid.coords[0], 
            ha='center',
            color="gray", 
            fontsize=4, 
        ), 
        axis=1
    )
    ax4.set_axis_off()
    ax4.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    ) 
    l+=1

    for ax,thresh in zip((ax1,ax2,ax3),(10,20,30)):
        
        # new gdf for union
        gdf_ar6_hex_union_1960 = gdf_ar6_hex.copy()
        gdf_ar6_hex_union_1960['union'] = 0    
        
        for extr in extremes:
            
            ds_global_emergence = d_global_emergence[extr]
            gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
            gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
            gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==1960]
            gdf_hex_landrac_plottable_1960['landfrac'] = gdf_hex_landrac_plottable_1960['landfrac'] * 100 
            gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable_1960[['geometry','birth_year','region','names','landfrac']]
            gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'] + (gdf_hex_landrac_plottable_1960['landfrac'] > thresh)
            gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'].astype(float)
        
        gdf_ar6_hex_union_1960.plot(
            column='union',
            cmap=cmap_list_union,
            norm=norm_union,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        
        # label for threshold
        ax.annotate(
            'X = {}'.format(thresh),
            (-0.1,0.4),
            xycoords=ax.transAxes,
            fontsize=10,
            rotation='vertical',
            color='gray',
        )           
        
        l+=1                 
        i+=1

    # 2020 birth cohort
    ax4.annotate(
        '2020 birth cohort',
        (0.15,1.3),
        xycoords=ax5.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )       

    # for ax,thresh in zip((ax4,ax5,ax6,ax7),(10,20,30,40)):
    # for ax,thresh in zip((ax4,ax5,ax6),(10,20,30)):
    for ax,thresh in zip((ax5,ax6,ax7),(10,20,30)):
        
        # new gdf for union
        gdf_ar6_hex_union_2020 = gdf_ar6_hex.copy()
        gdf_ar6_hex_union_2020['union'] = 0
        
        for extr in extremes:
        
            ds_global_emergence = d_global_emergence[extr]
            gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
            gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
            gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==2020]
            gdf_hex_landrac_plottable_2020['landfrac'] = gdf_hex_landrac_plottable_2020['landfrac'] * 100     
            gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable_2020[['geometry','birth_year','region','names','landfrac']]
            gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'] + (gdf_hex_landrac_plottable_2020['landfrac'] > thresh)
            gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'].astype(float)
        
        gdf_ar6_hex_union_2020.plot(
            column='union',
            cmap=cmap_list_union,
            norm=norm_union,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )        
        
        l+=1          
        i+=1  
        

    # colorbar for union of emergence across extremes
    cb_u = mpl.colorbar.ColorbarBase(
        ax=cax_union, 
        cmap=cmap_list_union,
        norm=norm_union,
        orientation='horizontal',
        extend='max',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )

    cb_u.set_label(
        'Extremes with median emergence in >X% of region',
        fontsize=10,
        labelpad=8,
        color='gray',
    )
    cb_u.ax.xaxis.set_label_position('top')
    cb_u.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb_u.outline.set_color('gray')

    f.savefig('./ms_figures/emergence_union_hexagons_multithresh.png',dpi=1000,bbox_inches='tight')

# %%
