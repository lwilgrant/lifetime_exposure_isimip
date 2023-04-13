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
cntry='Kenya'
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
nairobi_lat = -1.29
nairobi_lon = 36.8   

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
            
da_test_nairobi = ds_spatial['cumulative_exposure'].sel({'lat':nairobi_lat,'lon':nairobi_lon},method='nearest').mean(dim='run')
da_test_kenya = ds_spatial['cumulative_exposure'].weighted(lat_weights).mean(('lat','lon')).mean(dim='run')
            
# load PIC pickle
with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
    ds_pic = pk.load(f)   

# plotting nairobi lat/lon pixel doesn't give smooth kde
df_pic_nairobi = ds_pic['lifetime_exposure'].sel({'lat':nairobi_lat,'lon':nairobi_lon},method='nearest').to_dataframe().drop(columns=['lat','lon','quantile'])         
da_pic_nairobi_9999 = ds_pic['99.99'].sel({'lat':nairobi_lat,'lon':nairobi_lon},method='nearest')  
sns.displot(data=df_pic_nairobi,kind='kde')

# plotting mean across kenya should be more smooth
df_pic_kenya = ds_pic['lifetime_exposure'].weighted(lat_weights).mean(('lon','lat')).to_dataframe().drop(columns=['quantile'])   
df_pic_kenya_9999 = ds_pic['99.99'].weighted(lat_weights).mean(('lon','lat'))       
sns.displot(data=df_pic_kenya,kind='kde')

# plot building
from mpl_toolkits.axes_grid1 import inset_locator as inset
colors = dict(zip(GMT_indices_plot,['steelblue','darkgoldenrod','darkred']))
x=14
y=8

# 1960
f,ax = plt.subplots()
for step in GMT_indices_plot:
    da_test_nairobi.loc[{'birth_year':1960,'GMT':step}].plot.line(
        ax=ax,
        color=colors[step],
    )
end_year=1960+np.floor(df_life_expectancy_5.loc[1960,'Kenya'])
ax.set_title(None)
ax.set_xlim(
    1960,
    end_year,
)
ax.set_ylim(
    0,
    np.round(da_test_nairobi.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max()),
)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)       
    
# 1990
ax2_l = 1990
ax2_b = np.round(da_test_nairobi.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max()) 
ax2_w = np.floor(df_life_expectancy_5.loc[1990,'Kenya'])
ax2_h = np.round(da_test_nairobi.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max())
ax2 = ax.inset_axes(
    bounds=(ax2_l, ax2_b, ax2_w, ax2_h),
    transform=ax.transData,
)

for step in GMT_indices_plot:
    da_test_nairobi.loc[{'birth_year':1990,'GMT':step}].plot.line(
        ax=ax2,
        color=colors[step],
    )
end_year=1990+np.floor(df_life_expectancy_5.loc[1990,'Kenya'])
ax2.set_title(None)
ax2.set_xlim(
    1990,
    end_year,
)
ax2.set_ylim(
    0,
    np.round(da_test_nairobi.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()),
)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)  
ax2.
    
# f = plt.figure(figsize=(x,y))    
# gs0 = gridspec.GridSpec(4,4)
# gs0.update(hspace=0.8,wspace=0.8)
# ax00 = f.add_subplot(gs0[0:2,0:2]) # heatmap
# ax10 = f.add_subplot(gs0[2:,0:2]) # scatterplot for 2020 by
# gs00 = gridspec.GridSpecFromSubplotSpec(
#     3,
#     1, 
#     subplot_spec=gs0[:4,2:],
# )
# ax01 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
# ax11 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
# ax21 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 
# pos00 = ax00.get_position()
# cax00 = f.add_axes([
#     pos00.x0,
#     pos00.y0+0.4,
#     pos00.width * 2.25,
#     pos00.height*0.1
# ])

#%% ----------------------------------------------------------------
# age emergence & pop frac testing
# ------------------------------------------------------------------        

if flags['testing']:
    
    pass

                        

# %%
