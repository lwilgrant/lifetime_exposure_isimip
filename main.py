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
flags['gdp_deprivation'] = 0        # do not process lifetime GDP average (load pickles)
                        # load lifetime GDP average analysis                                   
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

# birth year aligned cohort sizes for gridscale analysis
if not os.path.isfile('./data/{}/gs_cohort_sizes.pkl'.format(flags['version'])):

    da_gs_popdenom = get_gridscale_popdenom(
        gridscale_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        da_population,
        df_life_expectancy_5,
    )

    # pickle birth year aligned cohort sizes for gridscale analysis
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
        pk.dump(da_gs_popdenom,f)  
        
else:
    
    # load pickle birth year aligned cohort sizes for gridscale analysis
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
        da_gs_popdenom = pk.load(f)               

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
if flags['gdp']:
    
    ds_gdp, ds_grdi = load_gdp_deprivation(
        grid_area,
        gridscale_countries,
        df_life_expectancy_5,
    )
    
else:
        
    pass   

# filter emergence masks and pop estimates based on percentiles of GDP and GRDI
# dataset for lifetime average GDP
ds_gdp = xr.Dataset(
    data_vars={
        'gdp_mean': (
            ['birth_year','lat','lon'],
            np.full(
                (len(birth_years),len(lat),len(lon)),
                fill_value=np.nan,
            ),
        ),      
        'gdp_sum': (
            ['birth_year','lat','lon'],
            np.full(
                (len(birth_years),len(lat),len(lon)),
                fill_value=np.nan,
            ),
        ),              
    },
    coords={
        'birth_year': ('birth_year', birth_years),
        'lat': ('lat', lat),
        'lon': ('lon', lon)
    }
)

# loop thru countries
# for cntry in list_countries:
for cntry in ['Belgium','Netherlands','France','Germany']:
# for cntry in ['Belgium','France']:

    # load demography pickle for country
    with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'rb') as f:
        ds_dmg = pk.load(f)  
        
    ds_dmg['country_extent'].plot()
    plt.show()

    da_gdp_cntry = da_gdp_pc.where(ds_dmg['country_extent'].notnull(),drop=True)    
    da_gdp_sum = xr.concat(
        [(da_gdp_cntry.loc[{'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1)}].sum(dim='time') +\
        da_gdp_cntry.sel(time=ds_dmg['death_year'].sel(birth_year=by).item()).drop('time') *\
        (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item()))\
        for by in birth_years],
        dim='birth_year',
    ).assign_coords({'birth_year':birth_years})     

    # make assignment of emergence mask to global emergence 
    ds_gdp['gdp_sum'].loc[{
        'birth_year':birth_years,
        'lat':ds_dmg.lat.data,
        'lon':ds_dmg.lon.data,                
    }] = xr.where(
            ds_dmg['country_extent'].notnull(),
            da_gdp_sum.loc[{'birth_year':birth_years,'lat':ds_dmg.lat.data,'lon':ds_dmg.lon.data}],
            ds_gdp['gdp_sum'].loc[{'birth_year':birth_years,'lat':ds_dmg.lat.data,'lon':ds_dmg.lon.data}],
        ).transpose('birth_year','lat','lon')
    
    
    
test=ds_gdp['gdp_sum'].sel(birth_year=2020)
test.plot()
plt.show()
test.sel(lat=slice(55.75,41.25),lon=slice(-6.25,16.75)).plot()
 

#%% ----------------------------------------------------------------
# gdp analysis; Wang & Sun 
# ------------------------------------------------------------------    

# use rioxarray to read geotiffs
import rioxarray as rxr
ssps = ('ssp1','ssp2','ssp3','ssp4','ssp5')
decades = ['2030','2040','2050','2060','2070','2080','2090','2100']
time_coords = decades.copy()
time_coords.insert(0,'2005')
time_coords = list(map(int,time_coords))
dims_dict = { # rename dimensions
    'x':'lon',
    'y':'lat',
}
ssp_colors={
    'ssp1':'forestgreen',
    'ssp2':'slategrey',
    'ssp3':'purple',
    'ssp4':'darkorange',
    'ssp5':'firebrick',
}

# read in tiffs as data array; concat all years, clean up in new array, pump out to netcdf
if len(glob.glob('./data/gdp_wang_sun/GDP_*_isimipgrid.nc4')) == 0:
    
    rds = {}
    for i,s in enumerate(ssps):
        rds[s] = {}
        rds[s]['2005'] = rxr.open_rasterio('./data/gdp_wang_sun/GDP2005.tif')
        for d in decades:
            rds[s][d] = rxr.open_rasterio('./data/gdp_wang_sun/GDP{}_{}.tif'.format(d,s))
        rds[s]['full_series'] = xr.concat(
            [rds[s][str(t)] for t in time_coords],
            dim='time'
        ).squeeze(dim='band').rename(dims_dict)
        rds[s]['full_series'].coords['time']=time_coords
        ds = rds[s]['full_series'].to_dataset(name=s)
        if i == 0:
            ds_fresh = xr.Dataset(
                data_vars={
                    s: (
                        ['time','lat','lon'],
                        np.full(
                            (len(ds.time.data),len(ds.lat.data),len(ds.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),        
                },
                coords={
                    'time': ('time', time_coords),
                    'lat': ('lat', ds.lat.data, {
                        'standard_name': 'latitude',
                        'long_name': 'latitude',
                        'units': 'degrees_north',
                        'axis': 'Y'
                    }),
                    'lon': ('lon', ds.lon.data, {
                        'standard_name': 'longitude',
                        'long_name': 'longitude',
                        'units': 'degrees_east',
                        'axis': 'X'
                    })
                }
            )
        else:
            ds_fresh[s] = xr.full_like(ds_fresh[ssps[0]],fill_value=np.nan)
        ds_fresh[s].loc[{
            'time':ds_fresh.time.data,
            'lat':ds_fresh.lat.data,
            'lon':ds_fresh.lon.data,
        }] = ds[s]
        ds_fresh[s].to_netcdf(
            './data/gdp_wang_sun/GDP_{}.nc4'.format(s),
            format='NETCDF4',
            encoding={
                'lat': {
                    'dtype': 'float64',
                },
                'lon': {
                    'dtype': 'float64',
                }            
            }
        )
        
# enter pycdo stuff for remapping to isimip grid        
        
else:

    # start again, read in netcdf that were regridded on command line
    for i,s in enumerate(ssps):
        if i == 0:
            ds_gdp_regrid = xr.open_dataset('./data/gdp_wang_sun/GDP_{}_isimipgrid.nc4'.format(s))
            ds_gdp_regrid.coords['time'] = time_coords
        else:
            other_ssp = xr.open_dataset('./data/gdp_wang_sun/GDP_{}_isimipgrid.nc4'.format(s))[s]
            other_ssp.coords['time'] = time_coords
            ds_gdp_regrid[s] = other_ssp
        
# associated years in da population
da_population_subset = da_population.loc[{'time':ds_gdp_regrid.time.values}]


# checking sample country time series for these ssps

legend_lw=3.5 # legend line width
x0 = 0.15 # bbox for legend
y0 = 0.7
xlen = 0.2
ylen = 0.2    
legend_entrypad = 0.5 # space between entries
legend_entrylen = 0.75 # length per entry
handles = [
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp1']),
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp2']),
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp3']),
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp4']),
    Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp5']),
]  
  
for cntry in ['Canada','United States','China', 'France', 'Germany']:
    
    print('')
    print('Entire series for {}'.format(cntry))  
    print('plots for {}'.format(cntry))
    f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    for s in ssps:
        
        # color
        clr = ssp_colors[s]
        
        ds_gdp_regrid[s].where(ds_gdp_regrid[s]>0).where(countries_mask==countries_regions.map_keys(cntry)).sum(dim=('lat','lon')).plot(
            ax=ax1,
            color=clr,
            add_legend='False'
        )
        
        da_gdp_regrid_s = ds_gdp_regrid[s]
        da_gdp_regrid_pc = da_gdp_regrid_s.where(countries_mask==countries_regions.map_keys(cntry)) / da_population_subset.where(da_population_subset>0).where(countries_mask==countries_regions.map_keys(cntry))
        da_gdp_regrid_pc_mean = da_gdp_regrid_pc.mean(dim=('lat','lon'))    
        da_gdp_regrid_pc_mean.plot(
            ax=ax2,
            color=clr,
            add_legend=False,
        )
    ax1.set_title('GDP: {}'.format(cntry))
    ax2.set_title('GDP per capita: {}'.format(cntry))
    ax2.legend(
        handles, 
        list(ssp_colors.keys()), 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        loc='lower left',
        ncol=1,
        fontsize=10, 
        mode="expand", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )     
    
    plt.show() 
    
# alternative testing for the Wang and Sun data while using ISIMIP historical GDP estimates
f_gdp_historical_1861_2005 = './data/isimip/gdp/gdp_histsoc_0p5deg_annual_1861_2005.nc4'
da_gdp_historical_1861_2005 = open_dataarray_isimip(f_gdp_historical_1861_2005)
da_gdp_historical_1861_2004 = da_gdp_historical_1861_2005.loc[{'time':np.arange(1960,2005)}] 
    
for cntry in ['Canada','United States','China', 'France', 'Germany']:
    
    print('')
    print('Entire series for {}'.format(cntry))  
    print('plots for {}'.format(cntry))
    f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

    for s in ssps:
        
        future = ds_gdp_regrid[s]
        historical = da_gdp_historical_1861_2004
        da_gdp_full = xr.concat([historical,future],dim='time')
        da_population_subset = da_population.loc[{'time':da_gdp_full.time.values}]    
        
        # color
        clr = ssp_colors[s]
        
        da_gdp_full.where(da_gdp_full>0).where(countries_mask==countries_regions.map_keys(cntry)).sum(dim=('lat','lon')).plot(
            ax=ax1,
            color=clr,
            add_legend='False'
        )
        
        da_gdp_full_pc = da_gdp_full.where(countries_mask==countries_regions.map_keys(cntry)) / da_population_subset.where(da_population_subset>0).where(countries_mask==countries_regions.map_keys(cntry))
        da_gdp_full_pc = da_gdp_full_pc.mean(dim=('lat','lon'))    
        da_gdp_full_pc.plot(
            ax=ax2,
            color=clr,
            add_legend=False,
        )
    ax1.set_title('GDP: {}'.format(cntry))
    ax2.set_title('GDP per capita: {}'.format(cntry))
    ax2.legend(
        handles, 
        list(ssp_colors.keys()), 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        loc='lower left',
        ncol=1,
        fontsize=10, 
        mode="expand", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )     
    
    plt.show()     

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
