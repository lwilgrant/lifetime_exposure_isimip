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
flags['run'] = 1          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure'] = 1       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute exposure (i.e. produce and save exposure as pickle)
flags['exposure_cohort'] = 1       # 0: do not process ISIMIP runs to compute exposure across cohorts (i.e. load exposure pickle)
                                   # 1: process ISIMIP runs to compute exposure across cohorts (i.e. produce and save exposure as pickle)                            
flags['exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)
flags['emergence'] = 1      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)
flags['birthyear_emergence'] = 1  # 0: only run calc_birthyear_align with birth years from 1960-2020
                                   # 1: run calc_birthyear_align with birth years from 1960-2100
flags['gridscale'] = 0      # 0: do not process grid scale analysis, load pickles
                            # 1: process grid scale analysis
flags['plot'] = 0

# TODO: add rest of flags


#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot = init()

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

if flags['exposure']: 
    
    start_time = time.time()
    
    # calculate exposure per country and per region and save data
    ds_le = calc_exposure(
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

    print('Loading processed exposures')

    # load lifetime exposure pickle
    with open('./data/pickles/exposure_{}_{}.pkl'.format(flags['extr'],flags['gmt']), 'rb') as f:
        ds_le = pk.load(f)

ds_le = calc_exposure_mmm_xr(ds_le)

# --------------------------------------------------------------------
# process exposure across cohorts

if flags['exposure_cohort']:
    
    start_time = time.time()
    
    # calculate exposure per country and per cohort
    calc_cohort_exposure(
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
# process picontrol data

if flags['exposure_pic']:
    
    start_time = time.time()
    
    # takes 38 mins crop failure
    d_exposure_perrun_pic = calc_exposure_pic(
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
    
    if not os.path.isfile('./data/pickles/cohort_per_birthyear.pkl'):
        
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
        
        # # global mean life expectancy
        # le_subset = pd.concat([df_life_expectancy_5.loc[:,c] for c in df_countries['name']],axis=1) # get relevant countries' life expectancy
        # da_le_subset = le_subset.to_xarray().to_array().rename({'index':'birth_year','variable':'country'})
        # da_gmle = da_le_subset.weighted(ds_cohorts['weights'].sel(birth_year=da_le_subset.birth_year.data)).mean(dim='country')
        # testdf = da_gmle.to_dataframe(name='gmle') # testing for comp against df_GMT_strj
        # not sure if there's logic to line limitation for plot_pop_frac_birth_year_GMT_strj()
            # need to ask, what is the common level of warming that 1960 birth year lives to in all trajectories (horizontal line in gmt trajects)
        
        # pickle birth year aligned cohort sizes and global mean life expectancy
        with open('./data/pickles/cohort_per_birthyear.pkl', 'wb') as f:
            pk.dump(ds_cohorts,f)  
        # with open('./data/pickles/global_mean_life_expectancy.pkl', 'wb') as f:
        #     pk.dump(da_gmle,f)
    
    else:
        
        # load pickled birth year aligned cohort sizes and global mean life expectancy
        with open('./data/pickles/cohort_per_birthyear.pkl', 'rb') as f:
            ds_cohorts = pk.load(f)          
        # with open('./data/pickles/global_mean_life_expectancy.pkl', 'rb') as f:
        #     da_gmle = pk.load(f)                      
    
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
    with open('./data/pickles/cohort_per_birthyear.pkl', 'rb') as f:
        ds_cohorts = pk.load(f)
    
    # pop frac
    with open('./data/pickles/pop_frac_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
        ds_pf_strj = pk.load(f)                
    
    # age emergence           
    with open('./data/pickles/age_emergence_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
        ds_ae_strj = pk.load(f)      
        
    # # remote testing
    # with open('./data/pickles/age_emergence_{}_{}_{}_remote.pkl'.format(flags['extr'],flags['gmt'],'strj'), 'rb') as f:
    #     ds_ae_strj_remote = pk.load(f)
    # with open('./data/pickles/pop_frac_{}_{}_{}_remote.pkl'.format(flags['extr'],flags['gmt'],'strj'), 'rb') as f:
    #     ds_pf_strj_remote = pk.load(f)        
        
#%% ----------------------------------------------------------------
# grid scale
# ------------------------------------------------------------------

# from gridscale import *

# if flags['gridscale']:
    
#     ds_le_gs, ds_ae_gs, ds_pf_gs = grid_scale_emergence(
#         d_isimip_meta,
#         d_pic_meta,
#         flags['extr'],
#         da_cohort_size,
#         df_countries,
#         countries_regions,
#         countries_mask,
#         df_life_expectancy_5,
#         GMT_indices,
#         da_population,
#     )
    
# else:
    
#     # load pickled aggregated lifetime exposure, age emergence and pop frac datasets
#     with open('./data/pickles/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['extr']), 'rb') as f:
#         ds_le_gs = pk.load(f)
#     with open('./data/pickles/gridscale_aggregated_age_emergence_{}.pkl'.format(flags['extr']), 'rb') as f:
#         ds_ae_gs = pk.load(f)
#     with open('./data/pickles/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['extr']), 'rb') as f:
#         ds_pf_gs = pk.load(f)

# # load spatially explicit datasets
# d_gs_spatial = {}
# for cntry in sample_countries:
#     with open('./data/pickles/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],cntry), 'rb') as f:
#         d_gs_spatial[cntry] = pk.load(f)

#%% ----------------------------------------------------------------
# plot emergence stuff
# ------------------------------------------------------------------

from plot import *

# plot pop frac and age emergence across GMT for stylized trajectories
# top panel; (y: frac unprecedented, x: GMT anomaly @ 2100)
# bottom panel; (y: age emergence, x: GMT anomaly @ 2100)
# plots both approaches to frac unprecedented; exposed vs full cohorts
# issue in these plots that early birth years for high warming trajectories won't live till 3-4 degree warming, but we misleadingly plot along these points
    # so, for e.g. 1970 BY, we need to limit the line up to life expectancy
    # will need cohort weighted mean of life expectancy across countries
    # 

#%% ----------------------------------------------------------------
# plot
# ------------------------------------------------------------------   
    
# comparison of weighted mean vs pixel scale (some prep required for pop frac from weighted mean)
if flags['plot']:
    
    plot_stylized_trajectories(
        df_GMT_strj,
        GMT_indices,
        d_isimip_meta,
    )    
    
    plot_le_by_GMT_strj(
        ds_le,
        df_GMT_strj,
        ds_cohorts,
        flags
    )    
        
    plot_pf_ae_by_GMT_strj(
        ds_pf_strj,
        ds_ae_strj,
        df_GMT_strj,
        ds_cohorts,
        flags['extr'],
        flags['gmt'],
    )
    
    plot_pf_t_GMT_strj(
        ds_pf_strj,
        ds_ae_strj,
        df_GMT_strj,
        ds_cohorts,
        flags['extr'],
        flags['gmt'],
    )    
    
    # # add emergence mask since i forgot to do it in gridscale.py (added it there but haven't rerun 10 Jan)
    # for cntry in sample_countries:
    #     d_gs_spatial[cntry]['emergence_mask'] = (['run','GMT','birth_year','lat','lon'],np.full(
    #                     (len(list(d_isimip_meta.keys())),len(GMT_indices),len(sample_birth_years),len(d_gs_spatial[cntry].lat.data),len(d_gs_spatial[cntry].lon.data)),
    #                     fill_value=np.nan,
    #                 ))
    #     for i in list(d_isimip_meta.keys()):
    #         for step in GMT_indices:
    #             if os.path.isfile('./data/pickles/gridscale_exposure_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],cntry,i,step)):
    #                 with open('./data/pickles/gridscale_exposure_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],cntry,i,step), 'rb') as f:
    #                     da_birthyear_exposure_mask = pk.load(f)
    #                     d_gs_spatial[cntry]['emergence_mask'].loc[{'run':i,'GMT':step}] = da_birthyear_exposure_mask.loc[{'birth_year':sample_birth_years}]
    
    # cntry = 'Canada'
    # ind_cntry = countries_regions.map_keys(cntry)
    # mask = xr.DataArray(
    #     np.in1d(countries_mask,ind_cntry).reshape(countries_mask.shape),
    #     dims=countries_mask.dims,
    #     coords=countries_mask.coords,
    # )
        
    # for cntry in sample_countries:
    #     ind_cntry = countries_regions.map_keys(cntry)
    #     mask = xr.DataArray(
    #         np.in1d(countries_mask,ind_cntry).reshape(countries_mask.shape),
    #         dims=countries_mask.dims,
    #         coords=countries_mask.coords,
    #     )        
    #     for analysis in ['lifetime_exposure','age_emergence','emergence_mask','population_emergence']:   
    #         if analysis == 'emergence_mask':
    #             d_gs_spatial[cntry][analysis] = d_gs_spatial[cntry][analysis].where(mask)
    #         if cntry != 'Russian Federation':
    #             projection = ccrs.PlateCarree()
    #         else:
    #             projection = ccrs.LambertConformal(central_longitude=36.6, central_latitude=53.7, cutoff=30)
    #         p = d_gs_spatial[cntry][analysis].loc[{
    #             'birth_year':birth_years_plot,
    #             'GMT':GMT_indices_plot,
    #         }].mean(dim='run').plot(
    #             col='birth_year',
    #             row='GMT',
    #             transform=ccrs.PlateCarree(),
    #             subplot_kws={"projection": projection},
    #             aspect=2,
    #             size=3
    #         )
    #         for ax in p.axes.flat:
    #             ax.coastlines()
    #             ax.gridlines()
    #         p.fig.savefig('./figures/gridscale_sample_{}_{}.png'.format(cntry,analysis))
    
                        
    # # spatial lifetime exposure dataset (subsetting birth years and GMT steps to reduce data load) per country
    #     # can also add spatial age emergence to here
    # ds_gs_spatial = xr.Dataset(
    #     data_vars={
    #         'lifetime_exposure': (
    #             ['run','GMT','birth_year','lat','lon'],
    #             np.full(
    #                 (len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(birth_years_plot),len(countries_mask.lat.data),len(countries_mask.lon.data)),
    #                 fill_value=np.nan,
    #             ),
    #         ),
    #         'age_emergence': (
    #             ['run','GMT','birth_year','lat','lon'],
    #             np.full(
    #                 (len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(birth_years_plot),len(countries_mask.lat.data),len(countries_mask.lon.data)),
    #                 fill_value=np.nan,
    #             ),
    #         ),
    #         'population_emergence': (
    #             ['run','GMT','birth_year','lat','lon'],
    #             np.full(
    #                 (len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(birth_years_plot),len(countries_mask.lat.data),len(countries_mask.lon.data)),
    #                 fill_value=np.nan,
    #             ),
    #         ),
    #         'emergence_mask': (
    #             ['run','GMT','birth_year','lat','lon'],
    #             np.full(
    #                 (len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(birth_years_plot),len(countries_mask.lat.data),len(countries_mask.lon.data)),
    #                 fill_value=np.nan,
    #             ), 
    #         )
    #     },
    #     coords={
    #         'lat': ('lat', countries_mask.lat.data),
    #         'lon': ('lon', countries_mask.lon.data),
    #         'birth_year': ('birth_year', birth_years_plot),
    #         'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
    #         'GMT': ('GMT', GMT_indices_plot)
    #     }
    # )
    
    # for cntry in sample_countries:
    #     for analysis in ['lifetime_exposure','age_emergence','emergence_mask']:
    #         ds_gs_spatial[analysis].loc[{
    #             'lat':d_gs_spatial[cntry].lat.data,
    #             'lon':d_gs_spatial[cntry].lon.data,
    #             'birth_year':birth_years_plot,
    #             'GMT':GMT_indices_plot,            
    #         }] = d_gs_spatial[cntry][analysis].loc[{
    #             'lat':d_gs_spatial[cntry].lat.data,
    #             'lon':d_gs_spatial[cntry].lon.data,
    #             'birth_year':birth_years_plot,
    #             'GMT':GMT_indices_plot,
    #         }]
    
    # ind_cntrs = []
    # for cntry in sample_countries:
    #     ind_cntrs.append(countries_regions.map_keys(cntry))
    # mask = xr.DataArray(
    #     np.in1d(countries_mask,ind_cntrs).reshape(countries_mask.shape),
    #     dims=countries_mask.dims,
    #     coords=countries_mask.coords,
    # )
    # ds_gs_spatial = ds_gs_spatial.where(mask)
    # plottable = ds_gs_spatial['lifetime_exposure'].mean(dim='run')
    # plottable.plot(transform=ccrs.PlateCarree(),col='birth_year',row='GMT',subplot_kws={'projection':ccrs.PlateCarree()})
    
    
    # for analysis in ['lifetime_exposure','age_emergence','emergence_mask']:
        
    #     gridscale_spatial(
    #         d_gs_spatial,
    #         analysis,
    #         countries_mask,
    #         countries_regions,
    #         flags['extr'],
    #     )

    # checking fraction of countries emerged from noise (appears to decrease over GMT trajectories per birth year, which)
    # for step in GMT_labels:
    #     emerged_countries = xr.where(ds_ae_strj['age_emergence'].sel(GMT=step,birth_year=2000).mean(dim='run')>0,1,0).sum(dim='country') / len(ds_ae_strj.country.data)
    #     print(emerged_countries.item())
    
    

    # # rename ds_pop_frac['mean_unprec_all'] to 'unprec_Fraction

    # for row,var in zip(axes,[ds_le['lifetime_exposure'],ds_ae['age_emergence'],ds_pf['unprec_fraction']]):
    #     ax,cntry in zip(row,sample_countries):
    #         frame = {
                
    #         }
    #         df
            

    # collect all data arrays for age of emergence into dataset for finding age per birth year
    # ds_age_emergence = xr.merge([
    #     ds_age_emergence_15.rename({'age_emergence':'age_emergence_15'}),
    #     ds_age_emergence_20.rename({'age_emergence':'age_emergence_20'}),
    #     ds_age_emergence_NDC.rename({'age_emergence':'age_emergence_NDC'}),
    # ])
            
    # # plot pop frac of 3 main GMT mapped scenarios across birth years
    # plot_pop_frac_birth_year(
    #     ds_pop_frac_NDC,
    #     ds_pop_frac_15,
    #     ds_pop_frac_20,
    #     year_range,
    # )

    # # plot pop frac for 0.8-3.5 degree stylized trajectories across birth years
    # plot_pop_frac_birth_year_strj(
    #     ds_pop_frac_strj,
    #     df_GMT_strj,
    # )

    # plot pop frac and age emergence across GMT for stylized trajectories
    # top panel; (y: frac unprecedented, x: GMT anomaly @ 2100)
    # bottom panel; (y: age emergence, x: GMT anomaly @ 2100)
    # plots both approaches to frac unprecedented; exposed vs full cohorts
    # issue in these plots that early birth years for high warming trajectories won't live till 3-4 degree warming, but we misleadingly plot along these points
        # so, for e.g. 1970 BY, we need to limit the line up to life expectancy
        # will need cohort weighted mean of life expectancy across countries
        # 

    # # plot country mean age of emergence of 3 main GMT mapped scenarios across birth year
    # plot_age_emergence(
    #     ds_age_emergence_NDC,
    #     ds_age_emergence_15,
    #     ds_age_emergence_20,
    #     year_range,
    # )

    # # plot country mean age of emergence of stylized trajectories across birth years
    # plot_age_emergence_strj(
    #     ds_age_emergence_strj,
    #     df_GMT_strj,
    #     ds_cohorts,
    #     year_range,
    # )

    # calculate birth year emergence in simple approach
    # age emergences here shouldn't be from isolated 1.5, 2.0 and NDC runs; should have function that takes these main scenarios from stylized trajectories
    # gdf_exposure_emergence_birth_year = calc_exposure_emergence(
    #     ds_exposure,
    #     ds_exposure_pic,
    #     ds_age_emergence,
    #     gdf_country_borders,
    # )

    # # country-scale spatial plot of birth and year emergence
    # spatial_emergence_plot(
    #     gdf_exposure_emergence_birth_year,
    #     flags['extr'],
    #     flags['gmt'],
    # )

    # # plot stylized trajectories (GMT only)
    # plot_stylized_trajectories(
    #     df_GMT_strj,
    #     d_isimip_meta,
    #     year_range,
    # )

    # # plot pop frac across GMT for stylized trajectories; add points for 1.5, 2.0 and NDC from original analysis as test
    # plot_pop_frac_birth_year_GMT_strj_points(
    #     ds_pop_frac_strj,
    #     ds_age_emergence_strj,
    #     df_GMT_strj,
    #     ds_cohorts,
    #     ds_age_emergence,
    #     ds_pop_frac_15,
    #     ds_pop_frac_20,
    #     ds_pop_frac_NDC,
    #     ind_15,
    #     ind_20,
    #     ind_NDC,
    #     year_range,
    # )

# %%
