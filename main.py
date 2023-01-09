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
flags['runs'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure'] = 0       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute exposure (i.e. produce and save exposure as pickle)
flags['exposure_cohort'] = 0       # 0: do not process ISIMIP runs to compute exposure across cohorts (i.e. load exposure pickle)
                                   # 1: process ISIMIP runs to compute exposure across cohorts (i.e. produce and save exposure as pickle)                            
flags['exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)
flags['emergence'] = 0      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)
flags['gridscale'] = 0      # 0: do not process grid scale analysis, load pickles
                            # 1: process grid scale analysis
flags['plot'] = 1

# TODO: add rest of flags


#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries = init()

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
    flags['gmt']
)

# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    d_countries,d_regions = all_country_data()

else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/pickles/country_info.pkl', 'rb'))
    
    # load regions pickle
    d_regions = pk.load(open('./data/pickles/region_info.pkl', 'rb'))    
    
# unpack country information
df_countries = d_countries['info_pop']
gdf_country_borders = d_countries['borders']
da_population = d_countries['population_map']
df_birthyears = d_countries['birth_years']
df_life_expectancy_5 = d_countries['life_expectancy_5']
d_cohort_size = d_countries['cohort_size']
d_all_cohorts = d_countries['all_cohorts']
countries_regions, countries_mask = d_countries['mask']    

# unpack region information
df_birthyears_regions = d_regions['birth_years']
df_life_expectancy_5_regions = d_regions['life_expectancy_5']
d_cohort_weights_regions = d_regions['cohort_size']
 
# --------------------------------------------------------------------
# load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta,d_pic_meta = load_isimip(
    flags['runs'],
    flags['gmt'],
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
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
        flags['extr'],
        flags['gmt'],
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
        flags['gmt'],
        flags['extr'],
        d_isimip_meta,
        df_countries,
        countries_regions,
        countries_mask,
        da_population,
        d_all_cohorts,
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
        flags['extr'],
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
    
    if not os.path.isfile('./data/pickles/cohort_per_birthyear.pkl'):
        
        # cohort conversion to data array (again)
        da_cohort_size = xr.DataArray(
            np.asarray([v for k,v in d_all_cohorts.items() if k in list(df_countries['name'])]),
            coords={
                'country': ('country', list(df_countries['name'])),
                'time': ('time', year_range),
                'ages': ('ages', np.arange(104,-1,-1)),
            },
            dims=[
                'country',
                'time',
                'ages',
            ]
        )
        
        # need new cohort dataset that has total population per birth year (using life expectancy info; each country has a different end point)
        da_cohort_aligned = calc_birthyear_align(
            da_cohort_size,
            df_life_expectancy_5,
        )
        
        # convert to dataset and add weights
        ds_cohorts = ds_cohort_align(da_cohort_aligned)
        
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
        flags['extr'],
        flags['gmt'],
        'strj',
    )
        
else: # load pickles
    
    # birth year aligned population
    with open('./data/pickles/cohort_per_birthyear.pkl', 'rb') as f:
        ds_cohorts = pk.load(f)
    
    # pop frac
    with open('./data/pickles/pop_frac_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],'strj'), 'rb') as f:
        ds_pf_strj = pk.load(f)                
    
    # age emergence           
    with open('./data/pickles/age_emergence_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],'strj'), 'rb') as f:
        ds_ae_strj = pk.load(f)      
        
    # # remote testing
    # with open('./data/pickles/age_emergence_{}_{}_{}_remote.pkl'.format(flags['extr'],flags['gmt'],'strj'), 'rb') as f:
    #     ds_ae_strj_remote = pk.load(f)
    # with open('./data/pickles/pop_frac_{}_{}_{}_remote.pkl'.format(flags['extr'],flags['gmt'],'strj'), 'rb') as f:
    #     ds_pf_strj_remote = pk.load(f)        
        
#%% ----------------------------------------------------------------
# grid scale
# ------------------------------------------------------------------

from gridscale import *

if flags['gridscale']:
    
    ds_le_gs, ds_ae_gs, ds_pf_gs = grid_scale_emergence(
        d_isimip_meta,
        d_pic_meta,
        flags['extr'],
        d_all_cohorts,
        df_countries,
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
d_le_gs_spatial = {}
for cntry in sample_countries:
    with open('./data/pickles/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],cntry), 'rb') as f:
        d_le_gs_spatial[cntry] = pk.load(f)

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
    
GMT_indices_plot = [0,10,19,28]
step_labels = {
    0:'1.0 °C',
    10:'2.0 °C',
    19:'3.0 °C',
    28:'4.0 °C',
}
birth_years_plot = np.arange(1960,2021,20)
x=36
y=8

col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.75   # linewidth for coastlines

# fonts
title_font = 20
cbtitle_font = 20
tick_font = 18
legend_font=12

letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']

# extent
east = 180
west = -180
north = 80
south = -60
extent = [west,east,south,north]

# placment lu trends cbar
cb_x0 = 0.925
cb_y0 = 0.1
cb_xlen = 0.01
cb_ylen = 1.75

# identify colors
cmap_whole = plt.cm.get_cmap('cividis')
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
cmap_50 = cmap_whole(0.95)  # yellow
cmap_55 = cmap_whole(0.99)

colors = [cmap55,cmap45,cmap25,cmap5,cmap_25,cmap_35,cmap_45]
cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
cmap_list.set_over(cmap_55)
levels_i = np.arange(0,16,5)
levels_f = np.arange(20,51,10)
levels = np.concatenate((levels_i,levels_f))
norm = mpl.colors.BoundaryNorm(levels,cmap_list.N)   

gdf_country_borders.loc[sample_countries,:]

f,axes = plt.subplots(
    nrows=len(GMT_indices_plot),
    ncols=len(birth_years_plot),
    figsize=(x,y),
    subplot_kw={'projection':ccrs.Robinson()},
)

cbax = f.add_axes([cb_x0,cb_y0,cb_xlen,cb_ylen])

i=0
for row,step in zip(axes,GMT_indices_plot):
    
   for ax,by in zip(row,birth_years_plot):
       
        for cntry in sample_countries:
           
            plottable = d_le_gs_spatial[cntry]['lifetime_exposure'].loc[{
                'GMT':step,
                'birth_year':by,
            }].mean(dim='run')

            plottable.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                levels=levels,
                colors=colors,
                cbar_ax=cbax,
                add_labels=False,
            )
            
            poly = gdf_country_borders.loc[cntry,'geometry']
            ax.add_geometries(
                poly,
                crs=ccrs.Robinson(),
                facecolor='none',
                edgecolor='0.5'
            )
            
        ax.coastlines(linewidth=cstlin_lw)
        # ax.set_extent(extent)
        ax.set_title(
            letters[i],
            loc='left',
            fontsize=title_font,
            fontweight='bold',
        )
        i+=1
        
        if step == GMT_indices_plot[0]:
            
            ax.set_title(
                by,
                loc='center',
                fontsize=title_font,
                fontweight='bold',
            )
            
        if by == birth_years_plot[0]:
            
            
            ax.text(
                -0.07, 0.55, 
                step_labels[step], 
                va='bottom', 
                ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                fontweight='bold',
                fontsize=title_font,
                rotation='vertical', 
                rotation_mode='anchor',
                transform=ax.transAxes
            )
    
# lu response pattern colorbar
cb = mpl.colorbar.ColorbarBase(
    ax=cbax, 
    cmap=cmap_list,
    norm=norm,
    spacing='proportional',
    orientation='vertical',
    extend='max',
    ticks=levels,
    drawedges=False,
)
cb.ax.tick_params(
    labelsize=title_font,
    # color=col_cbtic,
    # length=cb_ticlen,
    # width=cb_ticwid,
    # direction='out'
)



    
# comparison of weighted mean vs pixel scale (some prep required for pop frac from weighted mean)
if flags['plot']:
    
    plot_le_by_GMT_strj(
        ds_le,
        df_GMT_strj,
        ds_cohorts,
        flags['extr'],
        flags['gmt'],
    )    
        
    plot_pf_ae_by_GMT_strj(
        ds_pf_strj,
        ds_ae_strj,
        df_GMT_strj,
        ds_cohorts,
        flags['extr'],
        flags['gmt'],
    )

    # checking fraction of countries emerged from noise (appears to decrease over GMT trajectories per birth year, which)
    for step in GMT_labels:
        emerged_countries = xr.where(ds_ae_strj['age_emergence'].sel(GMT=step,birth_year=2000).mean(dim='run')>0,1,0).sum(dim='country') / len(ds_ae_strj.country.data)
        print(emerged_countries.item())
    
    

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
