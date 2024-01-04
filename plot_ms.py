# ---------------------------------------------------------------
# Functions to compute emergence of exposure from noise
# ----------------------------------------------------------------

#               
#%%  ----------------------------------------------------------------
# IMPORT AND PATH 
# ----------------------------------------------------------------

import os
import requests
from zipfile import ZipFile
import io
import xarray as xr
import pickle as pk
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
import mapclassify as mc
from copy import deepcopy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import regionmask as rm
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs
import seaborn as sns
import cartopy as cr
import cartopy.feature as feature
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

# %% ----------------------------------------------------------------
# Conceptual plot for emergence in one location,   

def plot_conceptual(
    da_cohort_size,
    countries_mask,
    countries_regions,
    d_isimip_meta,
    flags,
    df_life_expectancy_5,
):
    # get data
    cntry='Belgium'
    city_name='Brussels'
    # concept_bys = np.arange(1960,2021,30)
    concept_bys = np.arange(1960,2021,1)
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
    # brussels coords  
    city_lat = 50.8476
    city_lon = 4.3572   

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
    with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'rb') as f:
        ds_dmg = pk.load(f)                  

    # loop over simulations
    for i in list(d_isimip_meta.keys()): 

        print('simulation {} of {}'.format(i,len(d_isimip_meta)))

        # load AFA data of that run
        with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
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

    # mean for brussels            
    da_test_city = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').mean(dim='run')
    da_test_city = da_test_city.rolling(time=5,min_periods=5).mean()

    # standard deviation for brussels
    da_test_city_std = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').std(dim='run')
    da_test_city_std = da_test_city_std.rolling(time=5,min_periods=5).mean()

    # fill in 1st 4 years with 1s
    # first for mean
    for by in da_test_city.birth_year.data:
        for step in GMT_indices_plot:
            da_test_city.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5)}] = da_test_city.loc[{'birth_year':by,'GMT':step}].min(dim='time')
    # then for std        
    for by in da_test_city_std.birth_year.data:
        for step in GMT_indices_plot:
            da_test_city_std.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5)}] = da_test_city_std.loc[{'birth_year':by,'GMT':step}].min(dim='time')        
                
    # load PIC pickle
    with open('./data/{}/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],cntry), 'rb') as f:
        ds_pic = pk.load(f)   

    # plotting city lat/lon pixel doesn't give smooth kde
    df_pic_city = ds_pic['lifetime_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').to_dataframe().drop(columns=['lat','lon','quantile'])         
    da_pic_city_9999 = ds_pic['99.99'].sel({'lat':city_lat,'lon':city_lon},method='nearest')  

    # concept figure
    # ------------------------------------------------------------------   
    
    # plot building
    from mpl_toolkits.axes_grid1 import inset_locator as inset
    plt.rcParams['patch.linewidth'] = 0.1
    plt.rcParams['patch.edgecolor'] = 'k'
    colors = dict(zip(GMT_indices_plot,['steelblue','darkgoldenrod','darkred']))
    x=5
    y=1
    l = 0
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }

    # ------------------------------------------------------------------   
    # 1960 time series
    f,ax = plt.subplots(
        figsize=(x,y)
    )
    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':1960,'GMT':step}].plot.line(
            ax=ax,
            color=colors[step],
            linewidth=1,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':1960,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax,
            color=colors[step],
            linewidth=3,
            zorder=4,
        )
             
    end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])     
    ax.annotate(
        'Born in 1960',
        (1965,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )    
    ax.set_title(None)
    ax.annotate(
        letters[l],
        (1960,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    )    
    l+=1      
      
    ax.set_xlim(
        1960,
        end_year,
    )
    ax.set_ylim(
        0,
        da_pic_city_9999+1,
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.tick_params(colors='gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.hlines(
        y=da_pic_city_9999, 
        xmin=1960, 
        xmax=da_test_city.loc[{'birth_year':1960}].time.max()+10, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )

    # 1960 pdf
    ax_pdf_l = end_year+5
    ax_pdf_b = -2
    ax_pdf_w = 20
    ax_pdf_h = ax.get_ylim()[-1]+2
    ax_pdf = ax.inset_axes(
        bounds=(ax_pdf_l, ax_pdf_b, ax_pdf_w, ax_pdf_h),
        transform=ax.transData,
    )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        color='lightgrey',
        discrete = True,
        ax=ax_pdf
    )
    ax_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    for step in GMT_indices_plot:
        ax_pdf.hlines(
            y=da_test_city.loc[{'birth_year':1960,'GMT':step}].max(), 
            xmin=0, 
            xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
            colors=colors[step], 
            linewidth=1, 
            linestyle='-', 
            label=gmt_legend[step], 
            zorder=2
        )
    ax_pdf.spines['right'].set_visible(False)
    ax_pdf.spines['top'].set_visible(False)      
    ax_pdf.set_ylabel(None)
    ax_pdf.set_xlabel(None)
    ax_pdf.set_ylim(-2,ax.get_ylim()[-1])
    ax_pdf.tick_params(colors='gray')
    ax_pdf.spines['left'].set_color('gray')
    ax_pdf.spines['bottom'].set_color('gray')
    ax_pdf.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10,
    )
    l+=1
        
    # ------------------------------------------------------------------       
    # 1990 time series
    ax2_l = 1990
    ax2_b = da_pic_city_9999 *2
    ax2_w = np.floor(df_life_expectancy_5.loc[1990,cntry])
    ax2_h = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())
    ax2 = ax.inset_axes(
        bounds=(ax2_l, ax2_b, ax2_w, ax2_h),
        transform=ax.transData,
    )

    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':1990,'GMT':step}].plot.line(
            ax=ax2,
            color=colors[step],
            linewidth=1,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax2,
            color=colors[step],
            linewidth=3,
            zorder=4,
        )    
                  
    end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)
    ax2.set_yticks([0,5,10])
    ax2.set_yticklabels([None,5,10])  
    ax2.set_xticks(np.arange(1990,2071,10))      
    ax2.set_xticklabels([None,2000,None,2020,None,2040,None,2060,None])
    ax2.set_xlim(
        1990,
        end_year,
    )
    ax2.set_ylim(
        0,
        np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)  
    ax2.spines['left'].set_position(('data',1990)) 
    ax2.tick_params(colors='gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['bottom'].set_color('gray')
    
    ax2.annotate(
        'Born in 1990',
        (1995,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )
    ax2.set_title(None)
    ax2.annotate(
        letters[l],
        (1990,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    )     
    l+=1           

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        if np.any(da.notnull()): # testing for median
            da_t = da.time.where(da == da.min()).dropna(dim='time').item()
            emergences.append(da_t)
    first_emerge = np.min(emergences)

    ax2.hlines(
        y=da_pic_city_9999, 
        xmin=first_emerge, 
        xmax=end_year, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )        

    # 1990 pdf
    ax2_pdf_l = end_year+5
    ax2_pdf_b = -2
    ax2_pdf_w = 20
    ax2_pdf_h = ax2.get_ylim()[-1]+2
    ax2_pdf = ax2.inset_axes(
        bounds=(ax2_pdf_l, ax2_pdf_b, ax2_pdf_w, ax2_pdf_h),
        transform=ax2.transData,
    )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        color='lightgrey',
        discrete = True,
        ax=ax2_pdf
    )
    ax2_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    for step in GMT_indices_plot:
        ax2_pdf.hlines(
            y=da_test_city.loc[{'birth_year':1990,'GMT':step}].max(), 
            xmin=0, 
            xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
            colors=colors[step], 
            linewidth=1, 
            linestyle='-', 
            label=gmt_legend[step], 
            zorder=2
        )
    ax2_pdf.spines['right'].set_visible(False)
    ax2_pdf.spines['top'].set_visible(False)      
    ax2_pdf.set_ylabel(None)
    ax2_pdf.set_xlabel(None)
    ax2_pdf.set_ylim(-2,ax2.get_ylim()[-1])
    ax2_pdf.tick_params(colors='gray')
    ax2_pdf.spines['left'].set_color('gray')
    ax2_pdf.spines['bottom'].set_color('gray')
    ax2_pdf.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10,
    )
    l+=1   
    
    ax2_pdf.annotate(
        'Unprecedented\nlifetime\nexposure\nfor {} people'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':1990,'lat':city_lat,'lon':city_lon},method='nearest').item())))),
        (1.1,0.3),
        xycoords=ax2_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        # fontweight='bold',
    )             

    # ------------------------------------------------------------------   
    # 2020 time series
    ax3_l = 2020
    ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    ax3_w = np.floor(df_life_expectancy_5.loc[2020,cntry])
    ax3_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())
    ax3 = ax2.inset_axes(
        bounds=(ax3_l, ax3_b, ax3_w, ax3_h),
        transform=ax2.transData,
    )
    # plot mean lines
    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':2020,'GMT':step}].plot.line(
            ax=ax3,
            color=colors[step],
            linewidth=1,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax3,
            color=colors[step],
            linewidth=3,
            zorder=4,
        )    

    end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])      
    
    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.set_yticks([0,5,10,15,20,25])
    ax3.set_yticklabels([None,5,10,15,20,25])   
    ax3.set_xticks(np.arange(2020,2101,10))      
    ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    ax3.set_xlim(
        2020,
        end_year,
    )
    ax3.set_ylim(
        0,
        np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)  
    ax3.spines['left'].set_position(('data',2020))  
    ax3.tick_params(colors='gray')
    ax3.spines['left'].set_color('gray')
    ax3.spines['bottom'].set_color('gray')

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        if np.any(da.notnull()):
            da_t = da.time.where(da == da.min()).dropna(dim='time').item()
            emergences.append(da_t)
    first_emerge = np.min(emergences)

    ax3.hlines(
        y=da_pic_city_9999, 
        xmin=first_emerge, 
        xmax=end_year, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    ax3.annotate(
        'Born in 2020',
        (2025,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )
    ax3.set_title(None)
    ax3.annotate(
        letters[l],
        (2020,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    ) 
    l+=1      

    # 2020 pdf
    ax3_pdf_l = end_year+5
    ax3_pdf_b = -2
    ax3_pdf_w = 20
    ax3_pdf_h = ax3.get_ylim()[-1]+2
    ax3_pdf = ax3.inset_axes(
        bounds=(ax3_pdf_l, ax3_pdf_b, ax3_pdf_w, ax3_pdf_h),
        transform=ax3.transData,
    )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        color='lightgrey',
        discrete = True,
        ax=ax3_pdf
    )
    ax3_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    for step in GMT_indices_plot:
        ax3_pdf.hlines(
            y=da_test_city.loc[{'birth_year':2020,'GMT':step}].max(), 
            xmin=0, 
            xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
            colors=colors[step], 
            linewidth=1, 
            linestyle='-', 
            label=gmt_legend[step], 
            zorder=2
        )
    ax3_pdf.spines['right'].set_visible(False)
    ax3_pdf.spines['top'].set_visible(False)      
    ax3_pdf.set_ylabel(None)
    ax3_pdf.set_xlabel(None)
    ax3_pdf.set_ylim(-2,ax3.get_ylim()[-1])
    ax3_pdf.tick_params(colors='gray')
    ax3_pdf.spines['left'].set_color('gray')
    ax3_pdf.spines['bottom'].set_color('gray')
    ax3_pdf.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10,
    )
    l+=1  
    
    ax3_pdf.annotate(
        'Unprecedented\nlifetime\nexposure\nfor {} people'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item())))),
        (1.1,0.6),
        xycoords=ax3_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        # fontweight='bold',
    )                   

    # City name
    ax3.annotate(
        '{}, {}'.format(city_name,cntry),
        (1960,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=16,
        rotation='horizontal',
        color='gray',
    )

    # axis labels ===================================================================

    # x axis label (time)
    x_i=1950
    y_i=-10
    x_f=2040
    y_f=y_i 
    con = ConnectionPatch(
        xyA=(x_i,y_i),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con)   

    con_arrow_top = ConnectionPatch(
        xyA=(x_f-2,y_f+1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_top)  

    con_arrow_bottom = ConnectionPatch(
        xyA=(x_f-2,y_f-1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_bottom) 
    ax.annotate(
        'Time',
        ((x_i+x_f)/2,y_f+1),
        xycoords=ax.transData,
        fontsize=12,
        color='gray',
    )

    # y axis label (Cumulative heatwave exposure since birth)
    x_i=1950
    y_i=-10
    x_f=x_i
    y_f=y_i + 61
    con = ConnectionPatch(
        xyA=(x_i,y_i),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con)   

    con_arrow_left = ConnectionPatch(
        xyA=(x_f-2,y_f-1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_left)  

    con_arrow_right = ConnectionPatch(
        xyA=(x_f+2,y_f-1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_right) 

    ax.annotate(
        'Cumulative heatwave exposure since birth',
        (x_i-10,(y_i+y_f)/5),
        xycoords=ax.transData,
        fontsize=12,
        rotation='vertical',
        color='gray',
    )

    # legend ===================================================================

    # bbox
    x0 = 1.5
    y0 = 0.5
    xlen = 0.5
    ylen = 0.5

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 10
    legend_lw=2
    
    legendcols = list(colors.values())+['gray']+['lightgrey']
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2]),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=legendcols[3]),
        Rectangle((0,0),1,1,color=legendcols[4]),
    ]
    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',
        '99.99% pre-industrial \n lifetime exposure',
        'pre-industrial lifetime \n exposure histogram'
    ]
    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item()
    
    # getting estimate of all birth years that emerge in 1.5 and 3.5 pathways and how many these cohorts sum to
    valid_bys=da_test_city.birth_year.where(da_test_city.loc[{'GMT':6}].max(dim='time')>da_pic_city_9999)
    y1 = valid_bys.min(dim='birth_year')
    y2 = valid_bys.max(dim='birth_year')
    unprecedented=ds_dmg['by_population_y0'].sel(birth_year=np.arange(y1,y2+1),lat=city_lat,lon=city_lon,method='nearest').sum(dim='birth_year').round().item()    
    print('{} thousand unprecedented born in {} and later under pathway {}'.format(unprecedented/10**3,y1,6))
    
    valid_bys=da_test_city.birth_year.where(da_test_city.loc[{'GMT':24}].max(dim='time')>da_pic_city_9999)
    y1 = valid_bys.min(dim='birth_year')
    y2 = valid_bys.max(dim='birth_year')
    unprecedented=ds_dmg['by_population_y0'].sel(birth_year=np.arange(y1,y2+1),lat=city_lat,lon=city_lon,method='nearest').sum(dim='birth_year').round().item()    
    print('{} thousand unprecedented born in {} and later under pathway {}'.format(unprecedented/10**3,y1,24))        


#%% ----------------------------------------------------------------
# plotting pf heatmaps for grid scale across hazards with and without
# limiting simulations to show ensemble effects on GMT scaling

def plot_heatmaps_allhazards(
    df_GMT_strj,
    da_gs_popdenom,
    flags,
):
    
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    extremes = [
        'heatwavedarea',     
        'cropfailedarea', 
        'burntarea', 
        'driedarea', 
        'floodedarea', 
        'tropicalcyclonedarea',
    ]
    # extremes_labels = {
    #     'burntarea': '$\mathregular{PF_{Wildfires}}$',
    #     'cropfailedarea': '$\mathregular{PF_{Crop failures}}$',
    #     'driedarea': '$\mathregular{PF_{Droughts}}$',
    #     'floodedarea': '$\mathregular{PF_{Floods}}$',
    #     'heatwavedarea': '$\mathregular{PF_{Heatwaves}}$',
    #     'tropicalcyclonedarea': '$\mathregular{PF_{Tropical cyclones}}$',
    # }    
    extremes_labels = {    
        'heatwavedarea': '$\mathregular{CF_{Heatwaves}}$ [%]',
        'cropfailedarea': '$\mathregular{CF_{Crop failures}}$ [%]',
        'burntarea': '$\mathregular{CF_{Wildfires}}$ [%]',
        'driedarea': '$\mathregular{CF_{Droughts}}$ [%]',
        'floodedarea': '$\mathregular{CF_{Floods}}$ [%]',
        'tropicalcyclonedarea': '$\mathregular{CF_{Tropical cyclones}}$ [%]',
    }            

    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    
    levels_hw=np.arange(0,101,10)
    levels_cf=np.arange(0,31,5)
    levels_other=np.arange(0,16,1)
    
    # --------------------------------------------------------------------
    # population fractions with simulation limits to avoid dry jumps
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)
        with open('./data/{}/{}/isimip_metadata_{}_ar6_rm.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)        
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)  
        if extr != 'cropfailedarea':
            p = ds_pf_gs_extr['unprec'].loc[{
                'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
                'run':sims_per_step[GMT_labels[-1]]
            }].sum(dim='country')
        else:
            p = ds_pf_gs_extr['unprec'].loc[{
                'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
                'run':sims_per_step[27]
            }].sum(dim='country')            
        p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})
    
    # plot
    mpl.rcParams['xtick.labelcolor'] = 'gray'
    mpl.rcParams['ytick.labelcolor'] = 'gray'
    x=14
    y=7
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )

    for ax,extr in zip(axes.flatten(),extremes):
        if extr == 'heatwavedarea':
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_hw,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,101,20)}
            ) 
        elif extr == 'cropfailedarea':
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_cf,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,31,5)}
            )         
        else:
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_other,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,16,3)}
            )       
        
    # ax stuff
    l=0
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
            color='gray',
            fontsize=12,
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10,
        )  
        l+=1                  
        ax.spines['right'].set_color('gray')
        ax.spines['top'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

        if not np.isin(n,[0,3]):
            ax.yaxis.set_ticklabels([])
        if n == 0:
            ax.annotate(
                    'GMT warming by 2100 [°C]',
                    (-.3,-0.6),
                    xycoords=ax.transAxes,
                    fontsize=12,
                    rotation='vertical',
                    color='gray',
                    fontweight='bold',        
                )            
        if n <= 2:
            ax.tick_params(labelbottom=False)    
        if n >= 3:
            ax.set_xlabel('Birth year',fontsize=12,color='gray')         
 
    f.savefig('./ms_figures/pf_heatmap_combined_simlim.png',dpi=1000,bbox_inches='tight')
    f.savefig('./ms_figures/pf_heatmap_combined_simlim.eps',format='eps',bbox_inches='tight')
    plt.show()     
    
    # --------------------------------------------------------------------
    # population fractions with all simulations
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        p = ds_pf_gs_extr['unprec'].loc[{
            'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
        }].sum(dim='country')       
        p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})

    # plot
    mpl.rcParams['xtick.labelcolor'] = 'gray'
    mpl.rcParams['ytick.labelcolor'] = 'gray'
    x=14
    y=7
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        if extr == 'heatwavedarea':
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_hw,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,101,20)}
            ) 
        elif extr == 'cropfailedarea':
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_cf,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,31,5)}
            )         
        else:
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_other,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,16,3)}
            )         
        
        ax.set_yticks(
            ticks=GMT_indices_ticks,
            labels=gmts2100,
            color='gray',
        )
        ax.set_xticks(
            ticks=np.arange(1960,2025,10),
            color='gray',
        )    
        
    # ax stuff
    l=0
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
            color='gray',
            fontsize=12,
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10,
        )  
        l+=1                  
        ax.spines['right'].set_color('gray')
        ax.spines['top'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        if not np.isin(n,[0,3]):
            ax.yaxis.set_ticklabels([])
        else:
            pass
        if n == 0:
            ax.annotate(
                    'GMT warming by 2100 [°C]',
                    (-.3,-0.6),
                    xycoords=ax.transAxes,
                    fontsize=12,
                    rotation='vertical',
                    color='gray',
                    fontweight='bold',        
                )            
        if n <= 2:
            ax.tick_params(labelbottom=False)    
        if n >= 3:
            ax.set_xlabel('Birth year',fontsize=12,color='gray')
    
    f.savefig('./ms_figures/pf_heatmap_combined_allsims.png',dpi=1000,bbox_inches='tight')
    f.savefig('./ms_figures/pf_heatmap_combined_allsims.pdf',dpi=500,bbox_inches='tight')    
    f.savefig('./ms_figures/pf_heatmap_combined_allsims.eps',format='eps',bbox_inches='tight')
    plt.show()         

#%% ----------------------------------------------------------------
# plot of locations of emergence

def plot_emergence_union(
    grid_area,
    da_emergence_mean,
):
    
    x=11
    y=8.1
    markersize=10
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = '0'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    
    density=6  # density of hatched lines showing frac of sims with emergence
    sim_frac=0.50  # fraction of sims to show emergence (0.25 for original one)
    gmt = 17 # gmt index to compare multihazard pf
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
    colors=[
        mpl.colors.to_rgb('steelblue'),
        mpl.colors.to_rgb('darkgoldenrod'),
        mpl.colors.to_rgb('peru'),
    ]
    cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

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
    norm=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(8,2)
    gs0.update(wspace=0.25,hspace=0.2)
    ax0 = f.add_subplot(gs0[5:8,0:1],projection=ccrs.Robinson())

    # left side for 1960
    # masp per hazard
    gsn0 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,0:1],
        wspace=0,
        hspace=0,
    )
    ax00 = f.add_subplot(gsn0[0],projection=ccrs.Robinson())
    ax10 = f.add_subplot(gsn0[1],projection=ccrs.Robinson())
    ax20 = f.add_subplot(gsn0[2],projection=ccrs.Robinson()) 

    ax01 = f.add_subplot(gsn0[3],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gsn0[4],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gsn0[5],projection=ccrs.Robinson())       

    # colorbars
    pos0 = ax0.get_position()
    # colorbar for emergence
    cax_e = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.075,
        pos0.width*.15,
        pos0.height*0.25
    ])
    # colorbar for union
    cax_u = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.175,
        pos0.width*1.5,
        pos0.height*0.15
    ])

    # right side for 2020
    ax1 = f.add_subplot(gs0[5:8,1:2],projection=ccrs.Robinson()) # map of emergence union
    gsn1 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,1:2],
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
    l=0

    ax00.annotate(
        '1960 birth cohort',
        (0.55,1.3),
        xycoords=ax00.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )          

    i+=1

    # raster of falses to populate based on regions where 25% of projections emerge
    template_1960 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=gmt,birth_year=1960),
        False
    )

    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p1960 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':1960,
        }]
        ax.contourf(
            p1960.lon.data,
            p1960.lat.data,
            p1960.where(p1960>sim_frac).notnull(),
            levels=[.5,1.5],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[density*'/',density*'/'],
            rasterized=True,
            zorder=10
        )       
        template_1960 = template_1960+p1960.where(p1960>sim_frac).notnull()
        p1960 = xr.where(p1960>0,1,0)
        p1960 = p1960.where(p1960).where(mask.notnull())*3
        p1960.plot(
            ax=ax,
            cmap=cmap_list,
            levels=levels,
            add_colorbar=False,
            add_labels=False,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=5
        )    
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray'
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1    
        i+=1
        
    p_u1960 = template_1960.where(mask.notnull())

    ax0.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax0.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u1960.plot(
        ax=ax0,
        cmap=cmap_list_union,
        levels=union_levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1                 
    i+=1

    # 2020 birth cohort
    ax02.annotate(
        '2020 birth cohort',
        (0.55,1.3),
        xycoords=ax02.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )       

    # list of hatched booleans for 2020
    template_2020 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=gmt,birth_year=2020),
        False
    )

    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p2020 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':2020,
        }]
        ax.contourf(
            p2020.lon.data,
            p2020.lat.data,
            p2020.where(p2020>sim_frac).notnull(),
            levels=[.5,1.5],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[density*'/',density*'/'],
            rasterized=True,
            zorder=10
        )      
        # test = p2020.where(p2020>sim_frac).notnull()
        # ax.contourf(
        #     p2020.lon.data,
        #     p2020.lat.data,
        #     test,
        #     levels=[.5,1.5],
        #     colors='k',
        #     transform=ccrs.PlateCarree(),
        #     rasterized=True,
        #     zorder=10
        # )          
        template_2020 = template_2020+p2020.where(p2020>sim_frac).notnull()           
        p2020 = xr.where(p2020>0,1,0)
        p2020 = p2020.where(p2020).where(mask.notnull())*3
        p2020.plot(
            ax=ax,
            cmap=cmap_list,
            levels=levels,
            add_colorbar=False,
            add_labels=False,
            rasterized=True,
            transform=ccrs.PlateCarree(),
            zorder=5
        )    
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray',
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1          
        i+=1  
        
    p_u2020 = template_2020.where(mask.notnull())
    ax1.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax1.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u2020.plot(
        ax=ax1,
        cmap=cmap_list_union,
        levels=union_levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=5
    )
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1  

    # colorbar for emergence per extreme
    colors_emergence_cb = ['white','peru']
    cmap_list_emergence_cb = mpl.colors.ListedColormap(colors_emergence_cb,N=len(colors_emergence_cb))
    levels_emergence_cb = np.arange(-0.5,1.6,1)
    norm_e=mpl.colors.BoundaryNorm(levels_emergence_cb,ncolors=len(levels_emergence_cb)-1)
    cb_e = mpl.colorbar.ColorbarBase(
        ax=cax_e, 
        cmap=cmap_list_emergence_cb,
        # norm=norm,
        orientation='vertical',
        spacing='uniform',
        ticks=[0.25,0.75],
        drawedges=False,
    )
    cb_e.ax.set_yticklabels(['No emergence in ensemble','Emergence in ensemble'])

    cb_e.ax.set_title(
        '(' + r"$\bf{a}$" + '-' + r"$\bf{f}$" + ',' + r"$\bf{h}$" + '-' + r"$\bf{m}$"+')',
        fontsize=10,
    )
    cb_e.ax.xaxis.set_label_position('top')
    cb_e.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=10,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )                 

    # colorbar for union of emergence across extremes
    cb_u = mpl.colorbar.ColorbarBase(
        ax=cax_u, 
        cmap=cmap_list_union,
        norm=norm,
        orientation='horizontal',
        extend='max',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )
    cb_u.ax.set_title(
        '(' + r"$\bf{g}$" + ',' + r"$\bf{n}$" + ')',
        y=1.06,
        fontsize=10,
        loc='left',
    )

    cb_u.set_label(
        'Number of emerged extremes',
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

    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    # eu=da_emergence_union.loc[{'GMT':17,'birth_year':2020}].where(mask.notnull())
    la_frac_eu_gteq3_2020 = xr.where(p_u2020>=3,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100
    la_frac_eu_gteq3_1960 = xr.where(p_u1960>=3,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100
    print('2020 percentage of land area \n with emergence of 3 extremes is {}'.format(la_frac_eu_gteq3_2020.item()))    
    print('1960 percentage of land area \n with emergence of 3 extremes {}'.format(la_frac_eu_gteq3_1960.item()))  
        
    f.savefig('./ms_figures/emergence_union_new_gmt17_50perc.png',dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/emergence_union_new_25perc.pdf',dpi=500,bbox_inches='tight')

# %% ----------------------------------------------------------------
# Combined plot for heatwaves showing box plots of 1.5, 2.5 and 3.5, 
# t series of 2020 by and maps of 2020 by for 1.5, 2.5 and 3.5         

def plot_combined(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
):
        
    x=12
    y=10
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
    gs0 = gridspec.GridSpec(10,2)
    # gs0.update(hspace=0.8,wspace=0.8)
    ax00 = f.add_subplot(gs0[0:8,0:1]) # box plots
    # ax10 = f.add_subplot(gs0[2:,0:2]) # tseries for 2020 by
    gs00 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:7,1:],
        # top=0.8
    )
    ax01 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 
    pos00 = ax21.get_position()
    cax00 = f.add_axes([
        pos00.x0-0.0775,
        pos00.y0-0.075,
        pos00.width*2.2,
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
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
        ds_pf_gs_plot = pk.load(file)
    da_p_gs_plot = ds_pf_gs_plot['unprec'].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }]
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        print('step {}'.format(step))
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
        ax=ax00,
    )
    p.legend_.remove()                  
    ax00.spines['right'].set_visible(False)
    ax00.spines['top'].set_visible(False)      
    ax00.tick_params(colors='gray')
    ax00.set_ylim(0,100)
    ax00.spines['left'].set_color('gray')
    ax00.spines['bottom'].set_color('gray')      
    ax00.set_ylabel('$\mathregular{PF_{Heatwaves}}$',color='gray',fontsize=14)
    ax00.set_xlabel('Birth year',color='gray',fontsize=14)       
    ax00.set_title(
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

    ax00.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=1,
        fontsize=legend_font, 
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
    map_letters = {24:'d',15:'c',6:'b'}
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
        l+=1          
        
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
            coordsA=ax00.transAxes,
            coordsB=ax.transAxes,
            color='gray'
        )
        ax00.add_artist(con)          
        
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
        '$\mathregular{PF_{Heatwaves}}$ for 2020 birth cohort',
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

    # f.savefig('./ms_figures/combined_plot_{}.png'.format(flags['extr']),dpi=1000,bbox_inches='tight')

    plt.show()            
    
#%% ----------------------------------------------------------------
# plot tseries, pie charts and maps of heatwave PF
# ------------------------------------------------------------------        

def plot_combined_piechart(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
    df_countries,
):
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

    # pie charts
    gs10 = gridspec.GridSpecFromSubplotSpec(
        1,
        3, 
        subplot_spec=gs0[7:,0:2],
        # top=0.8
    )
    ax00 = f.add_subplot(gs10[0])
    ax10 = f.add_subplot(gs10[1])
    ax20 = f.add_subplot(gs10[2]) 

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
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
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
        
        # pointers connecting 2020, GMT step box plot to map panels ------------------
        # if step == gmt_indices_152535[0]:
        #     x_h=1 
        # elif step == gmt_indices_152535[1]:
        #     x_h=0.95                      
        # elif step == gmt_indices_152535[-1]:
        #     x_h=0.9
        # y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        # x_m=0
        # y_m=0.5
        # con = ConnectionPatch(
        #     xyA=(x_h,y_h),
        #     xyB=(x_m,y_m),
        #     coordsA=ax0.transAxes,
        #     coordsB=ax.transAxes,
        #     color='gray'
        # )
        # ax0.add_artist(con)   
        
        # triangles showing connections between GMT step box plot to map panels ---------------
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        if step == gmt_indices_152535[0]:
            x_h=1 
        elif step == gmt_indices_152535[1]:
            x_h=0.95                      
        elif step == gmt_indices_152535[-1]:
            x_h=0.9
        y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        x_m=0
        y_m_i=0
        y_m_f=1.0
        con_low = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m_i),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        con_hi = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m_f),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )   
        con_vert = ConnectionPatch(
            xyA=(x_m,y_m_i),
            xyB=(x_m,y_m_f),
            coordsA=ax.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        
        line_low = con_low.get_path().vertices
        line_hi = con_hi.get_path().vertices
        
        ax0.add_artist(con_low)  
        ax0.add_artist(con_hi) 
        
        tri_coords = np.stack(
            (con_low.get_path().vertices[0],con_low.get_path().vertices[-1],con_hi.get_path().vertices[-1]),
            axis=0,
        )
        triangle = plt.Polygon(tri_coords,ec='lightgray',fc='lightgray',alpha=0.5,zorder=0,clip_on=False)    
        ax0.add_artist(triangle)           
        
        # triangles showing connections between GMT step box plot to map panels ---------------
             
        
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


    income_unprec = {}
    da_unprec = ds_pf_gs['unprec']
    for category in list(income_countries.keys()):
        income_unprec[category] = da_unprec.loc[{
            'country':income_countries[category],
            'GMT':gmt_indices_sample,
        }]

        
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)
            
    pi_totals = {}
    pi_ratios = {}
    for by in by_sample:
        
        # populate each birth year with totals per income group
        pi_totals[by] = [da_gs_popdenom.loc[{
            'country':income_countries[category],
            'birth_year':by,
        }].sum(dim='country').item() for category in list(income_countries.keys())]
        
        
        pi_ratios[by] = {}
        for category in list(income_countries.keys()):
            pi_ratios[by][category] = {}
            for step in gmt_indices_sample:
                unprec = income_unprec[category].loc[{
                    'GMT':step,
                    'run':sims_per_step[step],
                    'birth_year':by
                }].sum(dim='country').mean(dim='run').item()
                pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                    'country':income_countries[category],
                    'birth_year':by,
                }].sum(dim='country').item()
        

    for by,ax in zip(by_sample,[ax00,ax10,ax20]):
        for i,category in enumerate(list(income_countries.keys())):
            order = np.argsort(list(pi_ratios[by][category].values())) # indices that would sort pf across gmt steps
            order = order[::-1] # want decreasing pf order, so we reverse the indices
            ordered_gmts = [gmt_indices_sample[o] for o in order]
            for step in ordered_gmts:
                colors_list =['None']*4
                colors_list[i] = 'white'
                # first paint white where we want actual color (semi transparent, white makes it look good) per category with nothing in other categories
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }     
                )
                # then paint actual color and radius
                colors_list[i] = colors[step]
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }
                )                     
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        ax.set_title(
            by,
            loc='center',
            fontweight='bold',
            fontsize=12,
            color='gray',       
        )        
        l+=1             
        percents = ['25%','50%','75%','100%']
        for i,r in enumerate(np.arange(0.25,1.01,0.25)):
            if r < 1:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    }
                )      
            else:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    labels=list(ig_dict.values()),
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    },
                    textprops={
                        'color':'0.5'
                    }
                )        
            
            if by == 1960:
                ax.annotate(
                    percents[i],
                    xy=(0,r+0.05),
                    color='gray',
                    ha='center',
                    fontsize=6
                )
                
        # triangles showing connections between box plot years on x axis to pie chart years ---------------
        if ax == ax00:
            x_b=0.075 
            x_p_i=x_b - x_b/2
            x_p_f=x_b + x_b*2
        elif ax == ax10:
            x_b=0.5   
            x_p_i=0.425
            x_p_f=0.575                         
        elif ax == ax20:
            x_b=0.925
            x_p_i=x_b - 0.075*2
            x_p_f=x_b + 0.075/2
            
        y_b=-0.1
        y_p=-0.35

        con_le = ConnectionPatch(
            xyA=(x_b,y_b),
            xyB=(x_p_i,y_p),
            coordsA=ax0.transAxes,
            coordsB=ax0.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        con_ri = ConnectionPatch(
            xyA=(x_b,y_b),
            xyB=(x_p_f,y_p),
            coordsA=ax0.transAxes,
            coordsB=ax0.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )   
        con_ho = ConnectionPatch(
            xyA=(x_p_i,y_p),
            xyB=(x_p_f,y_p),
            coordsA=ax0.transAxes,
            coordsB=ax0.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        
        line_le = con_le.get_path().vertices
        line_ri = con_ri.get_path().vertices
        line_ho = con_ho.get_path().vertices
        
        ax0.add_artist(con_le)  
        ax0.add_artist(con_ri) 
        ax0.add_artist(con_ho)  
        
        pi_tri_coords = np.stack(
            (con_le.get_path().vertices[0].copy(),
            con_le.get_path().vertices[-1].copy(),
            con_ri.get_path().vertices[-1].copy()),
            axis=0,
        )
        pi_tri = plt.Polygon(pi_tri_coords,ec='lightgray',fc='lightgray',alpha=0.5,zorder=0,clip_on=False)    
        ax0.add_artist(pi_tri)                    
            
    for i,k in enumerate(list(ig_dict.keys())):
        ax0.annotate(
            '{}: {}'.format(ig_dict[k],k),
            xy=(0,-1.1-i*0.1),
            color='gray',
            # ha='center',
            xycoords=ax0.transAxes
        )
            
    f.savefig('./ms_figures/combined_plot_piecharts_new.png',dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/combined_plot_piecharts_50.pdf',dpi=50,bbox_inches='tight')
    # f.savefig('./ms_figures/combined_plot_piecharts_500.pdf',dpi=500,bbox_inches='tight')

#%% ----------------------------------------------------------------
# combined plot showing absolute cohort sizes and pie charts
# ------------------------------------------------------------------

def plot_combined_population(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
    df_countries,
):

    # plot characteristics
    x=12
    y=7
    markersize=10
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    by=2020
    cmap_whole = plt.cm.get_cmap('Reds')
    levels_cmap = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels_cmap[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    levels = np.arange(0,101,5)
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)
    l = 0 # letter indexing
    gmt_indices_152535 = [24,15,6]
    map_letters = {24:'g',15:'f',6:'e'}

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(10,3)

    # box plots
    ax0 = f.add_subplot(gs0[0:4,0:2]) 

    # pop totals
    ax1 = f.add_subplot(gs0[4:,0:2],sharex=ax0)

    gs0.update(hspace=0)

    # maps
    gs01 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:9,2:],
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

    # get data
    df_list_gs = []
    extr='heatwavedarea'
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
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
                
    # --------------------------------------------------------------------
    # pf boxplot time series
    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)

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
    x0 = 0.025
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

    # --------------------------------------------------------------------
    # populations

    ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)      
    ax1.tick_params(colors='gray')
    # ax1.set_ylim(0,100)
    ax1.spines['top'].set_color('gray')   
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')    

    colors_pop = {
        'Low income': plt.cm.get_cmap('tab20b')(1),
        'Lower middle income': plt.cm.get_cmap('tab20b')(5),
        'Upper middle income': plt.cm.get_cmap('tab20b')(9),
        'High income': plt.cm.get_cmap('tab20b')(13),
    }
    incomegroups = df_countries['incomegroup'].unique()
    income_countries = {}
    for category in incomegroups:
        income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])

    heights={}
    for category in incomegroups:
        heights[category] = da_gs_popdenom.loc[{
            'birth_year':np.arange(1960,2021,10),'country':income_countries[category]
        }].sum(dim='country').values / 10**6
    testdf = pd.DataFrame(heights)    
    testdf['birth_year'] = np.arange(1960,2021,10)
    testdf = testdf.set_index('birth_year')             
    p1 = testdf.plot(
        kind='bar',
        stacked=True,
        color=colors_pop,      
        ax=ax1,
        legend=False,
        rot=0.5
    ) 
    ax1.invert_yaxis()
    ax1.set_xlabel('Birth year',color='gray',fontsize=14)
    ax1.set_ylabel('Global cohort \n totals [in millions]',color='gray',fontsize=14)  
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )  

    # bbox for legend
    x0 = 0.025
    y0 = 0.0
    xlen = 0.2
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75
    legend_font = 10
    legend_lw=3.5   

    legendcols = list(colors_pop.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2]),\
        Rectangle((0,0),1,1,color=legendcols[3])
    ]

    labels= list(colors_pop.keys())

    ax1.legend(
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
    
    l+=1

    # --------------------------------------------------------------------
    # pie charts

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

    # --------------------------------------------------------------------
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways

    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]
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

    f.savefig('./ms_figures/combined_plot_popsizes.png',dpi=1000)
#%% ----------------------------------------------------------------
# combined plot showing absolute cohort sizes and pie charts
# ------------------------------------------------------------------

def plot_combined_population_piechart(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
    df_countries,
):

    # plot characteristics
    x=12
    y=7
    markersize=10
    labelfontsize=12
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    by=2020
    cmap_whole = plt.cm.get_cmap('Reds')
    levels_cmap = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels_cmap[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    levels = np.arange(0,101,5)
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)
    l = 0 # letter indexing
    gmt_indices_152535 = [24,15,6]
    map_letters = {24:'g',15:'f',6:'e'}

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(12,3)

    # box plots
    ax0 = f.add_subplot(gs0[0:4,0:2]) 

    # pop totals
    # ax1 = f.add_subplot(gs0[3:7,0:2],sharex=ax0)
    ax1 = f.add_subplot(gs0[4:7,0:2])

    gs0.update(hspace=0)

    # pie charts
    gs10 = gridspec.GridSpecFromSubplotSpec(
        1,
        3, 
        subplot_spec=gs0[9:,0:2],
        # top=0.8
    )
    ax00 = f.add_subplot(gs10[0])
    ax10 = f.add_subplot(gs10[1])
    ax20 = f.add_subplot(gs10[2]) 

    # maps
    gs01 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:10,2:],
    )
    ax01 = f.add_subplot(gs01[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs01[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs01[2],projection=ccrs.Robinson()) 
    pos00 = ax21.get_position()
    cax00 = f.add_axes([
        pos00.x0-0.05,
        pos00.y0-0.15,
        pos00.width*1.95,
        pos00.height*0.2
    ])

    # get data
    df_list_gs = []
    extr='heatwavedarea'
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
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
                
    # --------------------------------------------------------------------
    # pf boxplot time series
    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)

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
    ax0.set_ylabel('$\mathregular{CF_{Heatwaves}}$ [%]',color='gray',fontsize=labelfontsize)
    ax0.set_xlabel('Birth year',color='gray',fontsize=labelfontsize)       
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    l+=1 

    # bbox
    x0 = 0.01
    y0 = 0.7
    xlen = 0.2
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 8
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

    # --------------------------------------------------------------------
    # populations

    ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)      
    ax1.tick_params(colors='gray')
    # ax1.set_ylim(0,100)
    ax1.spines['top'].set_color('gray')   
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')    

    colors_pop = {
        'Low income': plt.cm.get_cmap('tab20b')(1),
        'Lower middle income': plt.cm.get_cmap('tab20b')(5),
        'Upper middle income': plt.cm.get_cmap('tab20b')(9),
        'High income': plt.cm.get_cmap('tab20b')(13),
    }
    incomegroups = df_countries['incomegroup'].unique()
    income_countries = {}
    for category in incomegroups:
        income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])

    heights={}
    for category in incomegroups:
        heights[category] = da_gs_popdenom.loc[{
            'birth_year':np.arange(1960,2021,10),'country':income_countries[category]
        }].sum(dim='country').values / 10**6
    testdf = pd.DataFrame(heights)    
    testdf['birth_year'] = np.arange(1960,2021,10)
    testdf = testdf.set_index('birth_year')             
    p1 = testdf.plot(
        kind='bar',
        stacked=True,
        color=colors_pop,      
        ax=ax1,
        legend=False,
        rot=0.5
    ) 
    ax1.invert_yaxis()
    ax1.set_xlabel('Birth year',color='gray',fontsize=labelfontsize)
    ax1.set_ylabel('Global cohort \n totals [in millions]',color='gray',fontsize=labelfontsize)  
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )  

    # bbox for legend
    # x0 = 0.01
    # y0 = 0.08
    # xlen = 0.2
    # ylen = 0.3
    x0 = 0.01
    y0 = -0.1
    xlen = 0.3
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75
    # legend_font = 10
    legend_lw=3.5   

    legendcols = list(colors_pop.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2]),\
        Rectangle((0,0),1,1,color=legendcols[3])
    ]

    # labels= list(colors_pop.keys())
    labels = list(ig_dict.values())

    ax1.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=4,
        fontsize=legend_font, 
        labelcolor='gray',
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )       
    
    l+=1

    # --------------------------------------------------------------------
    # pie charts

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

    income_unprec = {}
    da_unprec = ds_pf_gs['unprec']
    for category in list(income_countries.keys()):
        income_unprec[category] = da_unprec.loc[{
            'country':income_countries[category],
            'GMT':gmt_indices_sample,
        }]

        
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)
            
    pi_totals = {}
    pi_ratios = {}
    for by in by_sample:
        
        # populate each birth year with totals per income group
        pi_totals[by] = [da_gs_popdenom.loc[{
            'country':income_countries[category],
            'birth_year':by,
        }].sum(dim='country').item() for category in list(income_countries.keys())]
        
        
        pi_ratios[by] = {}
        for category in list(income_countries.keys()):
            pi_ratios[by][category] = {}
            for step in gmt_indices_sample:
                unprec = income_unprec[category].loc[{
                    'GMT':step,
                    'run':sims_per_step[step],
                    'birth_year':by
                }].sum(dim='country').mean(dim='run').item()
                pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                    'country':income_countries[category],
                    'birth_year':by,
                }].sum(dim='country').item()
        

    for by,ax in zip(by_sample,[ax00,ax10,ax20]):
        for i,category in enumerate(list(income_countries.keys())):
            order = np.argsort(list(pi_ratios[by][category].values())) # indices that would sort pf across gmt steps
            order = order[::-1] # want decreasing pf order, so we reverse the indices
            ordered_gmts = [gmt_indices_sample[o] for o in order]
            for step in ordered_gmts:
                colors_list =['None']*4
                colors_list[i] = 'white'
                # first paint white where we want actual color (semi transparent, white makes it look good) per category with nothing in other categories
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }     
                )
                # then paint actual color and radius
                colors_list[i] = colors[step]
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }
                )                 
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        ax.set_title(
            '{} CF'.format(by),
            loc='center',
            fontweight='bold',
            fontsize=labelfontsize,
            color='gray',       
        )        
        l+=1             
        percents = ['25%','50%','75%','100%']
        for i,r in enumerate(np.arange(0.25,1.01,0.25)):
            if r < 1:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    }
                )      
            else:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    labels=list(ig_dict.values()),
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    },
                    textprops={
                        'color':'0.5'
                    }
                )        
            
            if by == 1990:
                ax.annotate(
                    percents[i],
                    xy=(0,r+0.05),
                    color='gray',
                    ha='center',
                    fontsize=6
                )
            
    for i,k in enumerate(list(ig_dict.keys())):
        ax1.annotate(
            '{}: {}'.format(ig_dict[k],k),
            # xy=(0,-1.1-i*0.1),
            xy=(0,-2.15-i*0.1),
            color='gray',
            # ha='center',
            xycoords=ax0.transAxes
        )
    # --------------------------------------------------------------------
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways

    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]
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
            fontsize=labelfontsize,
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
        '$\mathregular{CF_{Heatwaves}}$ for 2020 \n birth cohort [%]',
        fontsize=labelfontsize,
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

    f.savefig('./ms_figures/combined_plot_popsizes_piecharts.png',dpi=1000)
                        
#%% ----------------------------------------------------------------
# plot ar6 hexagons with landfrac per extreme and multi extreme panels
    
def plot_hexagon_landfrac_union(
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

    x=11
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

    # colorbar stuff for landfrac emerging

    cmap_landfrac = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors_landfrac = [cmap_landfrac(i) for i in levels[:-1]]
    cmap_list_landfrac = mpl.colors.ListedColormap(colors_landfrac,N=len(colors_landfrac))
    ticks_landfrac = np.arange(0,101,10)
    norm_landfrac = mpl.colors.BoundaryNorm(levels*100,cmap_list_landfrac.N)   

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
    gs0 = gridspec.GridSpec(8,2)
    gs0.update(wspace=0.25,hspace=0.2)
    ax0 = f.add_subplot(gs0[5:8,0:1])

    # left side for 1960
    # masp per hazard
    gsn0 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,0:1],
        wspace=0,
        hspace=0,
    )
    ax00 = f.add_subplot(gsn0[0])
    ax10 = f.add_subplot(gsn0[1])
    ax20 = f.add_subplot(gsn0[2]) 

    ax01 = f.add_subplot(gsn0[3])
    ax11 = f.add_subplot(gsn0[4])
    ax21 = f.add_subplot(gsn0[5])       

    # colorbars
    pos0 = ax0.get_position()

    # colorbar for landfrac of emergence
    cax_landfrac = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.0575,
        pos0.width*1.5,
        pos0.height*0.15
    ])
    # colorbar for union
    cax_union = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.2,
        pos0.width*1.5,
        pos0.height*0.15
    ])

    # right side for 2020
    ax1 = f.add_subplot(gs0[5:8,1:2]) # map of emergence union
    gsn1 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,1:2],
        wspace=0,
        hspace=0,
    )
    ax02 = f.add_subplot(gsn1[0])
    ax12 = f.add_subplot(gsn1[1])
    ax22 = f.add_subplot(gsn1[2]) 

    ax03 = f.add_subplot(gsn1[3])
    ax13 = f.add_subplot(gsn1[4])
    ax23 = f.add_subplot(gsn1[5])     

    # plot 1960
    i=0
    l=0

    ax00.annotate(
        '1960 birth cohort',
        (0.55,1.3),
        xycoords=ax00.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )          

    i+=1

    # new gdf for union
    gdf_ar6_hex_union_1960 = gdf_ar6_hex.copy()
    gdf_ar6_hex_union_1960['union'] = 0

    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        ds_global_emergence = d_global_emergence[extr]
        gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
        # gdf_hex_landrac_plottable = gdf_ar6_emerged_landfrac.merge(gdf_ar6_hex,left_index=True,right_index=True)
        gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
        gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==1960]
        gdf_hex_landrac_plottable_1960['landfrac'] = gdf_hex_landrac_plottable_1960['landfrac'] * 100 
        gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable_1960[['geometry','birth_year','region','names','landfrac']]
        gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'] + (gdf_hex_landrac_plottable_1960['landfrac'] > landfrac_threshold)
        gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'].astype(float)
        
        gdf_hex_landrac_plottable_1960.plot(
            column='landfrac',
            cmap=cmap_list_landfrac,
            norm=norm_landfrac,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray'
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1    
        i+=1
        
    gdf_ar6_hex_union_1960.plot(
        column='union',
        cmap=cmap_list_union,
        norm=norm_union,
        edgecolor='gray',
        ax=ax0,
    )
    ax0.set_axis_off()
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1                 
    i+=1

    # 2020 birth cohort
    ax02.annotate(
        '2020 birth cohort',
        (0.55,1.3),
        xycoords=ax02.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )       

    # new gdf for union
    gdf_ar6_hex_union_2020 = gdf_ar6_hex.copy()
    gdf_ar6_hex_union_2020['union'] = 0

    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ds_global_emergence = d_global_emergence[extr]
        gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
        gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
        gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==2020]
        gdf_hex_landrac_plottable_2020['landfrac'] = gdf_hex_landrac_plottable_2020['landfrac'] * 100     
        gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable_2020[['geometry','birth_year','region','names','landfrac']]
        gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'] + (gdf_hex_landrac_plottable_2020['landfrac'] > landfrac_threshold)
        gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'].astype(float)
        
        gdf_hex_landrac_plottable_2020.plot(
            column='landfrac',
            cmap=cmap_list_landfrac,
            norm=norm_landfrac,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray',
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1          
        i+=1  
        
    gdf_ar6_hex_union_2020.plot(
        column='union',
        cmap=cmap_list_union,
        norm=norm_union,
        edgecolor='gray',
        ax=ax1,
    )
    ax1.set_axis_off()
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1  

    # colorbar for median landfrac emergence per extreme
    cb_landfrac = mpl.colorbar.ColorbarBase(
        ax=cax_landfrac, 
        cmap=cmap_list_landfrac,
        norm=norm_landfrac,
        orientation='horizontal',
        extend='neither',
        spacing='uniform',
        ticks=ticks_landfrac,
        drawedges=False,
    )
    cb_landfrac.ax.set_title(
        '(' + r"$\bf{g}$" + ',' + r"$\bf{n}$" + ')',
        y=1.06,
        fontsize=10,
        loc='left',
    )
    cb_landfrac.set_label(
        'Median percent of land area emerging'.format(str(landfrac_threshold)),
        fontsize=10,
        labelpad=8,
        color='gray',
    )
    cb_landfrac.ax.xaxis.set_label_position('top')
    cb_landfrac.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   

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
    cb_u.ax.set_title(
        '(' + r"$\bf{g}$" + ',' + r"$\bf{n}$" + ')',
        y=1.06,
        fontsize=10,
        loc='left',
    )

    cb_u.set_label(
        'Number of extremes emerging in at least {}% of region'.format(str(landfrac_threshold)),
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

    f.savefig('./ms_figures/emergence_landfrac_union_hexagons_{}.png'.format(landfrac_threshold),dpi=1000,bbox_inches='tight')

# %%

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

    # f.savefig('./ms_figures/emergence_union_hexagons_multithresh.png',dpi=1000,bbox_inches='tight')
# %%
