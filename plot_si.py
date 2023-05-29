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
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()


# %% ---------------------------------------------------------------
# population fractions tseries for all hazards for 2020 birth cohort

def plot_pf_gmt_tseries_allhazards(
    df_GMT_strj,
    da_gs_popdenom,
):

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

    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)        

    by=2020
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        
        da_plt = ds_pf_gs_extr['unprec'].loc[{
            'birth_year':by,
            'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
        }].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
        da_plt_gmt = da_plt.where(da_plt!=0) / da_gs_popdenom.loc[{'birth_year':by}].sum(dim='country') * 100 
        
        list_extrs_pf.append(da_plt_gmt)
        
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
    l=0
    for ax,extr in zip(axes.flatten(),extremes):
        da_plt= ds_pf_gs_extrs.loc[{'hazard':extr}]
        ax.plot(
            np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
            da_plt.mean(dim='run').values,
            linestyle='-',
            color='darkred'
        )    
        ax.fill_between(
            np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
            y1=da_plt.max(dim='run').values,
            y2=da_plt.min(dim='run').values,
            color='peachpuff',
        )     
        ax.set_ylabel(
            None, 
        )         
        ax.set_xlabel(
            'GMT anomaly at 2100 [°C]', 
            va='center', 
            labelpad=10,
            fontsize=12,
            color='gray'
        )                                           
        ax.set_xticks(
            ticks=GMT_indices_ticks,
            labels=gmts2100,
        )    
        ax.set_xlim(
            GMT_indices_ticks[0]-0.5,
            GMT_indices_ticks[-1]+0.5,
        )
        ax.set_title(
            extremes_labels[extr],
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
        if l <= 2:
            ax.tick_params(labelbottom=False)    
            ax.set_xlabel(
                None, 
            )   
        if (l == 0) or (l == 3):
            ax.set_ylabel(
                'Population %', 
                va='center', 
                rotation='vertical',
                labelpad=10,
                fontsize=12,
                color='gray',
            )     
        if l == 4:
            handles = [
                Line2D([0],[0],linestyle='-',color='darkred'),
                Rectangle((0,0),1,1,color='peachpuff'),
            ]    

            labels= [
                'Mean',
                'Range',     
            ]    
            x0 = 0.55 # bbox for legend
            y0 = 0.25
            xlen = 0.2
            ylen = 0.2    
            legend_font = 12        
            ax.legend(
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
                    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')         
        ax.tick_params(colors='gray')      
        l+=1  
        
        f.savefig('./si_figures/pf_2020_tseries_allhazards.png',dpi=1000)

# %% ---------------------------------------------------------------
# population fractions tseries for all hazards for 2020 birth cohort

def plot_pf_by_tseries_allhazards(
    flags,
    df_GMT_strj,
    da_gs_popdenom,
):

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

    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)        

    # GMT step representing CAT policy pledges for 2.7 degree warming
    gmtlevel=17
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)               
        
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)         
        
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        
        da_plt = ds_pf_gs_extr['unprec'].loc[{
            'birth_year':birth_years,
            'GMT':gmtlevel,
            'run':sims_per_step[gmtlevel]
        }].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
        da_plt_gmt = da_plt.where(da_plt!=0) / da_gs_popdenom.sum(dim='country') * 100 
        
        list_extrs_pf.append(da_plt_gmt)
        
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
    l=0
    for ax,extr in zip(axes.flatten(),extremes):
        da_plt= ds_pf_gs_extrs.loc[{'hazard':extr}]
        ax.plot(
            birth_years,
            da_plt.mean(dim='run').values,
            linestyle='-',
            color='darkred'
        )    
        ax.fill_between(
            birth_years,
            y1=da_plt.max(dim='run').values,
            y2=da_plt.min(dim='run').values,
            color='peachpuff',
        )     
        ax.set_ylabel(
            None, 
        )         
        ax.set_xlabel(
            'Birth year', 
            va='center', 
            labelpad=10,
            fontsize=12,
            color='gray'
        )                                           
        ax.set_xticks(
            ticks=np.arange(birth_years[0],birth_years[-1]+1,20),
        )    
        ax.set_xlim(
            1960-0.5,
            2020+0.5,
        )
        ax.set_title(
            extremes_labels[extr],
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
        if l <= 2:
            ax.tick_params(labelbottom=False)    
            ax.set_xlabel(
                None, 
            )   
        if (l == 0) or (l == 3):
            ax.set_ylabel(
                'Population %', 
                va='center', 
                rotation='vertical',
                labelpad=10,
                fontsize=12,
                color='gray',
            )     
        if l == 4:
            handles = [
                Line2D([0],[0],linestyle='-',color='darkred'),
                Rectangle((0,0),1,1,color='peachpuff'),
            ]    

            labels= [
                'Mean',
                'Range',     
            ]    
            x0 = 0.55 # bbox for legend
            y0 = 0.25
            xlen = 0.2
            ylen = 0.2    
            legend_font = 12        
            ax.legend(
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
                    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')         
        ax.tick_params(colors='gray')      
        l+=1  
        
        f.savefig('./si_figures/pf_2.7_tseries_allhazards.png',dpi=1000)

# %% ---------------------------------------------------------------
# population fractions box plot tseries for all hazards

def plot_boxplots_allhazards(
    da_gs_popdenom,
    df_GMT_strj,
    flags,
):

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

    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)        
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))

    # get data
    df_list_gs = []
    for extr in extremes:
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
    x=14
    y=7
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    l = 0
    for ax,extr in zip(axes.flatten(),extremes):
        
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
            ax=ax,
        )
        p.legend_.remove()                  
        ax.set_ylabel(
            None, 
        )         
        ax.set_xlabel(
            'Birth year', 
            va='center', 
            labelpad=10,
            fontsize=12,
            color='gray'
        )                                            
        ax.set_title(
            extremes_labels[extr],
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
        if l <= 2:
            ax.tick_params(labelbottom=False)    
            ax.set_xlabel(
                None, 
            )   
        if (l == 0) or (l == 3):
            ax.set_ylabel(
                'Population %', 
                va='center', 
                rotation='vertical',
                labelpad=10,
                fontsize=12,
                color='gray',
            )     
        if l == 0:
            # bbox
            x0 = 0.065
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

            ax.legend(
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
                    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')         
        ax.tick_params(colors='gray')      
        l+=1
          
        f.savefig('./si_figures/pf_boxplots_allhazards.png',dpi=1000)

# %% ---------------------------------------------------------------
# population fractions maps for all hazards for 2020 birth cohort

def plot_pf_maps_allhazards(
    da_gs_popdenom,
    gdf_country_borders,
):
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     
    x=16
    y=5
    markersize=10
    # cbar stuff
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   

    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',+
        'tropicalcyclonedarea': 'Tropical cyclones',
    }  

    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]

    # colorbar stuff ------------------------------------------------------------
    cmap_whole = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    norm = mpl.colors.BoundaryNorm(levels*100,cmap_list_frac.N)   

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    by=2020
    gmt_indices_152535 = [24,15,6]

    # since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
    # so we only take sims or runs valid per GMT level and make sure nans are 0
    df_list_gs = []
    for extr in extremes:
        with open('./data/pickles/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)         
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)  
        da_p_gs_plot = ds_pf_gs['unprec'].loc[{
            'GMT':gmt_indices_152535,
            'birth_year':by,
        }]          
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)        
        for step in gmt_indices_152535:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
            da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
            df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
            df_p_gs_plot_step['extreme'] = extr
            df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    f,axes = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(x,y),
        subplot_kw={'projection': ccrs.Robinson()}
    )    

    pos00 = axes[0,-1].get_position()
    cax = f.add_axes([
        pos00.x0+0.15,
        pos00.y0-0.51,
        0.025,
        pos00.height*3.2
    ])

    # gmt_indices_123 = [19,10,0]
    by=2020
    gmt_indices_152535 = [24,15,6]

    l=0
    for i,step in enumerate(gmt_indices_152535):
        for ax,extr in zip(axes[i],extremes):
            gdf_p['pf']=df_p_gs_plot['pf'][(df_p_gs_plot['GMT']==step)&(df_p_gs_plot['extreme']==extr)].values
            ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
            gdf_p.to_crs(robinson).plot(
                ax=ax,
                column='pf',
                cmap=cmap_list_frac,
                norm=norm,
                cax=cax,
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
                letters[l],
                loc='left',
                fontweight='bold',
                fontsize=10
            )    
            
            if l == 0 or l == 6 or l ==12:
                
                ax.annotate(
                    '{} °C'.format(gmt_legend[step]),
                    xy=(-0.2,0.2),
                    xycoords=ax.transAxes,
                    fontsize=12,
                    rotation='vertical',
                    color='gray',
                    fontweight='bold',     
                )  
                
            if l < 6:
                
                ax.set_title(
                    extremes_labels[extr],
                    loc='center',
                    color='gray',
                    fontweight='bold',
                    fontsize=12                
                )
            
            l+=1    
            
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax, 
        cmap=cmap_list_frac,
        norm=norm,
        orientation='vertical',
        spacing='uniform',
        ticks=ticks,
        drawedges=False,
    )

    cb.set_label(
        '$\mathregular{PF_{HW}}$ for 2020 birth cohort',
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
    cax.yaxis.set_label_position('right')

    f.savefig('./si_figures/maps_pf_allhazards.png',dpi=1000,bbox_inches='tight')    
    
#%% ----------------------------------------------------------------
# plot of locations of emergence

def plot_emergence_fracs(
    grid_area,
    da_emergence_mean,
):
    x=12
    y=6
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
    density=6
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

    # colorbar stuff ------------------------------------------------------------
    cmap_whole = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.0009,0.1)
    levels = np.insert(levels,1,0.001)
    levels[0] = -0.1
    colors = [cmap_whole(i) for i in levels[2:]]
    colors.insert(0,'gray')
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,1.01,.10)
    ticks[0] = -0.05
    tick_labels = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    norm = mpl.colors.BoundaryNorm(levels,cmap_list_frac.N)   


    # # colorbar args
    # values = [-5,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,5]
    # tick_locs = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(3,2)
    gs0.update(wspace=0.25)

    # left side for 1960
    # masp per hazard
    gsn0 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:3,0:1],
        wspace=0,
        hspace=0,
    )
    ax00 = f.add_subplot(gsn0[0],projection=ccrs.Robinson())
    ax10 = f.add_subplot(gsn0[1],projection=ccrs.Robinson())
    ax20 = f.add_subplot(gsn0[2],projection=ccrs.Robinson()) 

    ax01 = f.add_subplot(gsn0[3],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gsn0[4],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gsn0[5],projection=ccrs.Robinson())       

    # colobar
    pos0 = ax21.get_position()
    cax = f.add_axes([
        pos0.x0,
        pos0.y0-0.1,
        pos0.width*3.05,
        pos0.height*0.2
    ])

    # right side for 2020
    gsn1 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:3,1:2],
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
    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p1960 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':17,
            'birth_year':1960,
        }]
        p1960 = xr.where( # applying correction since 1st are missing in plot for some reason
            p1960 == 1,
            0.99,
            p1960
        )
        # p = p.where(mask.notnull())
        p1960.plot(
            ax=ax,
            cmap=cmap_list_frac,
            levels=levels,
            add_colorbar=False,
            add_labels=False,
            transform=ccrs.PlateCarree(),
            zorder=5
        )    
        ax.contourf(
            p1960.lon.data,
            p1960.lat.data,
            p1960.where(p1960>0.25).notnull(),
            levels=[.5,1.5],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[density*'/',density*'/'],
            zorder=10,
        )        
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
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
    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p2020 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':17,
            'birth_year':2020,
        }]
        p2020 = xr.where( # applying correction since 1st are missing in plot for some reason
            p2020 == 1,
            0.99,
            p2020
        )    
        # p = p.where(mask.notnull())
        p2020.plot(
            ax=ax,
            cmap=cmap_list_frac,
            levels=levels,
            add_colorbar=False,
            add_labels=False,
            transform=ccrs.PlateCarree(),
            zorder=5
        )    
        ax.contourf(
            p2020.lon.data,
            p2020.lat.data,
            p2020.where(p2020>0.25).notnull(),
            levels=[.5,1.5],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[density*'/',density*'/'],
            zorder=10,
        )                
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
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
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax, 
        cmap=cmap_list_frac,
        norm=norm,
        orientation='horizontal',
        spacing='uniform',
        ticks=ticks,
        drawedges=False,
    )

    cb.set_label(
        'Fraction of projections with emergence',
        fontsize=14,
        labelpad=10,
        color='gray',
    )
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.ax.set_xticklabels(tick_labels)

    # lat = grid_area.lat.values
    # lon = grid_area.lon.values
    # mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    # eu=da_emergence_union.loc[{'GMT':17,'birth_year':2020}].where(mask.notnull())
    # la_frac_eu_gteq3 = xr.where(eu>=3,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon'))
    # print(la_frac_eu_gteq3)    
        
    f.savefig('./si_figures/emergence_fracs.png',dpi=1000,bbox_inches='tight')    
    
#%% ----------------------------------------------------------------
# plot of locations of exposure for showing geograpical constraints

def plot_exposure_locations(
    grid_area,
):
    x=8
    y=6
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
    density=6
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

    # colorbar stuff ------------------------------------------------------------
    colors=[
        mpl.colors.to_rgb('steelblue'),
        mpl.colors.to_rgb('darkgoldenrod'),
        mpl.colors.to_rgb('peru'),
    ]
    cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
    levels = np.arange(0.5,3.6,1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(3,2)
    gs0.update(wspace=0.25)

    # maps per hazard
    ax00 = f.add_subplot(gs0[0],projection=ccrs.Robinson())
    ax10 = f.add_subplot(gs0[1],projection=ccrs.Robinson())
    ax20 = f.add_subplot(gs0[2],projection=ccrs.Robinson()) 

    ax01 = f.add_subplot(gs0[3],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs0[4],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs0[5],projection=ccrs.Robinson())       

    # plot 1960
    i=0
    l=0     

    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        # first get all regions that have exposure to extr in ensemble
        with open('./data/pickles/{}/exposure_occurrence_{}.pkl'.format(extr,extr), 'rb') as file:
            da_exposure_occurrence = pk.load(file)   
            
        da_exposure_occurrence = da_exposure_occurrence.where(da_exposure_occurrence).where(mask.notnull())*3 
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        da_exposure_occurrence.plot(
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

        
    f.savefig('./si_figures/exposure_locations.png',dpi=1000,bbox_inches='tight')    

