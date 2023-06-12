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
from scipy import stats as sts
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
    x=8
    y=10
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
        'cropfailedarea': 'Crop \nfailures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical \ncyclones',
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
    gmt_indices_152535 = [6,15,24]
    
    fig,axes = plt.subplots(
        nrows=6,
        ncols=3,
        figsize=(x,y),
        subplot_kw={'projection': ccrs.Robinson()}
    )    

    pos00 = axes[-1,0].get_position()
    cax = fig.add_axes([
        pos00.x0,
        0.05,
        0.685,
        pos00.height*0.2
    ])    

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

    l=0
    for i,extr in enumerate(extremes):
        for ax,step in zip(axes[i],gmt_indices_152535):
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
            
            if np.any(np.isin([0,3,6,9,12,15],l)):
                
                ax.annotate(
                    extremes_labels[extr],
                    xy=(-0.2,0.5),
                    xycoords=ax.transAxes,
                    fontsize=10,
                    rotation='vertical',
                    va='center',
                    color='gray',
                    fontweight='bold',     
                )  
                
            if l < 3:
                
                ax.set_title(
                    '{} °C'.format(gmt_legend[step]),
                    loc='center',
                    color='gray',
                    fontweight='bold',
                    fontsize=10                
                )
            
            l+=1    
            
        
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

    fig.savefig('./si_figures/maps_pf_allhazards_vertical.png',dpi=1000,bbox_inches='tight')    
    
# %% ---------------------------------------------------------------
# population fractions box plot tseries for all hazards when computed with geoconstraints

def plot_geoconstrained_boxplots(
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

    gmt_indices_sample = [6,15,24]
    gmt_legend={
        gmt_indices_sample[0]:'1.5',
        gmt_indices_sample[1]:'2.5',
        gmt_indices_sample[2]:'3.5',
    }
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    df_list = []

    # for extr,ax in zip(extremes,axes.flatten()):   
    for extr in extremes:
                            
        with open('./data/pickles/{}/pf_geoconstrained_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_geoconstrained = pk.load(f)      
            
        # get metadata for extreme
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)
        
        # maybe not necessary since means are ignoring nans for runs not included in some steps
        sims_per_step = {}
        for step in gmt_indices_sample:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)          
            
        da_pf_gc = ds_pf_geoconstrained['pf_perrun'].loc[{
            'GMT':gmt_indices_sample,
            'birth_year': sample_birth_years,
        }] * 100
        
        for step in gmt_indices_sample:
            da_pf_gs_plot_step = da_pf_gc.loc[{'run':sims_per_step[step],'GMT':step}]
            df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
            df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
            df_pf_gs_plot_step['hazard'] = extr
            df_list.append(df_pf_gs_plot_step)    
        
        df_pf_gs_plot = pd.concat(df_list)
        
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
        
    f.savefig('./si_figures/pf_geoconstrained_boxplots_allhazards.png',dpi=1000)
    
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
    countries_mask,
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
        da_exposure_occurrence = da_exposure_occurrence.where(countries_mask.notnull())
        
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
    
#%% ----------------------------------------------------------------
# plot gmt pathways in rcps and ar6

def plot_gmt_pathways(
    df_GMT_strj,
    d_isimip_meta,
):

    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=5
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 10
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_hi = 'darkred'       # mean color for GMT trajectories above 2.5 at 2100
    col_med = 'darkgoldenrod'   # mean color for GMT trajectories above 1.5 to 2.5 at 2100     
    col_low = 'steelblue'       # mean color for GMT trajectories from min to 1.5 at 2100
    gmt_indices_sample = [6,15,17,24]
    colors_rcp = {
        'rcp26': col_low,
        'rcp60': col_med,
        'rcp85': col_hi,
    }
    colors = {
        '3.5':'firebrick',
        'CP':'darkorange',
        '2.5':'yellow',
        '1.5':'steelblue',
    }    
    legend_lw=3.5 # legend line width
    x0 = 0.15 # bbox for legend
    y0 = 0.5
    xlen = 0.2
    ylen = 0.2    
    legend_entrypad = 0.5 # space between entries
    legend_entrylen = 0.75 # length per entry
    col_bis = 'black'     # color bisector
    style_bis = '--'     # style bisector
    lw_bis = 1     # lineweight bisector
    time = year_range
    # xmin = np.min(time)
    # xmax = np.max(time)
    xmin = 1960
    xmax = 2100

    ymin=0
    ymax=4

    axar6_ylab = 'GMT [°C]'
    axar6_xlab = 'Time'

    gcms = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']
    rcps = ['rcp26','rcp60','rcp85']
    GMTs = {}
    for gcm in gcms:
        GMTs[gcm] = {}
        for rcp in rcps:
            i=0
            while i < 1:
                for k,v in list(d_isimip_meta.items()):
                    if v['gcm'] == gcm and v['rcp'] == rcp:
                        GMTs[gcm][rcp] = v['GMT']
                        i+=1
                    if i == 1:
                        break
            
                

    f,(axrcp,axar6) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot GMTs

    # plot all new scenarios in grey, then overlay marker scens
    df_GMT_strj.loc[:,6:24].plot(
        ax=axar6,
        color='grey',
        zorder=1,
        lw=lw_mean,
    )

    # plot smooth gmts from RCPs
    for gcm in gcms:
        for rcp in rcps:  
            GMTs[gcm][rcp].plot(
                ax=axrcp,
                color=colors_rcp[rcp],
                zorder=2,
                lw=lw_mean,
                style='-'
            )  
            
    # plot new ar6 marker scenarios in color

    df_GMT_15 = df_GMT_strj.loc[:,gmt_indices_sample[0]]
    df_GMT_15.plot(
        ax=axar6,
        color=colors['1.5'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_25 = df_GMT_strj.loc[:,gmt_indices_sample[1]]
    df_GMT_25.plot(
        ax=axar6,
        color=colors['2.5'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_NDC = df_GMT_strj.loc[:,gmt_indices_sample[2]]
    df_GMT_NDC.plot(
        ax=axar6,
        color=colors['CP'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_35 = df_GMT_strj.loc[:,gmt_indices_sample[3]]
    df_GMT_35.plot(
        ax=axar6,
        color=colors['3.5'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_35.loc[1960:2009].plot(
        ax=axar6,
        color='grey',
        zorder=3,
        lw=2,
    )                


    axrcp.set_ylabel(
        axar6_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )

    axar6.set_ylabel(
        None, 
    )

    axrcp.set_xlabel(
        axar6_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    

    axar6.set_xlabel(
        axar6_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    


    for i,ax in enumerate([axrcp,axar6]):
        ax.set_title(letters[i],loc='left',fontweight='bold')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_xticks(
            ticks=np.arange(1960,2101,20),
            labels=[None,1980,None,2020,None,2060,None,2100],
        )    
        ax.set_axisbelow(True) 

    handles_ar6 = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['1.5']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['2.5']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['CP']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['3.5']),
    ]    

    handles_rcp = [
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=colors_rcp['rcp85']),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=colors_rcp['rcp60']),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=colors_rcp['rcp26'])         
    ]

    labels_ar6= [
        '1.5 °C',
        '2.5 °C',
        'Current pledges',
        '3.5 °C',
    ]
    labels_rcp = [
        'RCP 8.5',
        'RCP 6.0',
        'RCP 2.6',        
    ]    
        
    axar6.legend(
        handles_ar6, 
        labels_ar6, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        loc=3,
        ncol=1,
        fontsize=legend_font, 
        mode="expand", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )  
    axrcp.legend(
        handles_rcp, 
        labels_rcp, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        loc=3,
        ncol=1,
        fontsize=legend_font, 
        mode="expand", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )               
            
    f.savefig('./si_figures/GMT_trajectories.png',bbox_inches='tight',dpi=1000)    
    
#%% ----------------------------------------------------------------
# plot heatmaps for countrylevel emergence

def plot_heatmaps_allhazards_countryemergence(
    df_GMT_strj,
):
    
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': '$\mathregular{PF_{Wildfires}}$',
        'cropfailedarea': '$\mathregular{PF_{Crop failures}}$',
        'driedarea': '$\mathregular{PF_{Droughts}}$',
        'floodedarea': '$\mathregular{PF_{Floods}}$',
        'heatwavedarea': '$\mathregular{PF_{Heatwaves}}$',
        'tropicalcyclonedarea': '$\mathregular{PF_{Tropical cyclones}}$',
    }        

    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    

    # using fracs already computed
    # # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            da_pf = pk.load(f)['mean_frac_unprec_all_b_y0'].loc[{
                'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
            }] *100    
        with open('./data/pickles/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)        
        list_extrs_pf.append(da_pf)
        
    da_pf_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})

    # compute fracs to account for runs per step
    # loop through extremes and concat pop and pop frac
    # list_extrs_p = []
    # gmts = np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')

    # for extr in extremes:
    #     with open('./data/pickles/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
    #         d_isimip_meta = pk.load(file)   
            
    #     with open('./data/pickles/{}/pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
    #         da_p = pk.load(f)['unprec_all_b_y0'].loc[{
    #             'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
    #         }] *1000            
                
    #     sims_per_step = {}
    #     for step in gmts:
    #         sims_per_step[step] = []
    #         # print('step {}'.format(step))
    #         for i in list(d_isimip_meta.keys()):
    #             if d_isimip_meta[i]['GMT_strj_valid'][step]:
    #                 sims_per_step[step].append(i)
    #     steps=[]
    #     for step in gmts:
    #         da_p_step = da_p.loc[{
    #             'run':sims_per_step[step],
    #             'GMT':step,
    #         }].mean(dim='run')
    #         steps.append(da_p_step)
    #     da_p_extr = xr.concat(steps,dim='GMT').assign_coords({'GMT':gmts})
            
    #     list_extrs_p.append(da_p_extr)
        
    # da_pf_extrs = xr.concat(list_extrs_p,dim='hazard').assign_coords({'hazard':extremes}) / ds_cohorts['by_population_y0'].sum(dim='country') * 1000

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
        p = da_pf_extrs.loc[{
            'hazard':extr,
            'birth_year':np.arange(1960,2021),
        }].plot(
            x='birth_year',
            y='GMT',
            ax=ax,
            add_labels=False,
            levels=10,
            cmap='Reds',
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

    f.savefig('./si_figures/pf_heatmap_combined_allsims_countryemergence.png',dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/pf_heatmap_combined_allsims.eps',format='eps',bbox_inches='tight')
    plt.show()         

#%% ----------------------------------------------------------------
# testing for pie charts plot
# ------------------------------------------------------------------    
def plot_allhazards_piecharts(
    da_gs_popdenom,
    df_countries,
):

    x=6
    y=10
    fig,axes = plt.subplots(
        nrows=6,
        ncols=3,
        figsize=(x,y),
    )

    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop \nfailures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical \ncyclones',
    }  

    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]    

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

    l=0
    for e,extr in enumerate(extremes):
        
        with open('./data/pickles/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)         
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)  

        ax00,ax10,ax20 = axes[e]

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
                    if extr == 'heatwavedarea':
                        pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                            'country':income_countries[category],
                            'birth_year':by,
                        }].sum(dim='country').item()
                    elif extr == 'cropfailedarea':
                        pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                            'country':income_countries[category],
                            'birth_year':by,
                        }].sum(dim='country').item() / 0.4                    
                    else:
                        pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                            'country':income_countries[category],
                            'birth_year':by,
                        }].sum(dim='country').item() / 0.2            

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
            
            if l < 3: 
                
                ax.set_title(
                    by,
                    loc='center',
                    fontweight='bold',
                    fontsize=12,
                    color='gray',       
                )   
                
            if np.any(np.isin([0,3,6,9,12,15],l)):
                
                ax.annotate(
                    extremes_labels[extr],
                    xy=(-0.4,0.5),
                    xycoords=ax.transAxes,
                    fontsize=10,
                    rotation='vertical',
                    va='center',
                    color='gray',
                    fontweight='bold',     
                )            
            # l+=1             
            
            if extr == 'heatwavedarea':
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
            elif extr == 'cropfailedarea':
                percents = ['10%','20%','30%','40%']
                # for i,r in enumerate([0.3333,0.6666,1]):
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
            else:
                percents = ['5%','10%','15%','20%']
                # for i,r in enumerate([0.3333,0.6666,1]):
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
                    
            l += 1    
                        
        if extr == extremes[-1]:
            for i,k in enumerate(list(ig_dict.keys())):
                ax00.annotate(
                    '{}: {}'.format(ig_dict[k],k),
                    xy=(0,-0.25-i*0.125),
                    color='gray',
                    # ha='center',
                    xycoords=ax00.transAxes
                )

    fig.savefig('./si_figures/all_hazards_piecharts.png',dpi=1000,bbox_inches='tight')


# %%
