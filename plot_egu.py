# ---------------------------------------------------------------
# Functions for emergence analysis at EGU
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

# %% ----------------------------------------------------------------
            
def combined_plot_egu(
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
    # tick_font = 12
    # cbar stuff
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

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
        pos00.width * 2.25,
        pos00.height*0.1
    ])
    
    # removing axis for plot popping
    # ax10.axis('off')
    # ax01.axis('off')
    # ax11.axis('off')
    # ax21.axis('off')
    
    i = 0 # letter indexing
    
    # colorbar stuff ------------------------------------------------------------
    
    cmap_whole = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    # ticks = np.arange(0,1.01,0.1)
    ticks = np.arange(0,101,10)
    norm = mpl.colors.BoundaryNorm(levels,cmap_list_frac.N)   


    # pop frac heatmap ----------------------------------------------------------
    # gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    

    # levels = np.arange(0,1.01,0.05)
    levels = np.arange(0,101,5)
      
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)
    p2 = ds_pf_gs['unprec'].loc[{
        'birth_year':np.arange(1960,2021),
        'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
    }].sum(dim='country')
    p2 = p2.where(p2!=0).mean(dim='run') /  da_gs_popdenom.sum(dim='country') * 100
    p2 = p2.plot(
        x='birth_year',
        y='GMT',
        ax=ax00,
        add_colorbar=False,
        levels=levels,
        norm=norm,
        cmap=cmap_list_frac,
    )
    p2.axes.set_yticks(
        ticks=GMT_indices_ticks,
        labels=gmts2100
    )
    p2.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    
        # '1.5 °C GMT warming by 2100',
        # '2.5 °C GMT warming by 2100',
        # '3.5 °C GMT warming by 2100',    
    
    p2.axes.set_ylabel(
        'GMT warming by 2100 [°C]',
        fontsize=12,
        color='gray',
    )
    p2.axes.set_xlabel(
        'Birth year',
        fontsize=12,
        color='gray',
    )
    p2.axes.spines['left'].set_color('gray')
    p2.axes.spines['bottom'].set_color('gray')        
    p2.axes.spines['top'].set_color('gray')
    p2.axes.spines['right'].set_color('gray')      
    p2.axes.tick_params(colors='gray')     
    
    # ax00.set_title(
    #     letters[i],
    #     loc='left',
    #     fontweight='bold',
    #     fontsize=10
    # )    
    i+=1

    # add rectangle to 2020 series
    ax00.add_patch(Rectangle(
        (2020-0.5,0-0.5),1,29,
        facecolor='none',
        ec='dimgray',
        lw=0.8
    ))
    
    
    # bracket connecting 2020 in heatmap to scatter plot panel ------------------
    
    # vertical line
    x_h=2020
    y_h=5.5
    x_s=0.995
    y_s=1.05    
    con = ConnectionPatch(
        xyA=(x_h,y_h),
        xyB=(x_s,y_s),
        coordsA=ax00.transData,
        coordsB=ax10.transAxes,
        color='dimgray',
    )
    ax00.add_artist(con)         
    
    # horizontal line
    x_s2=0.075
    y_s2=y_s
    con = ConnectionPatch(
        xyA=(x_s,y_s),
        xyB=(x_s2,y_s2),
        coordsA=ax10.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray'
    )
    ax00.add_artist(con)    
    
    # brace outliers
    # left 
    x_s3=x_s2-0.025
    y_s3=y_s2-0.05  
    con = ConnectionPatch(
        xyA=(x_s2,y_s2),
        xyB=(x_s3,y_s3),
        coordsA=ax10.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray'
    )
    ax10.add_artist(con)       
    
    # right
    x_s4=x_s+0.025
    y_s4=y_s-0.05    
    con = ConnectionPatch(
        xyA=(x_s,y_s),
        xyB=(x_s4,y_s4),
        coordsA=ax10.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray'
    )
    ax10.add_artist(con)

    # pop frac scatter ----------------------------------------------------------

    by=2020
    da_plt = ds_pf_gs['unprec'].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
    da_plt_gmt = da_plt.loc[{'birth_year':by}].where(da_plt.loc[{'birth_year':by}]!=0)
    da_plt_gmt = da_plt_gmt / da_gs_popdenom.loc[{'birth_year':by}].sum(dim='country') * 100
    # p = da_plt_gmt.to_dataframe(name='pf').reset_index(level="run")
    # x = p.index.values
    # y = p['pf'].values
    # ax10.scatter(
    #     x,
    #     y,
    #     s=markersize,
    #     c='steelblue'
    # )
    # ax10.plot(
    #     GMT_labels,
    #     da_plt_gmt.mean(dim='run').values,
    #     marker='_',
    #     markersize=markersize/2,
    #     linestyle='',
    #     color='r'
    # )
    ax10.plot(
        GMT_labels,
        da_plt_gmt.mean(dim='run').values,
        # marker='-',
        # markersize=markersize/2,
        linestyle='-',
        color='darkred'
    )    
    ax10.fill_between(
        da_plt_gmt.GMT.values,
        y1=da_plt_gmt.max(dim='run').values,
        y2=da_plt_gmt.min(dim='run').values,
        color='peachpuff',
    )
    ax10.set_ylabel(
        'Population %', 
        va='center', 
        rotation='vertical',
        labelpad=10,
        fontsize=12,
        color='gray',
    )          
    ax10.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
        fontsize=12,
        color='gray'
    )                                           
    ax10.set_xticks(
        ticks=GMT_indices_ticks,
        labels=gmts2100,
    )    
    ax10.set_xlim(
        GMT_indices_ticks[0]-0.5,
        GMT_indices_ticks[-1]+0.5,
    )
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)    
    ax10.spines['left'].set_color('gray')
    ax10.spines['bottom'].set_color('gray')         
    ax10.tick_params(colors='gray')         

    # handles = [
    #     Line2D([0],[0],linestyle='None',marker='o',color='steelblue'),
    #     Line2D([0],[0],marker='_',color='r'),
            
    # ]
    
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
    
    # ax10.set_title(
    #     letters[i],
    #     loc='left',
    #     fontweight='bold',
    #     fontsize=10
    # )    
    i+=1     

    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     

    # gmt_indices_123 = [19,10,0]
    gmt_indices_152535 = [24,15,6]
    map_letters = {24:'e',15:'d',6:'c'}
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

    # ax = ax21 
    # step = 6
    for ax,step in zip((ax01,ax11,ax21),gmt_indices_152535):
    # for ax,step in zip((ax11,ax21),gmt_indices_152535[1:]):
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
        # ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))        
        
        # ax.set_title(
        #     map_letters[step],
        #     loc='left',
        #     fontweight='bold',
        #     fontsize=10,
        # )
        i+=1
        
        ax.set_title(
            # '{} °C'.format(str(np.round(df_GMT_strj.loc[2100,step],1))),
            '{} °C'.format(gmt_legend[step]),
            loc='center',
            fontweight='bold',
            fontsize=12,
            color='gray',       
        )
        
        # pointers connecting 2020, GMT step pixel in heatmap to map panels ------------------
        
        # if step == gmt_indices_152535[-2]:
        x_h=2020
        y_h=step
        x_m=0
        y_m=0.5
        con = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m),
            coordsA=ax00.transData,
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
        'Population % living unprecedented exposure to heatwaves',
        fontsize=14,
        color='gray'
    )
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        # labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)   
    cax00.xaxis.set_label_position('top')                   

    f.savefig('./figures/combined_heatmap_scatter_mapsofpf_gs_{}_egu_35.png'.format(flags['extr']),dpi=1000,bbox_inches='tight')
    plt.show()            

# %% ----------------------------------------------------------------
            
def combined_plot_boxplot(
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
    
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(4,4)
    gs0.update(hspace=0.8,wspace=0.8)
    ax00 = f.add_subplot(gs0[0:2,0:2]) # heatmap
    ax10 = f.add_subplot(gs0[2:,0:2]) # scatterplot for 2020 by
    gs00 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[:4,2:],
        # top=0.8
    )
    ax01 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 
    pos00 = ax00.get_position()
    cax00 = f.add_axes([
        # pos00.x0,
        pos00.x0+0.415,
        pos00.y0+0.4,
        # pos00.width * 2.25,
        pos00.width * 1.125,
        pos00.height*0.1
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
    # gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)
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
    # ax00.set_ylabel('Population % living unprecedented \n exposure to heatwaves',color='gray')
    ax00.set_ylabel('$\mathregular{PF_{HW}}$',color='gray',fontsize=14)
    # '$\mathregular{PF_{HW}}$',
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
    
    # bracket connecting 2020 in box plot to tseries plot panel ------------------
    
    # vertical line
    x_h=0.95
    y_h=-.075
    x_s=0.995
    y_s=1.05    
    con = ConnectionPatch(
        xyA=(x_h,y_h),
        xyB=(x_s,y_s),
        coordsA=ax00.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray',
    )
    ax00.add_artist(con)         
    
    # horizontal line
    x_s2=0.075
    y_s2=y_s
    con = ConnectionPatch(
        xyA=(x_s,y_s),
        xyB=(x_s2,y_s2),
        coordsA=ax10.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray'
    )
    ax00.add_artist(con)    
    
    # brace outliers
    # left 
    x_s3=x_s2-0.025
    y_s3=y_s2-0.05  
    con = ConnectionPatch(
        xyA=(x_s2,y_s2),
        xyB=(x_s3,y_s3),
        coordsA=ax10.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray'
    )
    ax10.add_artist(con)       
    
    # right
    x_s4=x_s+0.025
    y_s4=y_s-0.05    
    con = ConnectionPatch(
        xyA=(x_s,y_s),
        xyB=(x_s4,y_s4),
        coordsA=ax10.transAxes,
        coordsB=ax10.transAxes,
        color='dimgray'
    )
    ax10.add_artist(con)

    # pop frac t series for 2020 ----------------------------------------------------------
    
    by=2020
    da_plt = ds_pf_gs['unprec'].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
    da_plt_gmt = da_plt.loc[{'birth_year':by}].where(da_plt.loc[{'birth_year':by}]!=0)
    da_plt_gmt = da_plt_gmt / da_gs_popdenom.loc[{'birth_year':by}].sum(dim='country') * 100
    ax10.plot(
        GMT_labels,
        da_plt_gmt.mean(dim='run').values,
        linestyle='-',
        color='darkred'
    )    
    ax10.fill_between(
        da_plt_gmt.GMT.values,
        y1=da_plt_gmt.max(dim='run').values,
        y2=da_plt_gmt.min(dim='run').values,
        color='peachpuff',
    )
    ax10.set_ylabel(
        '$\mathregular{PF_{HW}}$ for 2020 birth cohort', 
        va='center', 
        rotation='vertical',
        labelpad=10,
        fontsize=14,
        color='gray',
    )         
    # ax00.set_ylabel('$\mathregular{PF_{HW}}$',color='gray',fontsize=14)
     
    ax10.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
        fontsize=14,
        color='gray'
    )                                           
    ax10.set_xticks(
        ticks=GMT_indices_ticks,
        labels=gmts2100,
    )    
    ax10.set_xlim(
        GMT_indices_ticks[0]-0.5,
        GMT_indices_ticks[-1]+0.5,
    )
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)    
    ax10.spines['left'].set_color('gray')
    ax10.spines['bottom'].set_color('gray')         
    ax10.tick_params(colors='gray')         
    
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
    # legend_font = 12        
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
    
    ax10.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    l+=1     

    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     

    # gmt_indices_123 = [19,10,0]
    gmt_indices_152535 = [24,15,6]
    map_letters = {24:'e',15:'d',6:'c'}
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
            letters[l],
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
    cax00.xaxis.set_label_position('top')                   

    f.savefig('./figures/combined_boxplots_scatter_maps_{}.png'.format(flags['extr']),dpi=1000,bbox_inches='tight')
    # f.savefig('./figures/combined_boxplots_scatter_maps_{}.eps'.format(flags['extr']),format='eps',bbox_inches='tight')
    
    plt.show()            



#%% ----------------------------------------------------------------
def emergence_union_plot_egu(
    grid_area,
    da_emergence_union,
    da_emergence_mean,
):
    # x=14
    # y=9
    x=12.6
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
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'WF',
        'cropfailedarea': 'CF',
        'driedarea': 'DR',
        'floodedarea': 'FL',
        'heatwavedarea': 'HW',
        'tropicalcyclonedarea': 'TC',
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
        cmap_reds(0.15),
        cmap_reds(0.3),
        cmap_reds(0.45),
        cmap_reds(0.6),
        cmap_reds(0.75),
        cmap_reds(0.9),
    ]
    cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
    levels = np.arange(0.5,3.6,1)
    union_levels = np.arange(-0.5,6.6,1)
    norm=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(4,2)
    gs0.update(wspace=0.25)
    ax0 = f.add_subplot(gs0[3:4,0:1],projection=ccrs.Robinson())

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
    
    # ax0.axis('off')
    
    # union map
    # ax0 = f.add_subplot(gs0[2:3,0:1],projection=ccrs.Robinson()) # map of emergence union

    pos0 = ax0.get_position()
    cax = f.add_axes([
        pos0.x0+0.19,
        pos0.y0+0.075,
        pos0.width*1.5,
        pos0.height*0.2
    ])
 
    # cax.axis('off')

    # right side for 2020
    ax1 = f.add_subplot(gs0[3:4,1:2],projection=ccrs.Robinson()) # map of emergence union
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
    
    # ax1.axis('off')   
    # ax02.axis('off')   
    # ax12.axis('off')   
    # ax22.axis('off')   
    # ax03.axis('off')   
    # ax13.axis('off')   
    # ax23.axis('off')   

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
        p3 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':17,
            'birth_year':1960,
        }]
        p3 = xr.where(p3>0,1,0)
        p3 = p3.where(p3).where(mask.notnull())*3
        p3.plot(
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
        
    p_u3 = da_emergence_union.loc[{'GMT':17,'birth_year':1960}].where(mask.notnull())
    
    ax0.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax0.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u3.plot(
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
    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p3 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':17,
            'birth_year':2020,
        }]
        p3 = xr.where(p3>0,1,0)
        p3 = p3.where(p3).where(mask.notnull())*3
        p3.plot(
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
        
    p_u3 = da_emergence_union.loc[{'GMT':17,'birth_year':2020}].where(mask.notnull())
    ax1.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax1.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u3.plot(
        ax=ax1,
        # cmap='Reds',
        cmap=cmap_list_union,
        levels=union_levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
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
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax, 
        cmap=cmap_list_union,
        norm=norm,
        orientation='horizontal',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )

    cb.set_label(
        'Number of \n emerged extremes',
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
        
    f.savefig('./figures/emergence_locations_egu_all_new.png',dpi=1000)

#%% ----------------------------------------------------------------
# plotting pop and pop frac for grid scale across hazards
def plot_heatmaps_allhazards_egu(
    df_GMT_strj,
    da_gs_popdenom,
):
    
    # --------------------------------------------------------------------
    # plotting utils
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
    # extremes_labels = {
    #     'burntarea': 'Wildfires',
    #     'cropfailedarea': 'Crop failure',
    #     'driedarea': 'Droughts',
    #     'floodedarea': 'Floods',
    #     'heatwavedarea': 'Heatwaves',
    #     'tropicalcyclonedarea': 'Tropical cyclones',
    # }   
    extremes_labels = {
        'burntarea': '$\mathregular{PF_{WF}}$',
        'cropfailedarea': '$\mathregular{PF_{CF}}$',
        'driedarea': '$\mathregular{PF_{DR}}$',
        'floodedarea': '$\mathregular{PF_{FL}}$',
        'heatwavedarea': '$\mathregular{PF_{HW}}$',
        'tropicalcyclonedarea': '$\mathregular{PF_{TC}}$',
    }        
    
    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    
    
    # --------------------------------------------------------------------
    # population fractions with simulation limits to avoid dry jumps
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)
        with open('./data/pickles/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
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
        p = ds_pf_gs_extrs.loc[{
            'hazard':extr,
            'birth_year':np.arange(1960,2021),
        }].plot.contourf(
            x='birth_year',
            y='GMT',
            ax=ax,
            add_labels=False,
            levels=10,
            cmap='Reds',
            # cbar_kwargs={'fontsize':'gray'}
        ) 
        
        ax.set_yticks(
            ticks=GMT_indices_ticks,
            labels=gmts2100,
            color='gray',
            # labelcolor='gray',
        )
        ax.set_xticks(
            ticks=np.arange(1960,2025,10),
            color='gray',
            # labelcolor='gray',
        )    
        # p.cbar.ax.tick_params(labelsize=12)
        # ax.set_ylabel('GMT anomaly at 2100 [°C]')
        # ax.set_xlabel('Birth year')
        
    # ax stuff
    l=0
    for n,ax in enumerate(axes.flatten()):
        # ax.set_title(
        #     letters[n],
        #     loc='left',
        #     fontweight='bold',
        # )
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
        # plt.rcParams["axes.grid"] = False
        # ax.grid(b=None)
        if not np.isin(n,[0,3]):
            ax.yaxis.set_ticklabels([])
        else:
            pass
            # ax.set_ylabel('GMT warming by 2100 [°C]',fontsize=12,color='gray')
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
    
    
    # ax.annotate(
    #     letters[l],
    #     # (1962,ax.get_ylim()[-1]+2),
    #     (1960,ax.get_ylim()[-1]+3),
    #     xycoords=ax.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='k',
    #     fontweight='bold',
    # )        
 
    f.savefig('./figures/pf_heatmap_combined_simlim.png',dpi=1000,bbox_inches='tight')
    f.savefig('./figures/pf_heatmap_combined_simlim.eps',format='eps',bbox_inches='tight')
    plt.show()     
    
    # --------------------------------------------------------------------
    # population fractions with all simulations
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
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
        p = ds_pf_gs_extrs.loc[{
            'hazard':extr,
            'birth_year':np.arange(1960,2021),
        }].plot.contourf(
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
    
    f.savefig('./figures/pf_heatmap_combined_allsims.png',dpi=1000,bbox_inches='tight')
    f.savefig('./figures/pf_heatmap_combined_allsims.eps',format='eps',bbox_inches='tight')
    plt.show()         


#%% ----------------------------------------------------------------
# plotting pop and pop frac for grid scale across hazards
def plot_p_pf_cs_heatmap_combined(
    df_GMT_strj,
):
    
    # --------------------------------------------------------------------
    # plotting utils
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
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failure',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }    
    
    # labels for GMT ticks
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    
    
    # --------------------------------------------------------------------
    # population totals
    
    # loop through extremes and concat pop
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_cs = pk.load(f)
        p = ds_pf_cs['mean_unprec_all_b_y0'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}] * 1000 / 10**6
        list_extrs_pf.append(p)
    da_pf_cs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})
    
    # plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        da_pf_cs_extrs.loc[{
            'hazard':extr,
            'birth_year':np.arange(1960,2021),
        }].plot(
            x='birth_year',
            y='GMT',
            ax=ax,
            add_labels=False,
            levels=10,
        ) 
        
        ax.set_yticks(
            ticks=GMT_indices_ticks,
            labels=gmts2100
        )
        ax.set_xticks(
            ticks=np.arange(1960,2025,10),
        )    
        
    # ax stuff
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            letters[n],
            loc='left',
            fontweight='bold',
        )
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
        )            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)                 
        if not np.isin(n,[0,3]):
            ax.yaxis.set_ticklabels([])
        else:
            ax.set_ylabel('GMT anomaly at 2100 [°C]')
        if n <= 2:
            ax.tick_params(labelbottom=False)    
        if n >= 3:
            ax.set_xlabel('Birth year')
                      
    f.savefig('./figures/cs_p_heatmap_combined.png',dpi=800)
    plt.show()    
    
    # --------------------------------------------------------------------
    # population fractions
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_cs = pk.load(f)
        pf = ds_pf_cs['mean_frac_unprec_all_b_y0'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}] * 100
        list_extrs_pf.append(pf)
        
    da_pf_cs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})
    
    # plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        da_pf_cs_extrs.loc[{
            'hazard':extr,
            'birth_year':np.arange(1960,2021),
        }].plot(
            x='birth_year',
            y='GMT',
            ax=ax,
            add_labels=False,
            levels=10,
        ) 
        
        ax.set_yticks(
            ticks=GMT_indices_ticks,
            labels=gmts2100
        )
        ax.set_xticks(
            ticks=np.arange(1960,2025,10),
        )    
        # ax.set_ylabel('GMT anomaly at 2100 [°C]')
        # ax.set_xlabel('Birth year')
        
    # ax stuff
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            letters[n],
            loc='left',
            fontweight='bold',
        )
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
        )            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)                 
        if not np.isin(n,[0,3]):
            ax.yaxis.set_ticklabels([])
        else:
            ax.set_ylabel('GMT anomaly at 2100 [°C]')
        if n <= 2:
            ax.tick_params(labelbottom=False)    
        if n >= 3:
            ax.set_xlabel('Birth year')    
 
    f.savefig('./figures/cs_pf_heatmap_combined.png',dpi=800)
    plt.show()     


# %% ======================================================================================================
def boxplot_heatwave_egu(
    da_gs_popdenom,
    flags,   
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
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failure',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }     
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    
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

    # get data
    df_list_gs = []
    # for extr in extremes:
    extr='heatwavedarea'
    with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
        d_isimip_meta = pk.load(f)              
    with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
        ds_pf_gs_plot = pk.load(f)
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
    # GMT_indices_plot = [6]
    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)
    
    # pf plot
    x=8
    y=4
    f,ax = plt.subplots(
        figsize=(x,y),
    )  
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    i = 0
    # for ax,extr in zip(axes.flatten(),extremes):
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
    i+=1    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)      
    ax.tick_params(colors='gray')
    ax.set_ylim(0,100)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')      
    ax.set_ylabel('Population % living unprecedented \n exposure to heatwaves',color='gray')
    ax.set_xlabel('Birth year',color='gray')    

    legendcols = list(colors.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2])
    ]
    # labels= ['1.5 °C','2.5 °C','3.5 °C']
    
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
    
    f.savefig('./figures/gs_boxplots_heatwave_pf_35.png',dpi=800,bbox_inches='tight')
    f.savefig('./figures/gs_boxplots_heatwave_pf_35.eps',format='eps',bbox_inches='tight')
    
    
# %% ======================================================================================================
def boxplot_combined_cs_pf(
    ds_cohorts,
    gridscale_countries,
    flags,
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
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failure',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }     
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    
    # bbox
    x0 = 0
    y0 = 0.5
    xlen = 1
    ylen = 0.9
    
    # space between entries
    legend_entrypad = 0.5
    
    # length per entry
    legend_entrylen = 0.75
    
    legend_font = 14
    legend_lw=3.5    

    # get data
    df_list_cs = []
    for extr in extremes:
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)              
        with open('./data/pickles/{}/pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_cs = pk.load(f)

        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)    
                    
                    
        # country scale pop emerging --------------------------------------------------------
        da_p_plot = ds_pf_cs['unprec_country_b_y0'].loc[{
            'GMT':GMT_indices_plot,
            'country':gridscale_countries,
            'birth_year':sample_birth_years,
        }] * 1000
        
        # this loop is done to make sure that for each GMT, we only include sims with valid mapping (and not unecessary nans that skew distribution and denominator when turned to 0)
        for step in GMT_indices_plot:
            da_pf_plot_step = da_p_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') \
                / (ds_cohorts['by_population_y0'].loc[{'country':gridscale_countries,'birth_year':sample_birth_years}].sum(dim='country') * 1000) * 100
            df_pf_plot_step = da_pf_plot_step.to_dataframe(name='pf').reset_index()
            df_pf_plot_step['GMT_label'] = df_pf_plot_step['GMT'].map(gmt_legend)       
            df_pf_plot_step['hazard'] = extr
            df_list_cs.append(df_pf_plot_step)
            
    df_pf_plot = pd.concat(df_list_cs)
    
    # pf plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )  
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    i = 0
    for ax,extr in zip(axes.flatten(),extremes):
        p = sns.boxplot(
            data=df_pf_plot[df_pf_plot['hazard']==extr],
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
        i+=1
    # ax stuff
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            letters[n],
            loc='left',
            fontweight='bold',
        )
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
        )            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        
        if not np.isin(n,[0,3]):
            # ax.yaxis.set_ticklabels([])
            ax.set_ylabel(None)
        else:
            ax.set_ylabel('Population %')
        if n <= 2:
            ax.tick_params(labelbottom=False)    
            ax.set_xlabel(None)
        if n >= 3:
            ax.set_xlabel('Birth year')    
    
        legendcols = list(colors.values())
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),\
            Rectangle((0,0),1,1,color=legendcols[1]),\
            Rectangle((0,0),1,1,color=legendcols[2])
        ]
        labels= ['1.5 °C','2.5 °C','3.5 °C']
        axes.flatten()[1].legend(
            handles, 
            labels, 
            bbox_to_anchor=(x0, y0, xlen, ylen), 
            # loc=9,   #bbox: (x, y, width, height)
            ncol=3,
            fontsize=legend_font, 
            mode="expand", 
            borderaxespad=0.,\
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad
        )              
    
    f.savefig('./figures/cs_boxplots_combined_pf.png',dpi=800,bbox_inches='tight')
    

# %% ======================================================================================================
def boxplot_combined_gs_p(
    flags,   
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
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failure',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }     
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    
    # bbox
    x0 = 0
    y0 = 0.5
    xlen = 1
    ylen = 0.9
    
    # space between entries
    legend_entrypad = 0.5
    
    # length per entry
    legend_entrylen = 0.75
    
    legend_font = 14
    legend_lw=3.5    

    # get data
    df_list_gs = []
    for extr in extremes:
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)              
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs_plot = pk.load(f)
        da_p_gs_plot = ds_pf_gs_plot['unprec'].loc[{
            'GMT':GMT_indices_plot,
            'birth_year':sample_birth_years,
        }] / 10**6
        
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)
                    
        da_p_gs_plot = da_p_gs_plot.sum(dim='country')
        for step in GMT_indices_plot:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}]
            df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe().reset_index()
            df_p_gs_plot_step['GMT_label'] = df_p_gs_plot_step['GMT'].map(gmt_legend)  
            df_p_gs_plot_step['hazard'] = extr
            df_list_gs.append(df_p_gs_plot_step)
        
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['unprec'] = df_p_gs_plot['unprec'].fillna(0)   
    
    # p plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )  
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    i = 0
    for ax,extr in zip(axes.flatten(),extremes):
        p = sns.boxplot(
            data=df_p_gs_plot[df_p_gs_plot['hazard']==extr],
            x='birth_year',
            y='unprec',
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
        i+=1
    # ax stuff
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            letters[n],
            loc='left',
            fontweight='bold',
        )
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
        )            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        
        if not np.isin(n,[0,3]):
            # ax.yaxis.set_ticklabels([])
            ax.set_ylabel(None)
        else:
            ax.set_ylabel('Millions unprecedented')
        if n <= 2:
            ax.tick_params(labelbottom=False)    
            ax.set_xlabel(None)
        if n >= 3:
            ax.set_xlabel('Birth year')    
    
        legendcols = list(colors.values())
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),\
            Rectangle((0,0),1,1,color=legendcols[1]),\
            Rectangle((0,0),1,1,color=legendcols[2])
        ]
        labels= ['1.5 °C','2.5 °C','3.5 °C']
        axes.flatten()[1].legend(
            handles, 
            labels, 
            bbox_to_anchor=(x0, y0, xlen, ylen), 
            # loc=9,   #bbox: (x, y, width, height)
            ncol=3,
            fontsize=legend_font, 
            mode="expand", 
            borderaxespad=0.,\
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad
        )              
    
    f.savefig('./figures/gs_boxplots_combined_p.png',dpi=800,bbox_inches='tight')


# %% ======================================================================================================

def plot_conceptual_egu_full(
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
    with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
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
            # add_labels=False,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':1960,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax,
            color=colors[step],
            linewidth=3,
            zorder=4,
            # add_labels=False,
        )
        
    # ax.set_title(None)
    # ax.annotate(
    #     letters[l],
    #     # (1962,ax.get_ylim()[-1]+2),
    #     (1960,ax.get_ylim()[-1]+3),
    #     xycoords=ax.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='k',
    #     fontweight='bold',
    # )    
    # l+=1    
    # ax.set_title(
    #     letters[l],
    #     loc='left',
    #     fontweight='bold',
    #     fontsize=10,
    # )
    # l+=1        
    end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])    
    # ax.tick_params(labelleft=False)    
    ax.annotate(
        'Born in 1960',
        # (1962,ax.get_ylim()[-1]+2),
        (1965,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )    
    ax.set_title(None)
    ax.annotate(
        letters[l],
        # (1962,ax.get_ylim()[-1]+2),
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
        # fill=True,
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
    # ax2_l = 1960
    ax2_l = 1990
    ax2_b = da_pic_city_9999 *2
    # ax2_w = 1990-1960+np.floor(df_life_expectancy_5.loc[1990,cntry])
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
            # add_labels=False,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax2,
            color=colors[step],
            linewidth=3,
            zorder=4,
            # add_labels=False,
        )    
        
    # ax2.set_title(None)
    # # ax2.set_title(
    # #     letters[l],
    # #     loc='left',
    # #     fontweight='bold',
    # #     fontsize=10,
    # # )
    # ax2.annotate(
    #     letters[l],
    #     (1990,ax2.get_ylim()[-1]),
    #     xycoords=ax2.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='k',
    #     fontweight='bold',
    # )     
    # l+=1              
    end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
    # ax2.set_title(
    #     letters[l],
    #     loc='left',
    #     fontweight='bold',
    # )
    # l+=1
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)
    ax2.set_yticks([0,5,10])
    ax2.set_yticklabels([None,5,10])  
    ax2.set_xticks(np.arange(1990,2071,10))      
    ax2.set_xticklabels([None,2000,None,2020,None,2040,None,2060,None])
    ax2.set_xlim(
        # 1960,
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
    # ax2.tick_params(labelleft=False)    
    ax2.tick_params(colors='gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['bottom'].set_color('gray')
    
    ax2.annotate(
        'Born in 1990',
        # (1992,ax2.get_ylim()[-1]),
        (1995,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )
    ax2.set_title(None)
    # ax2.set_title(
    #     letters[l],
    #     loc='left',
    #     fontweight='bold',
    #     fontsize=10,
    # )
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
    
    # ax2.set_title(None)
    # ax2.annotate(
    #     letters[l],
    #     # (1962,ax.get_ylim()[-1]+2),
    #     (1990,ax2.get_ylim()[-1]),
    #     xycoords=ax.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='k',
    #     fontweight='bold',
    # )       

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
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
        '{} \npeople'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':1990,'lat':city_lat,'lon':city_lon},method='nearest').item())))),
        # (1962,ax.get_ylim()[-1]+2),
        (1.1,0.6),
        xycoords=ax2_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )             

    # ------------------------------------------------------------------   
    # 2020 time series
    # ax3_l = 1960
    ax3_l = 2020
    ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    # ax3_w = 2020-1960+np.floor(df_life_expectancy_5.loc[2020,cntry])
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
            # add_labels=False,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax3,
            color=colors[step],
            linewidth=3,
            zorder=4,
            # add_labels=False,
        )    
    # plot std uncertainty bars
    # for step in GMT_indices_plot:
    #     ax3.fill_between(
    #         x=da_test_city_std.loc[{'birth_year':2020,'GMT':step}].time.data,
    #         y1=da_test_city.loc[{'birth_year':2020,'GMT':step}] + da_test_city_std.loc[{'birth_year':2020,'GMT':step}],
    #         y2=da_test_city.loc[{'birth_year':2020,'GMT':step}] - da_test_city_std.loc[{'birth_year':2020,'GMT':step}],
    #         color=colors[step],
    #         alpha=0.2
    #     )
    end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])
    # ax3.set_title(None)
    # ax3.set_title(
    #     letters[l],
    #     loc='left',
    #     fontweight='bold',
    #     fontsize=10,
    # )
    # l+=1  
    # ax3.set_title(None)
    # ax3.annotate(
    #     letters[l],
    #     # (1962,ax.get_ylim()[-1]+2),
    #     (2020,ax3.get_ylim()[-1]),
    #     xycoords=ax3.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='k',
    #     fontweight='bold',
    # ) 
    # l+=1        
    
    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.set_yticks([0,5,10,15,20,25])
    ax3.set_yticklabels([None,5,10,15,20,25])   
    ax3.set_xticks(np.arange(2020,2101,10))      
    ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    ax3.set_xlim(
        # 1960,
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
    # ax3.tick_params(labelleft=False)    
    ax3.tick_params(colors='gray')
    ax3.spines['left'].set_color('gray')
    ax3.spines['bottom'].set_color('gray')

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
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
        # (2022,ax3.get_ylim()[-1]),
        (2025,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )
    ax3.set_title(None)
    ax3.annotate(
        letters[l],
        # (1962,ax.get_ylim()[-1]+2),
        (2020,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    ) 
    l+=1      
    
    # ax3.set_title(None)
    # ax3.annotate(
    #     letters[l],
    #     # (1962,ax.get_ylim()[-1]+2),
    #     (2020,ax3.get_ylim()[-1]),
    #     xycoords=ax.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='k',
    #     fontweight='bold',
    # ) 
    # l+=1       

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
        '{} \npeople'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item())))),
        # (1962,ax.get_ylim()[-1]+2),
        (1.1,0.6),
        xycoords=ax3_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
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
        # loc=3
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    f.savefig('./figures/concept_{}_{}_egu.png'.format(city_name,cntry),dpi=900,bbox_inches='tight')
    f.savefig('./figures/concept_{}_{}_egu.eps'.format(city_name,cntry),format='eps',bbox_inches='tight')

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item()

# %% ======================================================================================================

def plot_conceptual_egu_1960_15(
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
    with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
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

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }

    # ------------------------------------------------------------------   
    # for step in GMT_indices_plot:
        # 1960 time series
    f,ax = plt.subplots(
        figsize=(x,y)
    )
    for step in GMT_indices_plot:
    # step=6
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
    ax.set_title(None)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])    
    # ax.tick_params(labelleft=False)    
    ax.annotate(
        'Born in 1960',
        (1962,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )
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
        # fill=True,
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
        
    # # ------------------------------------------------------------------       
    # # 1990 time series
    # # ax2_l = 1960
    # ax2_l = 1990
    # ax2_b = da_pic_city_9999 *2
    # # ax2_w = 1990-1960+np.floor(df_life_expectancy_5.loc[1990,cntry])
    # ax2_w = np.floor(df_life_expectancy_5.loc[1990,cntry])
    # ax2_h = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())
    # ax2 = ax.inset_axes(
    #     bounds=(ax2_l, ax2_b, ax2_w, ax2_h),
    #     transform=ax.transData,
    # )

    # for step in GMT_indices_plot:
    #     da_test_city.loc[{'birth_year':1990,'GMT':step}].plot.line(
    #         ax=ax2,
    #         color=colors[step],
    #         linewidth=1,
    #     )
    #     # bold line for emergence
    #     da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
    #     da = da.where(da>da_pic_city_9999)
    #     da.plot.line(
    #         ax=ax2,
    #         color=colors[step],
    #         linewidth=3,
    #         zorder=4,
    #     )    
    # end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
    # ax2.set_title(None)
    # ax2.set_ylabel(None)
    # ax2.set_xlabel(None)
    # ax2.set_yticks([0,5,10])
    # ax2.set_yticklabels([None,5,10])  
    # ax2.set_xticks(np.arange(1990,2071,10))      
    # ax2.set_xticklabels([None,2000,None,2020,None,2040,None,2060,None])
    # ax2.set_xlim(
    #     # 1960,
    #     1990,
    #     end_year,
    # )
    # ax2.set_ylim(
    #     0,
    #     np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())+1,
    # )
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)  
    # ax2.spines['left'].set_position(('data',1990))
    # # ax2.tick_params(labelleft=False)    
    # ax2.tick_params(colors='gray')
    # ax2.spines['left'].set_color('gray')
    # ax2.spines['bottom'].set_color('gray')

    # # get time of first line to cross PIC thresh
    # emergences = []
    # for step in GMT_indices_plot:
    #     da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
    #     da = da.where(da>da_pic_city_9999)
    #     da_t = da.time.where(da == da.min()).dropna(dim='time').item()
    #     emergences.append(da_t)
    # first_emerge = np.min(emergences)

    # ax2.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=first_emerge, 
    #     xmax=end_year, 
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    # ax2.annotate(
    #     'Born in 1990',
    #     (1992,ax2.get_ylim()[-1]),
    #     xycoords=ax2.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='gray',
    # )

    # # 1990 pdf
    # ax2_pdf_l = end_year+5
    # ax2_pdf_b = -2
    # ax2_pdf_w = 20
    # ax2_pdf_h = ax2.get_ylim()[-1]+2
    # ax2_pdf = ax2.inset_axes(
    #     bounds=(ax2_pdf_l, ax2_pdf_b, ax2_pdf_w, ax2_pdf_h),
    #     transform=ax2.transData,
    # )
    # sns.histplot(
    #     data=df_pic_city.round(),
    #     y='lifetime_exposure',
    #     color='lightgrey',
    #     discrete = True,
    #     ax=ax2_pdf
    # )
    # ax2_pdf.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=0, 
    #     xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    # for step in GMT_indices_plot:
    #     ax2_pdf.hlines(
    #         y=da_test_city.loc[{'birth_year':1990,'GMT':step}].max(), 
    #         xmin=0, 
    #         xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    #         colors=colors[step], 
    #         linewidth=1, 
    #         linestyle='-', 
    #         label=gmt_legend[step], 
    #         zorder=2
    #     )
    # ax2_pdf.spines['right'].set_visible(False)
    # ax2_pdf.spines['top'].set_visible(False)      
    # ax2_pdf.set_ylabel(None)
    # ax2_pdf.set_xlabel(None)
    # ax2_pdf.set_ylim(-2,ax2.get_ylim()[-1])
    # ax2_pdf.tick_params(colors='gray')
    # ax2_pdf.spines['left'].set_color('gray')
    # ax2_pdf.spines['bottom'].set_color('gray')

    # # ------------------------------------------------------------------   
    # # 2020 time series
    # # ax3_l = 1960
    # ax3_l = 2020
    # ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    # # ax3_w = 2020-1960+np.floor(df_life_expectancy_5.loc[2020,cntry])
    # ax3_w = np.floor(df_life_expectancy_5.loc[2020,cntry])
    # ax3_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())
    # ax3 = ax2.inset_axes(
    #     bounds=(ax3_l, ax3_b, ax3_w, ax3_h),
    #     transform=ax2.transData,
    # )
    # # plot mean lines
    # for step in GMT_indices_plot:
    #     da_test_city.loc[{'birth_year':2020,'GMT':step}].plot.line(
    #         ax=ax3,
    #         color=colors[step],
    #         linewidth=1
    #     )
    #     # bold line for emergence
    #     da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
    #     da = da.where(da>da_pic_city_9999)
    #     da.plot.line(
    #         ax=ax3,
    #         color=colors[step],
    #         linewidth=3,
    #         zorder=4,
    #     )    
    # # plot std uncertainty bars
    # # for step in GMT_indices_plot:
    # #     ax3.fill_between(
    # #         x=da_test_city_std.loc[{'birth_year':2020,'GMT':step}].time.data,
    # #         y1=da_test_city.loc[{'birth_year':2020,'GMT':step}] + da_test_city_std.loc[{'birth_year':2020,'GMT':step}],
    # #         y2=da_test_city.loc[{'birth_year':2020,'GMT':step}] - da_test_city_std.loc[{'birth_year':2020,'GMT':step}],
    # #         color=colors[step],
    # #         alpha=0.2
    # #     )
    # end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])
    # ax3.set_title(None)
    # ax3.set_ylabel(None)
    # ax3.set_xlabel(None)
    # ax3.set_yticks([0,5,10,15,20,25])
    # ax3.set_yticklabels([None,5,10,15,20,25])   
    # ax3.set_xticks(np.arange(2020,2101,10))      
    # ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    # ax3.set_xlim(
    #     # 1960,
    #     2020,
    #     end_year,
    # )
    # ax3.set_ylim(
    #     0,
    #     np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
    # )
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)  
    # ax3.spines['left'].set_position(('data',2020))
    # # ax3.tick_params(labelleft=False)    
    # ax3.tick_params(colors='gray')
    # ax3.spines['left'].set_color('gray')
    # ax3.spines['bottom'].set_color('gray')

    # # get time of first line to cross PIC thresh
    # emergences = []
    # for step in GMT_indices_plot:
    #     da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
    #     da = da.where(da>da_pic_city_9999)
    #     da_t = da.time.where(da == da.min()).dropna(dim='time').item()
    #     emergences.append(da_t)
    # first_emerge = np.min(emergences)

    # ax3.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=first_emerge, 
    #     xmax=end_year, 
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    # ax3.annotate(
    #     'Born in 2020',
    #     (2022,ax3.get_ylim()[-1]),
    #     xycoords=ax3.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='gray',
    # )

    # # 2020 pdf
    # ax3_pdf_l = end_year+5
    # ax3_pdf_b = -2
    # ax3_pdf_w = 20
    # ax3_pdf_h = ax3.get_ylim()[-1]+2
    # ax3_pdf = ax3.inset_axes(
    #     bounds=(ax3_pdf_l, ax3_pdf_b, ax3_pdf_w, ax3_pdf_h),
    #     transform=ax3.transData,
    # )
    # sns.histplot(
    #     data=df_pic_city.round(),
    #     y='lifetime_exposure',
    #     color='lightgrey',
    #     discrete = True,
    #     ax=ax3_pdf
    # )
    # ax3_pdf.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=0, 
    #     xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    # for step in GMT_indices_plot:
    #     ax3_pdf.hlines(
    #         y=da_test_city.loc[{'birth_year':2020,'GMT':step}].max(), 
    #         xmin=0, 
    #         xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    #         colors=colors[step], 
    #         linewidth=1, 
    #         linestyle='-', 
    #         label=gmt_legend[step], 
    #         zorder=2
    #     )
    # ax3_pdf.spines['right'].set_visible(False)
    # ax3_pdf.spines['top'].set_visible(False)      
    # ax3_pdf.set_ylabel(None)
    # ax3_pdf.set_xlabel(None)
    # ax3_pdf.set_ylim(-2,ax3.get_ylim()[-1])
    # ax3_pdf.tick_params(colors='gray')
    # ax3_pdf.spines['left'].set_color('gray')
    # ax3_pdf.spines['bottom'].set_color('gray')

    # # City name
    # ax3.annotate(
    #     '{}, {}'.format(city_name,cntry),
    #     (1960,ax3.get_ylim()[-1]),
    #     xycoords=ax3.transData,
    #     fontsize=16,
    #     rotation='horizontal',
    #     color='gray',
    # )

    # axis labels ===================================================================

    # x axis label (time)
    x_i=1950
    y_i=-10
    # x_f=2040
    x_f=2110
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
    
    legendcols = list(colors.values())
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2]),
        Rectangle((0,0),1,1,color='lightgrey'),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color='gray'),
        # Rectangle((0,0),1,1,color=legendcols[4]),
    ]
    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',
        'pre-industrial lifetime \n exposure histogram',
        '99.99% pre-industrial \n lifetime exposure',
        # 'pre-industrial lifetime \n exposure histogram'
    ]
    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        # loc=3
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    f.savefig('./figures/concept_{}_{}_egu_1960_all.png'.format(city_name,cntry),dpi=900,bbox_inches='tight')

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest')

# %% ======================================================================================================

def plot_conceptual_egu_1990(
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
    with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
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
        # da = da_test_city.loc[{'birth_year':1960,'GMT':step}]
        # da = da.where(da>da_pic_city_9999)
        # da.plot.line(
        #     ax=ax,
        #     color=colors[step],
        #     linewidth=3,
        #     zorder=4,
        # )
    end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
    ax.set_title(None)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])    
    # ax.tick_params(labelleft=False)    
    ax.annotate(
        'Born in 1960',
        (1962,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )
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
        # fill=True,
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
        # da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        # da = da.where(da>da_pic_city_9999)
        # da.plot.line(
        #     ax=ax2,
        #     color=colors[step],
        #     linewidth=3,
        #     zorder=4,
        # )    
    end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
    ax2.set_title(None)
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

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
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
    ax2.annotate(
        'Born in 1990',
        (1992,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
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

    # ------------------------------------------------------------------   
    # 2020 time series
    # ax3_l = 2020
    # ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    # ax3_w = np.floor(df_life_expectancy_5.loc[2020,cntry])
    # ax3_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())
    # ax3 = ax2.inset_axes(
    #     bounds=(ax3_l, ax3_b, ax3_w, ax3_h),
    #     transform=ax2.transData,
    # )
    # # plot mean lines
    # for step in GMT_indices_plot:
    #     da_test_city.loc[{'birth_year':2020,'GMT':step}].plot.line(
    #         ax=ax3,
    #         color=colors[step],
    #         linewidth=1
    #     )
    #     # bold line for emergence
    #     da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
    #     da = da.where(da>da_pic_city_9999)
    #     da.plot.line(
    #         ax=ax3,
    #         color=colors[step],
    #         linewidth=3,
    #         zorder=4,
    #     )    
    # end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])
    # ax3.set_title(None)
    # ax3.set_ylabel(None)
    # ax3.set_xlabel(None)
    # ax3.set_yticks([0,5,10,15,20,25])
    # ax3.set_yticklabels([None,5,10,15,20,25])   
    # ax3.set_xticks(np.arange(2020,2101,10))      
    # ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    # ax3.set_xlim(
    #     # 1960,
    #     2020,
    #     end_year,
    # )
    # ax3.set_ylim(
    #     0,
    #     np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
    # )
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)  
    # ax3.spines['left'].set_position(('data',2020))
    # ax3.tick_params(colors='gray')
    # ax3.spines['left'].set_color('gray')
    # ax3.spines['bottom'].set_color('gray')

    # # get time of first line to cross PIC thresh
    # emergences = []
    # for step in GMT_indices_plot:
    #     da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
    #     da = da.where(da>da_pic_city_9999)
    #     da_t = da.time.where(da == da.min()).dropna(dim='time').item()
    #     emergences.append(da_t)
    # first_emerge = np.min(emergences)

    # ax3.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=first_emerge, 
    #     xmax=end_year, 
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    # ax3.annotate(
    #     'Born in 2020',
    #     (2022,ax3.get_ylim()[-1]),
    #     xycoords=ax3.transData,
    #     fontsize=10,
    #     rotation='horizontal',
    #     color='gray',
    # )

    # # 2020 pdf
    # ax3_pdf_l = end_year+5
    # ax3_pdf_b = -2
    # ax3_pdf_w = 20
    # ax3_pdf_h = ax3.get_ylim()[-1]+2
    # ax3_pdf = ax3.inset_axes(
    #     bounds=(ax3_pdf_l, ax3_pdf_b, ax3_pdf_w, ax3_pdf_h),
    #     transform=ax3.transData,
    # )
    # sns.histplot(
    #     data=df_pic_city.round(),
    #     y='lifetime_exposure',
    #     color='lightgrey',
    #     discrete = True,
    #     ax=ax3_pdf
    # )
    # ax3_pdf.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=0, 
    #     xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    # for step in GMT_indices_plot:
    #     ax3_pdf.hlines(
    #         y=da_test_city.loc[{'birth_year':2020,'GMT':step}].max(), 
    #         xmin=0, 
    #         xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
    #         colors=colors[step], 
    #         linewidth=1, 
    #         linestyle='-', 
    #         label=gmt_legend[step], 
    #         zorder=2
    #     )
    # ax3_pdf.spines['right'].set_visible(False)
    # ax3_pdf.spines['top'].set_visible(False)      
    # ax3_pdf.set_ylabel(None)
    # ax3_pdf.set_xlabel(None)
    # ax3_pdf.set_ylim(-2,ax3.get_ylim()[-1])
    # ax3_pdf.tick_params(colors='gray')
    # ax3_pdf.spines['left'].set_color('gray')
    # ax3_pdf.spines['bottom'].set_color('gray')

    # City name
    # ax3.annotate(
    #     '{}, {}'.format(city_name,cntry),
    #     (1960,ax3.get_ylim()[-1]),
    #     xycoords=ax3.transData,
    #     fontsize=16,
    #     rotation='horizontal',
    #     color='gray',
    # )

    # axis labels ===================================================================

    # x axis label (time)
    x_i=1950
    y_i=-10
    # x_f=2040
    x_f=2110
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
    
    legendcols = list(colors.values())
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2]),
        Rectangle((0,0),1,1,color='lightgrey'),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color='gray'),
    ]
    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',
        'pre-industrial lifetime \n exposure histogram',
        '99.99% pre-industrial \n lifetime exposure',
    ]
    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        # loc=3
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    f.savefig('./figures/concept_{}_{}_egu_1990.png'.format(city_name,cntry),dpi=900,bbox_inches='tight')

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest')


# %% ======================================================================================================

def plot_conceptual_egu_2020(
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
    with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
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
        # da = da_test_city.loc[{'birth_year':1960,'GMT':step}]
        # da = da.where(da>da_pic_city_9999)
        # da.plot.line(
        #     ax=ax,
        #     color=colors[step],
        #     linewidth=3,
        #     zorder=4,
        # )
    end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
    ax.set_title(None)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])    
    # ax.tick_params(labelleft=False)    
    ax.annotate(
        'Born in 1960',
        (1962,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )
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
        # fill=True,
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
    ax2.set_title(None)
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

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
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
    ax2.annotate(
        'Born in 1990',
        (1992,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
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

    # ------------------------------------------------------------------   
    # 2020 time series
    ax3_l = 2020
    ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.6
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
            linewidth=1
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
    ax3.set_title(None)
    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.set_yticks([0,5,10,15,20,25])
    ax3.set_yticklabels([None,5,10,15,20,25])   
    ax3.set_xticks(np.arange(2020,2101,10))      
    ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    ax3.set_xlim(
        # 1960,
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
        (2022,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )

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


    # axis labels ===================================================================

    # x axis label (time)
    x_i=1950
    y_i=-10
    # x_f=2040
    x_f=2110
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
    
    legendcols = list(colors.values())
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2]),
        Rectangle((0,0),1,1,color='lightgrey'),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color='gray'),
    ]
    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',
        'pre-industrial lifetime \n exposure histogram',
        '99.99% pre-industrial \n lifetime exposure',
    ]
    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        # loc=3
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    f.savefig('./figures/concept_{}_{}_egu_2020_boldemerge.png'.format(city_name,cntry),dpi=900,bbox_inches='tight')

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest')



# %%
