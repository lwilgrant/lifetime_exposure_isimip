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
import mapclassify as mc
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs

#%% --------------------------------------------------------------------
# test colors for plotting

def c(x):
    col = plt.cm.OrRd(x)
    fig, ax = plt.subplots(figsize=(1,1))
    fig.set_facecolor(col)
    ax.axis("off")
    plt.show()
    
#%% --------------------------------------------------------------------
# convert floats to color category 
    
def floater(f):
    if f < 1.5:
        col = 'low'
    elif f >=1.5 and f < 2.5:
        col = 'med'
    elif f >= 2.5:
        col = 'hi'
    return col
    
#%% ----------------------------------------------------------------
# plot timing and EMF of exceedence of pic-defined extreme
def spatial_emergence_plot(
    gdf_exposure_emergence_birth_year,
):
    
    # plot
    f,axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(20,16),
        subplot_kw = dict(projection=ccrs.PlateCarree()),
    )

    # letters
    letters = ['a','b','c','d','e','f','g','h','i','j','k']

    # trajectory labels
    trj_labs = {
        '15':'1.5 °C',
        '20':'2.0 °C',
        'NDC':'NDC'
    }
    
    # placment birth year cbar
    cb_by_x0 = 0.185
    cb_by_y0 = 0.05
    cb_by_xlen = 0.225
    cb_by_ylen = 0.015

    # placment emf cbar
    cb_emf_x0 = 0.60
    cb_emf_y0 = 0.05
    cb_emf_xlen = 0.225
    cb_emf_ylen = 0.015

    # identify colors for birth years
    cmap_by = plt.cm.get_cmap('viridis')
    cmap55 = cmap_by(0.01)
    cmap50 = cmap_by(0.05)   #light
    cmap45 = cmap_by(0.1)
    cmap40 = cmap_by(0.15)
    cmap35 = cmap_by(0.2)
    cmap30 = cmap_by(0.25)
    cmap25 = cmap_by(0.3)
    cmap20 = cmap_by(0.325)
    cmap10 = cmap_by(0.4)
    cmap5 = cmap_by(0.475)
    cmap0 = 'gray'
    cmap_5 = cmap_by(0.525)
    cmap_10 = cmap_by(0.6)
    cmap_20 = cmap_by(0.625)
    cmap_25 = cmap_by(0.7)
    cmap_30 = cmap_by(0.75)
    cmap_35 = cmap_by(0.8)
    cmap_40 = cmap_by(0.85)
    cmap_45 = cmap_by(0.9)
    cmap_50 = cmap_by(0.95)  #dark
    cmap_55 = cmap_by(0.99)

    # colors_by = [
    #     cmap55,cmap45,cmap35,cmap25,cmap10,cmap5, # 6 dark colors for 1960 - 1990
    #     cmap_5,cmap_10,cmap_25,cmap_35,cmap_45,cmap_55, # 6 light colors for 1990-2020
    # ]
    colors_by = [
        cmap45,cmap35,cmap30,cmap25,cmap10,cmap5, # 6 dark colors for 1960 - 1965
        cmap_5,cmap_10,cmap_25,cmap_30,cmap_35, # 6 light colors for 1966-1970
    ]    

    # declare list of colors for discrete colormap of colorbar for birth years
    cmap_list_by = mpl.colors.ListedColormap(colors_by,N=len(colors_by))
    cmap_list_by.set_under(cmap55)
    cmap_list_by.set_over(cmap_45)

    # colorbar args for birth years
    # values_by = [1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
    values_by = [1959.5,1960.5,1961.5,1962.5,1963.5,1964.5,1965.5,1966.5,1967.5,1968.5,1969.5,1970.5]
    # tick_locs_by = [1960,1970,1980,1990,2000,2010,2020]
    tick_locs_by = [1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970]
    tick_labels_by = list(str(n) for n in tick_locs_by)
    norm_by = mpl.colors.BoundaryNorm(values_by,cmap_list_by.N)

    # identify colors for EMF
    cmap_emf = plt.cm.get_cmap('OrRd')
    cmap55 = cmap_emf(0.01)
    cmap50 = cmap_emf(0.05)   #purple
    cmap45 = cmap_emf(0.1)
    cmap40 = cmap_emf(0.15)
    cmap35 = cmap_emf(0.2)
    cmap30 = cmap_emf(0.25)
    cmap25 = cmap_emf(0.3)
    cmap20 = cmap_emf(0.325)
    cmap10 = cmap_emf(0.4)
    cmap5 = cmap_emf(0.475)
    cmap0 = 'gray'
    cmap_5 = cmap_emf(0.525)
    cmap_10 = cmap_emf(0.6)
    cmap_20 = cmap_emf(0.625)
    cmap_25 = cmap_emf(0.7)
    cmap_30 = cmap_emf(0.75)
    cmap_35 = cmap_emf(0.8)
    cmap_40 = cmap_emf(0.85)
    cmap_45 = cmap_emf(0.9)
    cmap_50 = cmap_emf(0.95)  #yellow
    cmap_55 = cmap_emf(0.99)

    # lump EMF data across scenarios for common colorbar
    colors_emf = [
        cmap55,cmap35,cmap10, # 3 light colors for low emfs
        cmap_10,cmap_35,cmap_55, # 6 dark colors for high emfs
    ]
    # declare list of colors for discrete colormap of colorbar for emf
    cmap_list_emf = mpl.colors.ListedColormap(colors_emf,N=len(colors_emf))

    data = np.empty(1)
    for trj in ['15','20','NDC']:
        data = np.append(data,gdf_exposure_emergence_birth_year.loc[:,'birth_year_age_{}'.format(trj)].values)        
    data = data[~np.isnan(data)]
    q_samples = []
    q_samples.append(np.abs(np.quantile(data,0.95)))
    q_samples.append(np.abs(np.quantile(data,0.05)))
        
    start = np.around(np.min(q_samples),decimals=4)
    inc = np.around(np.max(q_samples),decimals=4)/6
    values_emf = [
        np.around(start,decimals=1),
        np.around(start+inc,decimals=1),
        np.around(start+inc*2,decimals=1),
        np.around(start+inc*3,decimals=1),
        np.around(start+inc*4,decimals=1),
        np.around(start+inc*5,decimals=1),
        np.around(start+inc*6,decimals=1),
    ]
    tick_locs_emf = [
        np.around(start,decimals=1),
        np.around(start+inc,decimals=1),
        np.around(start+inc*2,decimals=1),
        np.around(start+inc*3,decimals=1),
        np.around(start+inc*4,decimals=1),
        np.around(start+inc*5,decimals=1),
        np.around(start+inc*6,decimals=1),
    ]
    tick_labels_emf = list(str(n) for n in tick_locs_emf)
    norm_emf = mpl.colors.BoundaryNorm(values_emf,cmap_list_emf.N)

    # colorbar axes
    cbax_by = f.add_axes([
        cb_by_x0, 
        cb_by_y0, 
        cb_by_xlen, 
        cb_by_ylen
    ])    

    cbax_emf = f.add_axes([
        cb_emf_x0, 
        cb_emf_y0, 
        cb_emf_xlen, 
        cb_emf_ylen
    ])    
    l = 0
    for row,trj in zip(axes,['15','20','NDC']):
        
        for i,ax in enumerate(row):
            
            # plot birth years
            if i == 0:
                
                gdf_exposure_emergence_birth_year.plot(
                    column='mmm_{}'.format(trj),
                    ax=ax,
                    norm=norm_by,
                    legend=False,
                    cmap=cmap_list_by,
                    cax=cbax_by,
                    # aspect="equal",
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "linewidth": 0.2,
                        "hatch": "///",
                        "label": "Missing values",
                    },
                )
                ax.set_yticks([])
                ax.set_xticks([])
                ax.text(
                    -0.07, 0.55, 
                    trj_labs[trj], 
                    va='bottom', 
                    ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                    fontweight='bold',
                    fontsize=16,
                    rotation='vertical', 
                    rotation_mode='anchor',
                    transform=ax.transAxes
                )            
            
            # plot associated age emergence
            else:
                
                gdf_exposure_emergence_birth_year.plot(
                    column='birth_year_age_{}'.format(trj),
                    ax=ax,
                    norm=norm_emf,
                    legend=False,
                    cmap=cmap_list_emf,
                    cax=cbax_emf,
                    # aspect="equal",
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "linewidth": 0.2,
                        "hatch": "///",
                        "label": "Missing values",
                    },                
                )
                ax.set_yticks([])
                ax.set_xticks([])     
            
            ax.set_title(
                letters[l],
                loc='left',
                fontsize = 16,
                fontweight='bold'
            )
            ax.set_aspect('equal')
            l += 1
            
                
    # birth year colorbar
    cb_by = mpl.colorbar.ColorbarBase(
        ax=cbax_by, 
        cmap=cmap_list_by,
        norm=norm_by,
        spacing='uniform',
        orientation='horizontal',
        extend='both',
        ticks=tick_locs_by,
        drawedges=False,
    )
    cb_by.set_label(
        'Birth year of cohort emergence',
        size=16,
    )
    cb_by.ax.xaxis.set_label_position('top')
    cb_by.ax.tick_params(
        labelcolor='0',
        labelsize=16,
        color='0.5',
        length=3.5,
        width=0.4,
        direction='out',
    ) 
    cb_by.ax.set_xticklabels(
        tick_labels_by,
        rotation=45,
    )
    cb_by.outline.set_edgecolor('0.9')
    cb_by.outline.set_linewidth(0)                  
                
    # emf colorbar
    cb_emf = mpl.colorbar.ColorbarBase(
        ax=cbax_emf, 
        cmap=cmap_list_emf,
        norm=norm_emf,
        spacing='uniform',
        orientation='horizontal',
        extend='neither',
        ticks=tick_locs_emf,
        drawedges=False,
    )
    cb_emf.set_label(
        'Age of emergence',
        size=16,
    )
    cb_emf.ax.xaxis.set_label_position('top')
    cb_emf.ax.tick_params(
        labelcolor='0',
        labelsize=16,
        color='0.5',
        length=3.5,
        width=0.4,
        direction='out',
    ) 
    cb_emf.ax.set_xticklabels(
        tick_labels_emf,
        rotation=45,
    )
    cb_emf.outline.set_edgecolor('0.9')
    cb_emf.outline.set_linewidth(0)
    
    f.savefig('./figures/birth_year_emergence.png',dpi=300)


#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac(
    ds_pop_frac_NDC,
    ds_pop_frac_15,
    ds_pop_frac_20,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=9
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    legend_lw=3.5 # legend line width
    x0 = 0.1 # bbox for legend
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

    ax1_ylab = 'Billions unprecendented'
    ax2_ylab = 'Unprecedented/Total'
    ax3_ylab = 'Unprecedented/Exposed'

    f,(ax1,ax2,ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot unprecedented pop numbers

    # NDC
    ax1.plot(
        time,
        # ds_pop_frac['mean_unprec'].values * 1000,
        ds_pop_frac_NDC['mean_unprec'].values / 1e6,
        lw=lw_mean,
        color=col_NDC,
        label='NDC',
        zorder=1,
    )
    ax1.fill_between(
        time,
        # (ds_pop_frac['mean_unprec'].values * 1000) + (ds_pop_frac['std_unprec'].values * 1000),
        (ds_pop_frac_NDC['mean_unprec'].values / 1e6) + (ds_pop_frac_NDC['std_unprec'].values / 1e6),
        # (ds_pop_frac['mean_unprec'].values * 1000) - (ds_pop_frac['std_unprec'].values * 1000),
        (ds_pop_frac_NDC['mean_unprec'].values / 1e6) - (ds_pop_frac_NDC['std_unprec'].values / 1e6),
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax1.plot(
        time,
        # ds_pop_frac['mean_unprec'].values * 1000,
        ds_pop_frac_20['mean_unprec'].values / 1e6,
        lw=lw_mean,
        color=col_20,
        label='2.0 °C',
        zorder=2,
    )
    ax1.fill_between(
        time,
        # (ds_pop_frac['mean_unprec'].values * 1000) + (ds_pop_frac['std_unprec'].values * 1000),
        (ds_pop_frac_20['mean_unprec'].values / 1e6) + (ds_pop_frac_20['std_unprec'].values / 1e6),
        # (ds_pop_frac['mean_unprec'].values * 1000) - (ds_pop_frac['std_unprec'].values * 1000),
        (ds_pop_frac_20['mean_unprec'].values / 1e6) - (ds_pop_frac_20['std_unprec'].values / 1e6),
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax1.plot(
        time,
        # ds_pop_frac['mean_unprec'].values * 1000,
        ds_pop_frac_15['mean_unprec'].values / 1e6,
        lw=lw_mean,
        color=col_15,
        label='1.5 °C',
        zorder=3,
    )
    ax1.fill_between(
        time,
        # (ds_pop_frac['mean_unprec'].values * 1000) + (ds_pop_frac['std_unprec'].values * 1000),
        (ds_pop_frac_15['mean_unprec'].values / 1e6) + (ds_pop_frac_15['std_unprec'].values / 1e6),
        # (ds_pop_frac['mean_unprec'].values * 1000) - (ds_pop_frac['std_unprec'].values * 1000),
        (ds_pop_frac_15['mean_unprec'].values / 1e6) - (ds_pop_frac_15['std_unprec'].values / 1e6),
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_15_fill,
        zorder=3,
    )

    ax1.set_ylabel(
        ax1_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop

    # NDC
    ax2.plot(
        time,
        ds_pop_frac_NDC['mean_frac_all_unprec'].values,
        lw=lw_mean,
        color=col_NDC,
        zorder=1,
    )
    ax2.fill_between(
        time,
        ds_pop_frac_NDC['mean_frac_all_unprec'].values + ds_pop_frac_NDC['std_frac_all_unprec'].values,
        ds_pop_frac_NDC['mean_frac_all_unprec'].values - ds_pop_frac_NDC['std_frac_all_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax2.plot(
        time,
        ds_pop_frac_20['mean_frac_all_unprec'].values,
        lw=lw_mean,
        color=col_20,
        zorder=2,
    )
    ax2.fill_between(
        time,
        ds_pop_frac_20['mean_frac_all_unprec'].values + ds_pop_frac_20['std_frac_all_unprec'].values,
        ds_pop_frac_20['mean_frac_all_unprec'].values - ds_pop_frac_20['std_frac_all_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax2.plot(
        time,
        ds_pop_frac_15['mean_frac_all_unprec'].values,
        lw=lw_mean,
        color=col_15,
        zorder=3,
    )
    ax2.fill_between(
        time,
        ds_pop_frac_15['mean_frac_all_unprec'].values + ds_pop_frac_15['std_frac_all_unprec'].values,
        ds_pop_frac_15['mean_frac_all_unprec'].values - ds_pop_frac_15['std_frac_all_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_15_fill,
        zorder=3,
    )

    ax2.set_ylabel(
        ax2_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop

    # NDC
    ax3.plot(
        time,
        ds_pop_frac_NDC['mean_frac_exposed_unprec'].values,
        lw=lw_mean,
        color=col_NDC,
        zorder=1,
    )
    ax3.fill_between(
        time,
        ds_pop_frac_NDC['mean_frac_exposed_unprec'].values + ds_pop_frac_NDC['std_frac_exposed_unprec'].values,
        ds_pop_frac_NDC['mean_frac_exposed_unprec'].values - ds_pop_frac_NDC['std_frac_exposed_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax3.plot(
        time,
        ds_pop_frac_20['mean_frac_exposed_unprec'].values,
        lw=lw_mean,
        color=col_20,
        zorder=2,
    )
    ax3.fill_between(
        time,
        ds_pop_frac_20['mean_frac_exposed_unprec'].values + ds_pop_frac_20['std_frac_exposed_unprec'].values,
        ds_pop_frac_20['mean_frac_exposed_unprec'].values - ds_pop_frac_20['std_frac_exposed_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax3.plot(
        time,
        ds_pop_frac_15['mean_frac_exposed_unprec'].values,
        lw=lw_mean,
        color=col_15,
        zorder=3,
    )
    ax3.fill_between(
        time,
        ds_pop_frac_15['mean_frac_exposed_unprec'].values + ds_pop_frac_15['std_frac_exposed_unprec'].values,
        ds_pop_frac_15['mean_frac_exposed_unprec'].values - ds_pop_frac_15['std_frac_exposed_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_15_fill,
        zorder=3,
    )

    ax3.set_ylabel(
        ax3_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )

    for i,ax in enumerate([ax1,ax2,ax3]):
        ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
        ax.set_xlim(xmin,xmax)
        # ax.xaxis.set_ticks(xticks_ts)
        # ax.xaxis.set_ticklabels(xtick_labels_ts)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_axisbelow(True) 
        if i < 2:
            ax.tick_params(labelbottom=False)
            
    # legend
    legendcols = [
        col_NDC,
        col_20,
        col_15,
    ]
    handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2])]
    labels= [
        'NDC',
        '2.0 °C',
        '1.5 °C',
    ]
    
    ax1.legend(
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
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )            
            
    f.savefig('./figures/pop_frac.png',dpi=300)

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac_birth_year(
    ds_pop_frac_NDC,
    ds_pop_frac_15,
    ds_pop_frac_20,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=9
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    legend_lw=3.5 # legend line width
    x0 = 0.1 # bbox for legend
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
    xmax = 2020

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Fraction unprecedented'
    ax2_xlab = 'Birth year'

    for cohort_type in ['exposed','all']:

        f,(ax1,ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(x,y),
        )

        # --------------------------------------------------------------------
        # plot mean unprecedented frac of pop, ax1 for mean +/- std

        # NDC
        ax1.plot(
            time,
            ds_pop_frac_NDC['mean_frac_unprec_{}'.format(cohort_type)].values,
            lw=lw_mean,
            color=col_NDC,
            zorder=1,
        )
        ax1.fill_between(
            time,
            ds_pop_frac_NDC['mean_frac_unprec_{}'.format(cohort_type)].values + ds_pop_frac_NDC['std_frac_unprec_{}'.format(cohort_type)].values,
            ds_pop_frac_NDC['mean_frac_unprec_{}'.format(cohort_type)].values - ds_pop_frac_NDC['std_frac_unprec_{}'.format(cohort_type)].values,
            lw=lw_fill,
            alpha=ub_alpha,
            color=col_NDC_fill,
            zorder=1,
        )

        # 2.0 degrees
        ax1.plot(
            time,
            ds_pop_frac_20['mean_frac_unprec_{}'.format(cohort_type)].values,
            lw=lw_mean,
            color=col_20,
            zorder=2,
        )
        ax1.fill_between(
            time,
            ds_pop_frac_20['mean_frac_unprec_{}'.format(cohort_type)].values + ds_pop_frac_20['std_frac_unprec_{}'.format(cohort_type)].values,
            ds_pop_frac_20['mean_frac_unprec_{}'.format(cohort_type)].values - ds_pop_frac_20['std_frac_unprec_{}'.format(cohort_type)].values,
            lw=lw_fill,
            alpha=ub_alpha,
            color=col_20_fill,
            zorder=2,
        )

        # 1.5 degrees
        ax1.plot(
            time,
            ds_pop_frac_15['mean_frac_unprec_{}'.format(cohort_type)].values,
            lw=lw_mean,
            color=col_15,
            zorder=3,
        )
        ax1.fill_between(
            time,
            ds_pop_frac_15['mean_frac_unprec_{}'.format(cohort_type)].values + ds_pop_frac_15['std_frac_unprec_{}'.format(cohort_type)].values,
            ds_pop_frac_15['mean_frac_unprec_{}'.format(cohort_type)].values - ds_pop_frac_15['std_frac_unprec_{}'.format(cohort_type)].values,
            lw=lw_fill,
            alpha=ub_alpha,
            color=col_15_fill,
            zorder=3,
        )

        ax1.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )

        # --------------------------------------------------------------------
        # plot unprecedented frac of pop, all runs

        # NDC
        for run in ds_pop_frac_NDC.runs:
            
            ax2.plot(
                time,
                ds_pop_frac_NDC['frac_unprec_{}'.format(cohort_type)].sel(runs=run).values,
                lw=lw_mean,
                color=col_NDC,
                zorder=1,
            )
        # 2.0 degrees
        for run in ds_pop_frac_20.runs:
            
            ax2.plot(
                time,
                ds_pop_frac_20['frac_unprec_{}'.format(cohort_type)].sel(runs=run).values,
                lw=lw_mean,
                color=col_20,
                zorder=2,
            )
        # 1.5 degrees
        for run in ds_pop_frac_15.runs:

            ax2.plot(
                time,
                ds_pop_frac_15['frac_unprec_{}'.format(cohort_type)].sel(runs=run).values,
                lw=lw_mean,
                color=col_15,
                zorder=3,
            )

        ax2.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )
        ax2.set_xlabel(
            ax2_xlab, 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )    

        for i,ax in enumerate([ax1,ax2]):
            ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
            ax.set_xlim(xmin,xmax)
            # ax.xaxis.set_ticks(xticks_ts)
            # ax.xaxis.set_ticklabels(xtick_labels_ts)
            ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
            ax.tick_params(labelsize=tick_font,axis="y",direction="in")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.set_axisbelow(True) 
            if i < 1:
                ax.tick_params(labelbottom=False)
                
        # legend
        legendcols = [
            col_NDC,
            col_20,
            col_15,
        ]
        handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),\
                Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),\
                Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2])]
        labels= [
            'NDC',
            '2.0 °C',
            '1.5 °C',
        ]
        
        ax1.legend(
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
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad,
        )            
                
        f.savefig('./figures/pop_frac_birthyear_{}.png'.format(cohort_type),dpi=300)
#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac_birth_year_strj(
    ds_pop_frac_strj,
    df_GMT_strj,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=6
    y=12
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  8
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_hi = 'darkred'       # mean color for GMT trajectories above 2.5 at 2100
    col_med = 'darkgoldenrod'   # mean color for GMT trajectories above 1.5 to 2.5 at 2100     
    col_low = 'steelblue'       # mean color for GMT trajectories from min to 1.5 at 2100
    colors = {
        'low': col_low,
        'med': col_med,
        'hi': col_hi,
    }
    legend_lw=3.5 # legend line width
    x0 = 0.05 # bbox for legend
    y0 = 0.75
    xlen = 0.2
    ylen = 0.2    
    legend_entrypad = 0.5 # space between entries
    legend_entrylen = 0.75 # length per entry
    col_bis = 'black'     # color bisector
    style_bis = '--'     # style bisector
    lw_bis = 1     # lineweight bisector
    # time = year_range
    # xmin = np.min(time)
    # xmax = np.max(time)
    xmin = 1960
    xmax = 2020
    time = np.arange(xmin,xmax+1)

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Fraction unprecedented'
    ax2_xlab = 'Birth year'

    # plot both analysis types; exposed is only showing pop frac for exposed members of birth cohorts, all is showing full birth cohorts if they reached unprecedented level of exposure
    for cohort_type in ['exposed','all']:
        
        f,(ax1,ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(x,y),
        )

        # --------------------------------------------------------------------
        # plot unprecedented frac of total pop, ax1 for mean +/- std

        for step in ds_pop_frac_strj.GMT.values:
            
            ax1.plot(
                time,
                ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(GMT=step,birth_year=time).values,
                lw=lw_mean,
                color=colors[floater(df_GMT_strj.loc[2100,step])],
                zorder=1,
            )
            ax1.annotate(
                text=str(round(df_GMT_strj.loc[2100,step],2)), 
                xy=(time[-1], ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(GMT=step,birth_year=time).values[-1]),
                color='k',
                fontsize=impactyr_font,
            )

        ax1.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )

        # --------------------------------------------------------------------
        # plot unprecedented frac of total pop, all runs

        for run in ds_pop_frac_strj.runs:
            
            for step in ds_pop_frac_strj.GMT.values:
            
                ax2.plot(
                    time,
                    ds_pop_frac_strj['frac_unprec_{}'.format(cohort_type)].sel(runs=run,GMT=step,birth_year=time).values,
                    lw=lw_mean,
                    color=colors[floater(df_GMT_strj.loc[2100,step])],
                    zorder=1,
                )            

        ax2.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )
        ax2.set_xlabel(
            ax2_xlab, 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )    

        for i,ax in enumerate([ax1,ax2]):
            ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
            ax.set_xlim(xmin,xmax)
            # ax.xaxis.set_ticks(xticks_ts)
            # ax.xaxis.set_ticklabels(xtick_labels_ts)
            ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
            ax.tick_params(labelsize=tick_font,axis="y",direction="in")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.set_axisbelow(True) 
            if i < 1:
                ax.tick_params(labelbottom=False)
                
        # legend
        handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['hi']),\
                Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['med']),\
                Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['low'])]
        labels= [
            'GMT @ 2100 >= 2.5°C',
            '1.5°C <= GMT @ 2100 < 2.5°C',
            'GMT @ 2100 < 1.5°C',
        ]
        
        ax1.legend(
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
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad,
        )            
                
        f.savefig('./figures/pop_frac_birthyear_strj_{}.png'.format(cohort_type),dpi=300)

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac_birth_year_GMT_strj(
    ds_pop_frac_strj,
    ds_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
    year_range,
    flag_gmt,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=9
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    legend_lw=3.5 # legend line width
    x0 = 0.05 # bbox for legend
    y0 = 0.7
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
    xmin = 1.0
    xmax = 4.0
    
    # placment birth year cbar
    cb_by_x0 = 0.975
    cb_by_y0 = 0.125
    cb_by_xlen = 0.025
    cb_by_ylen = 0.75

    # placment emf cbar
    cb_emf_x0 = 0.60
    cb_emf_y0 = 0.05
    cb_emf_xlen = 0.225
    cb_emf_ylen = 0.015

    # identify colors for birth years
    cmap_by = plt.cm.get_cmap('viridis')
    cmap55 = cmap_by(0.01)
    cmap50 = cmap_by(0.05)   #light
    cmap45 = cmap_by(0.1)
    cmap40 = cmap_by(0.15)
    cmap35 = cmap_by(0.2)
    cmap30 = cmap_by(0.25)
    cmap25 = cmap_by(0.3)
    cmap20 = cmap_by(0.325)
    cmap10 = cmap_by(0.4)
    cmap5 = cmap_by(0.475)
    cmap0 = 'gray'
    cmap_5 = cmap_by(0.525)
    cmap_10 = cmap_by(0.6)
    cmap_20 = cmap_by(0.625)
    cmap_25 = cmap_by(0.7)
    cmap_30 = cmap_by(0.75)
    cmap_35 = cmap_by(0.8)
    cmap_40 = cmap_by(0.85)
    cmap_45 = cmap_by(0.9)
    cmap_50 = cmap_by(0.95)  #dark
    cmap_55 = cmap_by(0.99)

    colors_by = [
        cmap45,cmap35,cmap30,cmap25,cmap10,cmap5, # 6 dark colors for 1960 - 1965
        cmap_5,cmap_10,cmap_25,cmap_30,cmap_35,cmap_45, # 6 light colors for 1966-1970
    ]    
    
    line_colors = []
    for c in colors_by:
        for repeat in range(5):
            line_colors.append(c)
    line_colors.append(colors_by[-1])

    # declare list of colors for discrete colormap of colorbar for birth years
    cmap_list_by = mpl.colors.ListedColormap(colors_by,N=len(colors_by))
    cmap_list_by.set_under(cmap55)
    cmap_list_by.set_over(cmap_45)

    # colorbar args for birth years
    values_by = [1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
    tick_locs_by = [1960,1970,1980,1990,2000,2010,2020]
    tick_labels_by = list(str(n) for n in tick_locs_by)
    norm_by = mpl.colors.BoundaryNorm(values_by,cmap_list_by.N)    

    xticks = np.arange(1,4.1,0.5)
    xticklabels = [1.0, None, 2.0, None, 3.0, None, 4.0]

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Age emergence'
    ax2_xlab = 'GMT anomaly at 2100 [°C]'

    for cohort_type in ['exposed','all']:
        
        f,(ax1,ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(x,y),
        )
        
        # colorbar axes
        cbax_by = f.add_axes([
            cb_by_x0, 
            cb_by_y0, 
            cb_by_xlen, 
            cb_by_ylen
        ])         

        # --------------------------------------------------------------------
        # plot unprecedented frac of total pop and age emergence

        # strj
        for i,by in enumerate(np.arange(1960,2021,1)):
            
            ax1.plot(
                df_GMT_strj.loc[2100,:].values,
                ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by).values,
                lw=lw_mean,
                color=line_colors[i],
                zorder=1,
            )
            
            ax2.plot(
                df_GMT_strj.loc[2100,:].values,
                ds_age_emergence_strj['age_emergence'].\
                    sel(birth_year=by).\
                        weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                            mean(dim=('country','runs')).values,
                lw=lw_mean,
                color=line_colors[i],
                zorder=1,
            )
            
        # for 
        
        #     ax1.annotate(
        #         text=str(by), 
        #         xy=(df_GMT_strj.loc[2100,:].values[-1], ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by).values[-1]),
        #         xytext=((df_GMT_strj.loc[2100,:].values[-1], ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by).values[-1])),
        #         color='k',
        #         fontsize=impactyr_font,
        #         # zorder=5
        #     )        
            
            # ax2.annotate(
            #     text=str(by), 
            #     xy=(df_GMT_strj.loc[2100,:].values[-1], ds_age_emergence_strj['age_emergence'].\
            #         sel(birth_year=by).\
            #             weighted(ds_cohorts['weights'].sel(birth_year=by)).\
            #                 mean(dim=('country','runs')).values[-1]),
            #     xytext=((df_GMT_strj.loc[2100,:].values[-1], ds_age_emergence_strj['age_emergence'].\
            #         sel(birth_year=by).\
            #             weighted(ds_cohorts['weights'].sel(birth_year=by)).\
            #                 mean(dim=('country','runs')).values[-1])),
            #     color='k',
            #     fontsize=impactyr_font,
            # )        

        ax1.set_ylabel(
            ax1_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )
        ax2.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )    
        ax2.set_xlabel(
            ax2_xlab, 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )    

        for i,ax in enumerate([ax1,ax2]):
            ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
            ax.set_xlim(xmin,xmax)
            ax.set_xticks(
                xticks,
                labels=xticklabels
            )            
            # ax.xaxis.set_ticklabels(xtick_labels_ts)
            ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
            ax.tick_params(labelsize=tick_font,axis="y",direction="in")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.set_axisbelow(True) 
            if i < 1:
                ax.tick_params(labelbottom=False)  
                
        # birth year colorbar
        cb_by = mpl.colorbar.ColorbarBase(
            ax=cbax_by, 
            cmap=cmap_list_by,
            norm=norm_by,
            spacing='uniform',
            orientation='vertical',
            extend='neither',
            ticks=tick_locs_by,
            drawedges=False,
        )
        cb_by.set_label(
            'Birth year',
            size=16,
        )
        cb_by.ax.xaxis.set_label_position('bottom')
        cb_by.ax.tick_params(
            labelcolor='0',
            labelsize=16,
            color='0.5',
            length=3.5,
            width=0.4,
            # direction='right',
        ) 
        cb_by.ax.set_yticklabels(
            tick_labels_by,
            fontsize=10
            # rotation=45,
        )
        cb_by.outline.set_edgecolor('0.9')
        cb_by.outline.set_linewidth(0)                  
                
        f.savefig(
            './figures/pop_frac_birthyear_GMT_strj_annual_{}_{}.png'.format(cohort_type,flag_gmt),
            bbox_inches = "tight",
            dpi=300,
        )

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac_birth_year_GMT_strj_points(
    ds_pop_frac_strj,
    ds_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
    ds_age_emergence,
    ds_pop_frac_15,
    ds_pop_frac_20,
    ds_pop_frac_NDC,
    ind_15,
    ind_20,
    ind_NDC,    
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=9
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    markersize = 6 # marker size
    markerstyle = 'o' # marker style
    legend_lw=3.5 # legend line width
    x0 = 0.05 # bbox for legend
    y0 = 0.7
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
    xmin = 0.85
    xmax = 3.5
    
    gmt_indices = [ind_15,ind_20,ind_NDC]

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Age emergence'
    ax2_xlab = 'GMT at 2100'

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop, ax1 for mean +/- std

    # strj
    for cohort_type in ['exposed','all']:
        
        f,(ax1,ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(x,y),
        )
        
        for by in np.arange(1960,2021,10):
            
            ax1.plot(
                df_GMT_strj.loc[2100,:].values,
                ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by).values,
                lw=lw_mean,
                color=col_NDC,
                zorder=1,
            )
            ax1.annotate(
                text=str(by), 
                xy=(df_GMT_strj.loc[2100,:].values[-1], ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by).values[-1]),
                xytext=((df_GMT_strj.loc[2100,:].values[-1], ds_pop_frac_strj['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by).values[-1])),          
                color='k',
                fontsize=impactyr_font,
                # zorder=5
            )
            i = 0
            for ds,col in zip([ds_pop_frac_15,ds_pop_frac_20,ds_pop_frac_NDC],[col_15,col_20,col_NDC]):
                ax1.plot(
                    df_GMT_strj.loc[2100,gmt_indices[i]],
                    ds['mean_frac_unprec_{}'.format(cohort_type)].sel(birth_year=by),
                    color=col,
                    marker=markerstyle,
                    markersize=markersize,
                )
                i+=1
            
            ax2.plot(
                df_GMT_strj.loc[2100,:].values,
                ds_age_emergence_strj['age_emergence'].\
                    sel(birth_year=by).\
                        weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                            mean(dim=('country','runs')).values,
                lw=lw_mean,
                color=col_NDC,
                zorder=1,
            )
            
            ax2.annotate(
                text=str(by), 
                xy=(df_GMT_strj.loc[2100,:].values[-1], ds_age_emergence_strj['age_emergence'].\
                    sel(birth_year=by).\
                        weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                            mean(dim=('country','runs')).values[-1]),
                xytext=((df_GMT_strj.loc[2100,:].values[-1], ds_age_emergence_strj['age_emergence'].sel(birth_year=by).\
                    weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                        mean(dim=('country','runs')).values[-1])),
                color='k',
                fontsize=impactyr_font,
            )
            
            i = 0
            for scen,col in zip(['15','20','NDC'],[col_15,col_20,col_NDC]):
                ax2.plot(
                    df_GMT_strj.loc[2100,gmt_indices[i]],
                    ds_age_emergence['age_emergence_{}'.format(scen)].\
                        sel(birth_year=by).\
                            weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                                mean(dim=('country','runs')),
                    color=col,
                    marker=markerstyle,
                    markersize=markersize,
                )
                i+=1          

        ax1.set_ylabel(
            ax1_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )
        ax2.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )    
        ax2.set_xlabel(
            ax2_xlab, 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )    

        for i,ax in enumerate([ax1,ax2]):
            ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
            ax.set_xlim(xmin,xmax)
            # ax.xaxis.set_ticks(xticks_ts)
            # ax.xaxis.set_ticklabels(xtick_labels_ts)
            ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
            ax.tick_params(labelsize=tick_font,axis="y",direction="in")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
            ax.set_axisbelow(True) 
            if i < 1:
                ax.tick_params(labelbottom=False)  
                
        # legend
        handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=col_NDC),\
                Line2D([0],[0],marker='o',color='w',markerfacecolor=col_20),\
                Line2D([0],[0],marker='o',color='w',markerfacecolor=col_15)]
        labels= [
            'NDC',
            '2.0 °C',
            '1.5 °C',
        ]   
            
        ax1.legend(
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
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad,
        )               
                
        f.savefig('./figures/pop_frac_birthyear_GMT_strj_points_{}.png'.format(cohort_type),dpi=300)



#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac_birth_year_gcms(
    ds_pop_frac_NDC,
    ds_pop_frac_15,
    ds_pop_frac_20,
    runs,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=13
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    legend_lw=3.5 # legend line width
    x0 = 0.1 # bbox for legend
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
    xmax = 2020

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Fraction unprecedented'
    ax2_xlab = 'Birth year'

    f,(ax1,ax2,ax3,ax4) = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop, all runs
    for ax,gcm in zip((ax1,ax2,ax3,ax4),list(runs.keys())):
        
        for run in runs[gcm]:
            
            # NDC
            if run in ds_pop_frac_NDC['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_NDC['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_NDC,
                    zorder=1,
                )
                
            else:
                
                pass
            
            # 2.0 degrees
            if run in ds_pop_frac_20['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_20['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_20,
                    zorder=2,
                )
            
            else:
                
                pass
            
            # 1.5 degrees
            if run in ds_pop_frac_15['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_15['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_15,
                    zorder=3,
                )
                
            else:
                
                pass
            
        ax.set_title(
            gcm,
            loc='center',
            fontweight='bold',
        )

        ax.set_ylabel(
            ax2_ylab, 
            va='center', 
            rotation='vertical', 
            fontsize=axis_font, 
            labelpad=10,
        )
        ax4.set_xlabel(
            ax2_xlab, 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )    

    for i,ax in enumerate([ax1,ax2,ax3,ax4]):
        ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
        ax.set_xlim(xmin,xmax)
        # ax.xaxis.set_ticks(xticks_ts)
        # ax.xaxis.set_ticklabels(xtick_labels_ts)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_axisbelow(True) 
        if i < 3:
            ax.tick_params(labelbottom=False)
            
    # legend
    legendcols = [
        col_NDC,
        col_20,
        col_15,
    ]
    handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2])]
    labels= [
        'NDC',
        '2.0 °C',
        '1.5 °C',
    ]
    
    ax1.legend(
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
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )            
            
    f.savefig('./figures/pop_frac_birthyear_gcms.png',dpi=300)
        
#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pop_frac_birth_year_models(
    ds_pop_frac_NDC,
    ds_pop_frac_15,
    ds_pop_frac_20,
    runs,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=20
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    legend_lw=3.5 # legend line width
    x0 = 0.1 # bbox for legend
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
    xmax = 2020

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Fraction unprecedented'
    ax2_xlab = 'Birth year'

    f,axes = plt.subplots(
        nrows=len(list(runs.keys())),
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop, all runs
    for ax,mod in zip(axes.flatten(),list(runs.keys())):
        
        for run in runs[mod]:
            
            # NDC
            if run in ds_pop_frac_NDC['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_NDC['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_NDC,
                    zorder=1,
                )
                
            else:
                
                pass
            
            # 2.0 degrees
            if run in ds_pop_frac_20['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_20['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_20,
                    zorder=2,
                )
            
            else:
                
                pass
            
            # 1.5 degrees
            if run in ds_pop_frac_15['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_15['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_15,
                    zorder=3,
                )
                
            else:
                
                pass
            
        ax.set_title(
            mod,
            loc='center',
            fontweight='bold',
        )
        ax.set_title(
            len(runs[mod]),
            loc='right',
            fontweight='bold',
        )        

        # ax.set_ylabel(
        #     ax2_ylab, 
        #     va='center', 
        #     rotation='vertical', 
        #     fontsize=axis_font, 
        #     labelpad=10,
        # )
        
    # legend
    legendcols = [
        col_NDC,
        col_20,
        col_15,
    ]
    handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2])]
    labels= [
        'NDC',
        '2.0 °C',
        '1.5 °C',
    ]        

    for i,ax in enumerate(axes.flatten()):
        
        ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
        ax.set_xlim(xmin,xmax)
        ax.yaxis.set_ticks([0,0.25,0.5,0.75,1])
        # ax.xaxis.set_ticklabels(xtick_labels_ts)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_axisbelow(True) 
        
        if i == 0:
            
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
                handlelength=legend_entrylen, 
                handletextpad=legend_entrypad,
            )             
        
        if i < len(axes.flatten())-1:
            
            ax.tick_params(labelbottom=False)
            
        if i == len(axes.flatten())-1:
            
                ax.set_xlabel(
                    ax2_xlab, 
                    va='center', 
                    rotation='horizontal', 
                    fontsize=axis_font, 
                    labelpad=10,
                )               
            
    f.savefig('./figures/pop_frac_birthyear_mods.png',dpi=300)

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_age_emergence(
    da_age_emergence_NDC,
    da_age_emergence_15,
    da_age_emergence_20,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=7
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_NDC = 'darkred'       # unprec mean color
    col_NDC_fill = '#F08080'     # unprec fill color
    col_15 = 'steelblue'       # normal mean color
    col_15_fill = 'lightsteelblue'     # normal fill color
    col_20 = 'darkgoldenrod'   # rcp60 mean color
    col_20_fill = '#ffec80'     # rcp60 fill color
    legend_lw=3.5 # legend line width
    x0 = 0.85 # bbox for legend
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

    ax1_ylab = 'Age of emergence'
    ax1_xlab = 'Birth year'


    f,ax1 = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot age emergence

    # NDC
    ax1.plot(
        time,
        da_age_emergence_NDC.mean(dim=('country','runs')).values,
        lw=lw_mean,
        color=col_NDC,
        zorder=1,
    )
    ax1.fill_between(
        time,
        da_age_emergence_NDC.mean(dim=('country','runs')).values + da_age_emergence_NDC.std(dim=('country','runs')).values,
        da_age_emergence_NDC.mean(dim=('country','runs')).values - da_age_emergence_NDC.std(dim=('country','runs')).values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax1.plot(
        time,
        da_age_emergence_20.mean(dim=('country','runs')).values,
        lw=lw_mean,
        color=col_20,
        zorder=2,
    )
    ax1.fill_between(
        time,
        da_age_emergence_20.mean(dim=('country','runs')).values + da_age_emergence_20.std(dim=('country','runs')).values,
        da_age_emergence_20.mean(dim=('country','runs')).values - da_age_emergence_20.std(dim=('country','runs')).values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax1.plot(
        time,
        da_age_emergence_15.mean(dim=('country','runs')).values,
        lw=lw_mean,
        color=col_15,
        zorder=3,
    )
    ax1.fill_between(
        time,
        da_age_emergence_15.mean(dim=('country','runs')).values + da_age_emergence_15.std(dim=('country','runs')).values,
        da_age_emergence_15.mean(dim=('country','runs')).values - da_age_emergence_15.std(dim=('country','runs')).values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_15_fill,
        zorder=3,
    )

    ax1.set_ylabel(
        ax1_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    
    ax1.set_xlabel(
        ax1_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    


    for i,ax in enumerate([ax1]):
        ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
        ax.set_xlim(xmin,xmax)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_axisbelow(True) 
        
    # legend
    legendcols = [
        col_NDC,
        col_20,
        col_15,
    ]
    handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2])]
    labels= [
        'NDC',
        '2.0 °C',
        '1.5 °C',
    ]        
        
    ax1.legend(
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
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )               
            
    f.savefig('./figures/age_emergence.png',dpi=300)    
#%% ----------------------------------------------------------------
# stylized trajectories
def plot_stylized_trajectories(
    df_GMT_strj,
    d_isimip_meta,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=6
    y=12
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  11
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_hi = 'darkred'       # mean color for GMT trajectories above 2.5 at 2100
    col_med = 'darkgoldenrod'   # mean color for GMT trajectories above 1.5 to 2.5 at 2100     
    col_low = 'steelblue'       # mean color for GMT trajectories from min to 1.5 at 2100
    colors = {
        'rcp26': col_low,
        'rcp60': col_med,
        'rcp85': col_hi,
    }
    legend_lw=3.5 # legend line width
    x0 = 0.3 # bbox for legend
    y0 = 0.75
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
    xmax = 2113

    ax1_ylab = 'GMT [°C]'
    ax1_xlab = 'Time'

    gcms = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']
    rcps = ['rcp26', 'rcp60','rcp85']
    GMTs = {}
    for gcm in gcms:
        GMTs[gcm] = {}
        for rcp in rcps:
            i=0
            while i < 1:
                for k,v in list(d_isimip_meta.items()):
                    if v['gcm'] == gcm and v['rcp'] == rcp:
                        GMTs[gcm][rcp] = v['GMT'].rolling(window=10,center=True).mean()
                        i+=1
                    if i == 1:
                        break
            
                

    f,ax1 = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot GMTs

    # NDC
    df_GMT_strj.plot(
        ax=ax1,
        color='grey',
        zorder=1,
        lw=lw_mean,
    )
    for gcm in gcms:
        for rcp in rcps:  
            GMTs[gcm][rcp].plot(
                ax=ax1,
                color=colors[rcp],
                zorder=2,
                lw=lw_mean
            )  


    ax1.set_ylabel(
        ax1_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    
    ax1.set_xlabel(
        ax1_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    


    for i,ax in enumerate([ax1]):
        ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
        ax.set_xlim(xmin,xmax)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_axisbelow(True) 
        
    # legend
    handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['rcp85']),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['rcp60']),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['rcp26'])]
    labels= [
        'RCP 8.5',
        'RCP 6.0',
        'RCP 2.6',
    ]        
        
    ax1.legend(
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
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )               
            
    f.savefig('./figures/stylized_trajectories_smooth.png',dpi=300)    

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_age_emergence_strj(
    da_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
    year_range,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=12
    lw_mean=1
    lw_fill=0.1
    ub_alpha = 0.5
    title_font = 14
    tick_font = 12
    axis_font = 11
    legend_font = 14
    impactyr_font =  8
    col_grid = '0.8'     # color background grid
    style_grid = 'dashed'     # style background grid
    lw_grid = 0.5     # lineweight background grid
    col_hi = 'darkred'       # mean color for GMT trajectories above 2.5 at 2100
    col_med = 'darkgoldenrod'   # mean color for GMT trajectories above 1.5 to 2.5 at 2100     
    col_low = 'steelblue'       # mean color for GMT trajectories from min to 1.5 at 2100
    colors = {
        'low': col_low,
        'med': col_med,
        'hi': col_hi,
    }
    legend_lw=3.5 # legend line width
    x0 = 0.3 # bbox for legend
    y0 = 0.85
    xlen = 0.2
    ylen = 0.2  
    legend_entrypad = 0.5 # space between entries
    legend_entrylen = 0.75 # length per entry
    col_bis = 'black'     # color bisector
    style_bis = '--'     # style bisector
    lw_bis = 1     # lineweight bisector
    # time = year_range
    # xmin = np.min(time)
    # xmax = np.max(time)
    xmin = 1960
    xmax = 2020
    time = np.arange(xmin,xmax)

    ax1_ylab = 'Age of emergence'
    ax2_ylab = 'Age of emergence'
    ax1_xlab = 'Birth year'
    ax2_xlab = 'Birth year'


    f,(ax1,ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot age emergence

    # strj
    for step in da_age_emergence_strj.GMT.values:
        
        ax1.plot(
            time,
            da_age_emergence_strj['age_emergence'].\
                sel(GMT=step,birth_year=time).\
                    weighted(ds_cohorts['weights'].sel(birth_year=time)).\
                        mean(dim=('country','runs')).values,
            lw=lw_mean,
            color=colors[floater(df_GMT_strj.loc[2100,step])],
            zorder=1,
        )
        ax1.annotate(
            text=str(round(df_GMT_strj.loc[2100,step],2)), 
            xy=(time[-1], da_age_emergence_strj['age_emergence'].\
                sel(GMT=step,birth_year=time).\
                    weighted(ds_cohorts['weights'].sel(birth_year=time)).\
                        mean(dim=('country','runs')).values[-1]),
            # xytext=((time, ds_pop_frac_strj['mean_frac_all_unprec'].sel(birth_year=by).values[-1])),
            color='k',
            fontsize=impactyr_font,
            # zorder=5
        )

    ax1.set_ylabel(
        ax1_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    
    ax1.set_xlabel(
        ax1_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )   
    
    for run in da_age_emergence_strj.runs:
        
        for step in da_age_emergence_strj.GMT.values:
        
            ax2.plot(
                time,
                da_age_emergence_strj['age_emergence'].\
                    sel(runs=run,GMT=step,birth_year=time).\
                        weighted(ds_cohorts['weights'].sel(birth_year=time)).\
                            mean(dim=('country')).values,
                lw=lw_mean,
                color=colors[floater(df_GMT_strj.loc[2100,step])],
                zorder=1,
            )      

    ax2.set_ylabel(
        ax2_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    ax2.set_xlabel(
        ax2_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )      


    for i,ax in enumerate([ax1,ax2]):
        ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
        ax.set_xlim(xmin,xmax)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
        ax.set_axisbelow(True) 

    # legend
    handles = [Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['hi']),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['med']),\
               Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['low'])]
    labels= [
        'GMT @ 2100 >= 2.5°C',
        '1.5°C <= GMT @ 2100 < 2.5°C',
        'GMT @ 2100 < 1.5°C',
    ]   
        
    ax1.legend(
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
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad,
    )               
            
    f.savefig('./figures/age_emergence.png',dpi=300)    
# %%
