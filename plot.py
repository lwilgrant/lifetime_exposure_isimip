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

#%% --------------------------------------------------------------------
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

#%% --------------------------------------------------------------------
# test colors for plotting

def c(x):
    col = plt.cm.Reds(x)
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

#%% --------------------------------------------------------------------
# plot trends
def plot_trend(
    ds_e,
    flags,
    gdf_country_borders,
    df_GMT_strj,
    GMT_indices,
    grid_area,
):

    #=========================================================================================
    # plot trends for 1960-2113

    continents = {}
    continents['North America'] = [1,2,3,4,5,6,7]
    continents['South America'] = [9,10,11,12,13,14,15]
    continents['Europe'] = [16,17,18,19]
    continents['Asia'] = [28,29,30,31,32,33,34,35,37,38]
    continents['Africa'] = [21,22,23,24,25,26]
    continents['Australia'] = [39,40,41,42]    

    regions = gpd.read_file('./data/shapefiles/IPCC-WGI-reference-regions-v4.shp')
    gdf_continents = gpd.read_file('./data/shapefiles/IPCC_WGII_continental_regions.shp')
    # gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
    gdf_basin = gpd.read_file('./data/shapefiles/Major_Basins_of_the_World.shp')
    gdf_basin = gdf_basin.loc[:,['NAME','geometry']]    
    # merge basins with multiple entries in dataframe
    basins_grouped = []
    bc = {k:0 for k in gdf_basin['NAME']} # bc for basin counter
    for b_name in gdf_basin['NAME']:
        if len(gdf_basin.loc[gdf_basin['NAME']==b_name]) > 1:
            if bc[b_name]==0:
                gdf_b = gdf_basin.loc[gdf_basin['NAME']==b_name]
                basins_grouped.append(gdf_b.dissolve())
            bc[b_name]+=1
        else:
            basins_grouped.append(gdf_basin.loc[gdf_basin['NAME']==b_name])
    gdf_basin = pd.concat(basins_grouped).reset_index().loc[:,['NAME','geometry']]    
    
    gdf_ar6 = gpd.clip(regions,gdf_continents)
    gdf_ar6['keep'] = [0]*len(gdf_ar6.Acronym)
    for r in ds_e.region.data:
        gdf_ar6.loc[r,'keep'] = 1 
    gdf_ar6 = gdf_ar6[gdf_ar6.keep!=0].sort_index()
    gdf_ar6 = gdf_ar6.drop(columns=['keep'])
    gdf_country = gdf_country_borders.reset_index()
    
    # basins3D only has 220 basins, whereas the original gdf has 226 basins. conversion to raster misses small basins then
    # therefore need to find the missing ones to remove from gdf_basin
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    basins_3D = rm.mask_3D_geopandas(gdf_basin,lon,lat)
    small_basins = []
    for i,b in enumerate(basins_3D.region.data):
        if b!= basins_3D.region.data[i-1] + 1:
            missing_b = np.arange(basins_3D.region.data[i-1],b+1)[1:-1]
            for m in missing_b:
                small_basins.append(m)
    gdf_basin = gdf_basin.loc[~gdf_basin.index.isin(small_basins)]
    
    # separate versions of shapefiles and their data
    gdf_ar6_data = cp(gdf_ar6)
    gdf_country_data = cp(gdf_country)
    gdf_basin_data = cp(gdf_basin)
    
    for i,GMT in enumerate(np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):
        for y in ds_e.year.data:
            gdf_ar6_data['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_ar6'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values
            gdf_country_data['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_country'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values
            gdf_basin_data['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_basin'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values
            
    gdfs = {
        'ar6':gdf_ar6,
        'country':gdf_country,
        'basin':gdf_basin, 
    }
    gdfs_data = {
        'ar6':gdf_ar6_data,
        'country':gdf_country_data,
        'basin':gdf_basin_data,
    }

    trend_cols = []
    for i,GMT in enumerate(np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):
        for y in ds_e.year.data:    
            trend_cols.append('{}_{}'.format(GMT,y))
            
    samples = {}
    for k in gdfs.keys():
        samples[k] = gdfs_data[k].loc[:,trend_cols].values.flatten()
        
    # identify colors
    if flags['extr'] == 'floodedarea':
        cmap = 'RdBu'
    elif flags['extr'] == 'driedarea':
        cmap = 'RdBu_r'
    
    #========================================================
    # plot km^2 trends for each spatial scale
    
    for k in gdfs.keys():
        
        vmin = np.around(np.min(samples[k]),-2)
        vmax = np.around(np.max(samples[k]),-2)
        range = vmax - vmin
        interv = range/10
        norm = MidpointNormalize(
            vmin=vmin,
            vmax=vmax,
            midpoint=0,
        )

        # cbar location
        cb_x0 = 0.925
        cb_y0 = 0.2
        cb_xlen = 0.025
        cb_ylen = 0.6

        # cbar stuff
        col_cbticlbl = '0'   # colorbar color of tick labels
        col_cbtic = '0.5'   # colorbar color of ticks
        col_cbedg = '0.9'   # colorbar color of edge
        cb_ticlen = 3.5   # colorbar length of ticks
        cb_ticwid = 0.4   # colorbar thickness of ticks
        cb_edgthic = 0   # colorbar thickness of edges between colors

        # fonts
        title_font = 14
        cbtitle_font = 20
        tick_font = 12
        legend_font=12

        f,axes = plt.subplots(
            nrows=len(ds_e.year.data),
            ncols=len(GMT_indices),
            figsize=(20,6.5),
        )

        cbax = f.add_axes([cb_x0,cb_y0,cb_xlen,cb_ylen,])

        for row,y in zip(axes,ds_e.year.data):    
            for ax,GMT in zip(row,np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):   
                gdfs_data[k].plot(
                    ax=ax,
                    column='{}_{}'.format(GMT,y),
                    cmap=cmap,
                    norm=norm,
                    cax=cbax,
                )           
                gdfs[k].plot(
                    ax=ax,
                    color='none', 
                    edgecolor='black',
                    linewidth=0.25,
                )                           
                if y == 1960:
                    ax.set_title(
                        '{} °C @ 2100'.format(GMT),
                        loc='center',
                        fontweight='bold',
                        fontsize=10,
                    )     
                if GMT == np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')[0]:
                    ax.text(
                        -0.07, 0.55, 
                        '{}-{}'.format(y,y+80), 
                        va='bottom', 
                        ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                        fontweight='bold',
                        rotation='vertical', 
                        rotation_mode='anchor',
                        transform=ax.transAxes,
                    )
        for i,ax in enumerate(axes.flatten()):
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(
                letters[i],
                loc='left',
                fontweight='bold',
                fontsize=10
            )     
            
        cb = mpl.colorbar.ColorbarBase(
            ax=cbax, 
            cmap=cmap,
            norm=norm,
            orientation='vertical',
            spacing='uniform',
            drawedges=False,
        )
        # cb_lu.set_label('LU trends (°C/5-years)',
        cb.set_label( '{} trends [km^2/year]'.format(flags['extr']))
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(
            labelcolor=col_cbticlbl,
            labelsize=tick_font,
            color=col_cbtic,
            length=cb_ticlen,
            width=cb_ticwid,
            direction='out'
        )
        cb.outline.set_edgecolor(col_cbedg)
        cb.outline.set_linewidth(cb_edgthic)
        f.savefig('./figures/{}_{}_trends.png'.format(flags['extr'],k),dpi=800,bbox_inches='tight')
        plt.show()
    
    #========================================================
    # plot km^2 trends as frac of spatial units for each spatial scale   
         
    country_3D = rm.mask_3D_geopandas(gdf_country.reset_index(),lon,lat)
    ar6_3D = rm.mask_3D_geopandas(gdf_ar6.reset_index(),lon,lat)
    basin_3D = rm.mask_3D_geopandas(gdf_basin.reset_index(),lon,lat)
    
    # get sums of exposed area per ar6, country & basin, convert m^2 to km^2
    country_area = country_3D.weighted(grid_area/10**6).sum(dim=('lat','lon'))
    ar6_area = ar6_3D.weighted(grid_area/10**6).sum(dim=('lat','lon'))
    basin_area = basin_3D.weighted(grid_area/10**6).sum(dim=('lat','lon'))
    
    # separate versions of shapefiles and their data
    gdf_country_frac = cp(gdf_country)
    gdf_ar6_frac = cp(gdf_ar6)
    gdf_basin_frac = cp(gdf_basin)
    
    for i,GMT in enumerate(np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):
        for y in ds_e.year.data:
            gdf_country_frac['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_country'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values / country_area.values
            gdf_ar6_frac['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_ar6'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values / ar6_area.values
            gdf_basin_frac['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_basin'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values / basin_area.values
            
    gdfs_frac_data = {
        'ar6':gdf_ar6_frac,
        'country':gdf_country_frac,
        'basin':gdf_basin_frac,
    }
            
    # samples = {}
    # for k in gdfs.keys():
    #     samples[k] = gdfs_frac_data[k].loc[:,trend_cols].values.flatten()            
    
    cmap_whole = plt.cm.get_cmap(cmap)
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
    cmap_50 = cmap_whole(0.95)  # red
    cmap_55 = cmap_whole(0.99)

    colors = [
        cmap55,cmap50,cmap45,cmap40,cmap35,cmap30,cmap25,cmap20,cmap10,cmap5,
        cmap0,
        cmap_5,cmap_10,cmap_20,cmap_25,cmap_30,cmap_35,cmap_40,cmap_45,cmap_50,cmap_55,
    ]

    # declare list of colors for discrete colormap of colorbar
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))

    # colorbar args
    values_frac = [-0.001,-0.0009,-0.0008,-0.0007,-0.0006,-0.0005,-0.0004,-0.0003,-0.0002,-0.0001,-0.00001,\
        0.00001,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
    tick_locs_frac = [-0.001,-0.0008,-0.0006,-0.0004,-0.0002,0,0.0002,0.0004,0.0006,0.0008,0.001]
    tick_labels_frac = ['-0.001','-0.0008','-0.0006','-0.0004','-0.0002','0','0.0002','0.0004','0.0006','0.0008','0.001']
    norm_frac = mpl.colors.BoundaryNorm(values_frac,cmap_list_frac.N)        
    
    for k in gdfs.keys():
        
        # cbar location
        cb_x0 = 0.925
        cb_y0 = 0.2
        cb_xlen = 0.025
        cb_ylen = 0.6

        # cbar stuff
        col_cbticlbl = '0'   # colorbar color of tick labels
        col_cbtic = '0.5'   # colorbar color of ticks
        col_cbedg = '0.9'   # colorbar color of edge
        cb_ticlen = 3.5   # colorbar length of ticks
        cb_ticwid = 0.4   # colorbar thickness of ticks
        cb_edgthic = 0   # colorbar thickness of edges between colors

        # fonts
        title_font = 14
        cbtitle_font = 20
        tick_font = 12
        legend_font=12

        f,axes = plt.subplots(
            nrows=len(ds_e.year.data),
            ncols=len(GMT_indices),
            figsize=(20,6.5),
        )

        cbax = f.add_axes([cb_x0,cb_y0,cb_xlen,cb_ylen,])

        for row,y in zip(axes,ds_e.year.data):    
            for ax,GMT in zip(row,np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):   
                gdfs_frac_data[k].plot(
                    ax=ax,
                    column='{}_{}'.format(GMT,y),
                    cmap=cmap_list_frac,
                    norm=norm_frac,
                    cax=cbax,
                )           
                gdfs[k].plot(
                    ax=ax,
                    color='none', 
                    edgecolor='black',
                    linewidth=0.25,
                )                           
                if y == 1960:
                    ax.set_title(
                        '{} °C @ 2100'.format(GMT),
                        loc='center',
                        fontweight='bold',
                        fontsize=10,
                    )     
                if GMT == np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')[0]:
                    ax.text(
                        -0.07, 0.55, 
                        '{}-{}'.format(y,y+80), 
                        va='bottom', 
                        ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                        fontweight='bold',
                        rotation='vertical', 
                        rotation_mode='anchor',
                        transform=ax.transAxes,
                    )
        for i,ax in enumerate(axes.flatten()):
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(
                letters[i],
                loc='left',
                fontweight='bold',
                fontsize=10
            )     
            
        cb = mpl.colorbar.ColorbarBase(
            ax=cbax, 
            cmap=cmap_list_frac,
            norm=norm_frac,
            orientation='vertical',
            spacing='uniform',
            ticks=tick_locs_frac,
            drawedges=False,
        )
        # cb_lu.set_label('LU trends (°C/5-years)',
        cb.set_label( '{} trends [fraction/year]'.format(flags['extr']))
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(
            labelcolor=col_cbticlbl,
            labelsize=tick_font,
            color=col_cbtic,
            length=cb_ticlen,
            width=cb_ticwid,
            direction='out'
        )
        cb.ax.set_yticklabels(
            tick_labels_frac,
            # rotation=45    
        )        
        cb.outline.set_edgecolor(col_cbedg)
        cb.outline.set_linewidth(cb_edgthic)
        f.savefig('./figures/{}_{}_trends_frac.png'.format(flags['extr'],k),dpi=800,bbox_inches='tight')
        plt.show()        
        

#%% --------------------------------------------------------------------
# plot trends
def plot_gridscale_se_p(
    flags,
    gdf_country_borders,
    list_countries,
    sims_per_step,
    d_gs_spatial,
    df_GMT_strj,
    GMT_indices,
    grid_area,
):

    #=========================================================================================
    # plot trends for 1960-2113
    
    df_data = gdf_country_borders.loc[list_countries].reset_index()
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.mask_geopandas(df_data,lon,lat)
    max_lat = mask.lat.where(mask.notnull()).max().item()+1
    min_lat = mask.lat.where(mask.notnull()).min().item()-1
    max_lon = mask.lon.where(mask.notnull()).max().item()+1
    min_lon = mask.lon.where(mask.notnull()).min().item()-1
    mask = mask.sel(lat=slice(max_lat,min_lat),lon=slice(min_lon,max_lon),drop=True)
    da_data_region = mask.expand_dims(
        {'GMT':GMT_indices_plot, 'birth_year':sample_birth_years},
        axis=(0,1)
    ).copy()
    
    # da_cntrys = []
    for cntry in d_gs_spatial.keys():
        da_data = d_gs_spatial[cntry]['population_emergence']
        da_data = xr.where(da_data.isnull(),0,da_data)
        da_steps = []
        for step in GMT_indices_plot:
            da_data_step = da_data.loc[{
                'GMT':step,
                'run':sims_per_step[step],
            }].mean(dim='run')
            da_steps.append(da_data_step)
        da_data_mean = xr.concat(da_steps,dim='GMT')
        # da_cntrys.append(da_data_mean)
        da_data_region.loc[{
            'GMT':GMT_indices_plot,
            'birth_year':sample_birth_years,
            'lat':da_data_mean.lat.data,
            'lon':da_data_mean.lon.data,
        }] = da_data_mean
        
    da_data_region = da_data_region.where(da_data_region!=0)
    
    p = da_data_region.plot(
        # figsize=()
        transform=ccrs.PlateCarree(),
        row='GMT',
        col='birth_year',
        subplot_kws={"projection": ccrs.PlateCarree()}
    )
    for ax in p.axes.flat:
        df_data.plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
        )
        ax.add_feature(
            feature.OCEAN,
            facecolor='lightsteelblue'
        )        
        ax.coastlines(linewidth=0.25)
    
    # da_data_cntrys = xr.concat(da_cntrys,dim='country').assign_coords({'country':list(d_gs_spatial.keys())})
        
        
    regions = gpd.read_file('./data/shapefiles/IPCC-WGI-reference-regions-v4.shp')
    gdf_continents = gpd.read_file('./data/shapefiles/IPCC_WGII_continental_regions.shp')
    # gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
    gdf_basin = gpd.read_file('./data/shapefiles/Major_Basins_of_the_World.shp')
    gdf_basin = gdf_basin.loc[:,['NAME','geometry']]    
    # merge basins with multiple entries in dataframe
    basins_grouped = []
    bc = {k:0 for k in gdf_basin['NAME']} # bc for basin counter
    for b_name in gdf_basin['NAME']:
        if len(gdf_basin.loc[gdf_basin['NAME']==b_name]) > 1:
            if bc[b_name]==0:
                gdf_b = gdf_basin.loc[gdf_basin['NAME']==b_name]
                basins_grouped.append(gdf_b.dissolve())
            bc[b_name]+=1
        else:
            basins_grouped.append(gdf_basin.loc[gdf_basin['NAME']==b_name])
    gdf_basin = pd.concat(basins_grouped).reset_index().loc[:,['NAME','geometry']]    
    
    gdf_ar6 = gpd.clip(regions,gdf_continents)
    gdf_ar6['keep'] = [0]*len(gdf_ar6.Acronym)
    for r in ds_e.region.data:
        gdf_ar6.loc[r,'keep'] = 1 
    gdf_ar6 = gdf_ar6[gdf_ar6.keep!=0].sort_index()
    gdf_ar6 = gdf_ar6.drop(columns=['keep'])
    gdf_country = gdf_country_borders.reset_index()
    
    # basins3D only has 220 basins, whereas the original gdf has 226 basins. conversion to raster misses small basins then
    # therefore need to find the missing ones to remove from gdf_basin
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    basins_3D = rm.mask_3D_geopandas(gdf_basin,lon,lat)
    small_basins = []
    for i,b in enumerate(basins_3D.region.data):
        if b!= basins_3D.region.data[i-1] + 1:
            missing_b = np.arange(basins_3D.region.data[i-1],b+1)[1:-1]
            for m in missing_b:
                small_basins.append(m)
    gdf_basin = gdf_basin.loc[~gdf_basin.index.isin(small_basins)]
    
    # separate versions of shapefiles and their data
    gdf_ar6_data = cp(gdf_ar6)
    gdf_country_data = cp(gdf_country)
    gdf_basin_data = cp(gdf_basin)
    
    for i,GMT in enumerate(np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):
        for y in ds_e.year.data:
            gdf_ar6_data['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_ar6'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values
            gdf_country_data['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_country'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values
            gdf_basin_data['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_basin'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values
            
    gdfs = {
        'ar6':gdf_ar6,
        'country':gdf_country,
        'basin':gdf_basin, 
    }
    gdfs_data = {
        'ar6':gdf_ar6_data,
        'country':gdf_country_data,
        'basin':gdf_basin_data,
    }

    trend_cols = []
    for i,GMT in enumerate(np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):
        for y in ds_e.year.data:    
            trend_cols.append('{}_{}'.format(GMT,y))
            
    samples = {}
    for k in gdfs.keys():
        samples[k] = gdfs_data[k].loc[:,trend_cols].values.flatten()
        
    # identify colors
    if flags['extr'] == 'floodedarea':
        cmap = 'RdBu'
    elif flags['extr'] == 'driedarea':
        cmap = 'RdBu_r'
    
    #========================================================
    # plot km^2 trends for each spatial scale
    
    for k in gdfs.keys():
        
        vmin = np.around(np.min(samples[k]),-2)
        vmax = np.around(np.max(samples[k]),-2)
        range = vmax - vmin
        interv = range/10
        norm = MidpointNormalize(
            vmin=vmin,
            vmax=vmax,
            midpoint=0,
        )

        # cbar location
        cb_x0 = 0.925
        cb_y0 = 0.2
        cb_xlen = 0.025
        cb_ylen = 0.6

        # cbar stuff
        col_cbticlbl = '0'   # colorbar color of tick labels
        col_cbtic = '0.5'   # colorbar color of ticks
        col_cbedg = '0.9'   # colorbar color of edge
        cb_ticlen = 3.5   # colorbar length of ticks
        cb_ticwid = 0.4   # colorbar thickness of ticks
        cb_edgthic = 0   # colorbar thickness of edges between colors

        # fonts
        title_font = 14
        cbtitle_font = 20
        tick_font = 12
        legend_font=12

        f,axes = plt.subplots(
            nrows=len(ds_e.year.data),
            ncols=len(GMT_indices),
            figsize=(20,6.5),
        )

        cbax = f.add_axes([cb_x0,cb_y0,cb_xlen,cb_ylen,])

        for row,y in zip(axes,ds_e.year.data):    
            for ax,GMT in zip(row,np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):   
                gdfs_data[k].plot(
                    ax=ax,
                    column='{}_{}'.format(GMT,y),
                    cmap=cmap,
                    norm=norm,
                    cax=cbax,
                )           
                gdfs[k].plot(
                    ax=ax,
                    color='none', 
                    edgecolor='black',
                    linewidth=0.25,
                )                           
                if y == 1960:
                    ax.set_title(
                        '{} °C @ 2100'.format(GMT),
                        loc='center',
                        fontweight='bold',
                        fontsize=10,
                    )     
                if GMT == np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')[0]:
                    ax.text(
                        -0.07, 0.55, 
                        '{}-{}'.format(y,y+80), 
                        va='bottom', 
                        ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                        fontweight='bold',
                        rotation='vertical', 
                        rotation_mode='anchor',
                        transform=ax.transAxes,
                    )
        for i,ax in enumerate(axes.flatten()):
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(
                letters[i],
                loc='left',
                fontweight='bold',
                fontsize=10
            )     
            
        cb = mpl.colorbar.ColorbarBase(
            ax=cbax, 
            cmap=cmap,
            norm=norm,
            orientation='vertical',
            spacing='uniform',
            drawedges=False,
        )
        # cb_lu.set_label('LU trends (°C/5-years)',
        cb.set_label( '{} trends [km^2/year]'.format(flags['extr']))
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(
            labelcolor=col_cbticlbl,
            labelsize=tick_font,
            color=col_cbtic,
            length=cb_ticlen,
            width=cb_ticwid,
            direction='out'
        )
        cb.outline.set_edgecolor(col_cbedg)
        cb.outline.set_linewidth(cb_edgthic)
        f.savefig('./figures/{}_{}_trends.png'.format(flags['extr'],k),dpi=800,bbox_inches='tight')
        plt.show()
    
    #========================================================
    # plot km^2 trends as frac of spatial units for each spatial scale   
         
    country_3D = rm.mask_3D_geopandas(gdf_country.reset_index(),lon,lat)
    ar6_3D = rm.mask_3D_geopandas(gdf_ar6.reset_index(),lon,lat)
    basin_3D = rm.mask_3D_geopandas(gdf_basin.reset_index(),lon,lat)
    
    # get sums of exposed area per ar6, country & basin, convert m^2 to km^2
    country_area = country_3D.weighted(grid_area/10**6).sum(dim=('lat','lon'))
    ar6_area = ar6_3D.weighted(grid_area/10**6).sum(dim=('lat','lon'))
    basin_area = basin_3D.weighted(grid_area/10**6).sum(dim=('lat','lon'))
    
    # separate versions of shapefiles and their data
    gdf_country_frac = cp(gdf_country)
    gdf_ar6_frac = cp(gdf_ar6)
    gdf_basin_frac = cp(gdf_basin)
    
    for i,GMT in enumerate(np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):
        for y in ds_e.year.data:
            gdf_country_frac['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_country'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values / country_area.values
            gdf_ar6_frac['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_ar6'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values / ar6_area.values
            gdf_basin_frac['{}_{}'.format(GMT,y)] = ds_e['mean_exposure_trend_basin'].loc[{'GMT':int(GMT_indices[i]),'year':y}].values / basin_area.values
            
    gdfs_frac_data = {
        'ar6':gdf_ar6_frac,
        'country':gdf_country_frac,
        'basin':gdf_basin_frac,
    }
            
    # samples = {}
    # for k in gdfs.keys():
    #     samples[k] = gdfs_frac_data[k].loc[:,trend_cols].values.flatten()            
    
    cmap_whole = plt.cm.get_cmap(cmap)
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
    cmap_50 = cmap_whole(0.95)  # red
    cmap_55 = cmap_whole(0.99)

    colors = [
        cmap55,cmap50,cmap45,cmap40,cmap35,cmap30,cmap25,cmap20,cmap10,cmap5,
        cmap0,
        cmap_5,cmap_10,cmap_20,cmap_25,cmap_30,cmap_35,cmap_40,cmap_45,cmap_50,cmap_55,
    ]

    # declare list of colors for discrete colormap of colorbar
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))

    # colorbar args
    values_frac = [-0.001,-0.0009,-0.0008,-0.0007,-0.0006,-0.0005,-0.0004,-0.0003,-0.0002,-0.0001,-0.00001,\
        0.00001,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
    tick_locs_frac = [-0.001,-0.0008,-0.0006,-0.0004,-0.0002,0,0.0002,0.0004,0.0006,0.0008,0.001]
    tick_labels_frac = ['-0.001','-0.0008','-0.0006','-0.0004','-0.0002','0','0.0002','0.0004','0.0006','0.0008','0.001']
    norm_frac = mpl.colors.BoundaryNorm(values_frac,cmap_list_frac.N)        
    
    for k in gdfs.keys():
        
        # cbar location
        cb_x0 = 0.925
        cb_y0 = 0.2
        cb_xlen = 0.025
        cb_ylen = 0.6

        # cbar stuff
        col_cbticlbl = '0'   # colorbar color of tick labels
        col_cbtic = '0.5'   # colorbar color of ticks
        col_cbedg = '0.9'   # colorbar color of edge
        cb_ticlen = 3.5   # colorbar length of ticks
        cb_ticwid = 0.4   # colorbar thickness of ticks
        cb_edgthic = 0   # colorbar thickness of edges between colors

        # fonts
        title_font = 14
        cbtitle_font = 20
        tick_font = 12
        legend_font=12

        f,axes = plt.subplots(
            nrows=len(ds_e.year.data),
            ncols=len(GMT_indices),
            figsize=(20,6.5),
        )

        cbax = f.add_axes([cb_x0,cb_y0,cb_xlen,cb_ylen,])

        for row,y in zip(axes,ds_e.year.data):    
            for ax,GMT in zip(row,np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')):   
                gdfs_frac_data[k].plot(
                    ax=ax,
                    column='{}_{}'.format(GMT,y),
                    cmap=cmap_list_frac,
                    norm=norm_frac,
                    cax=cbax,
                )           
                gdfs[k].plot(
                    ax=ax,
                    color='none', 
                    edgecolor='black',
                    linewidth=0.25,
                )                           
                if y == 1960:
                    ax.set_title(
                        '{} °C @ 2100'.format(GMT),
                        loc='center',
                        fontweight='bold',
                        fontsize=10,
                    )     
                if GMT == np.round(df_GMT_strj.loc[2100,GMT_indices],1).values.astype('str')[0]:
                    ax.text(
                        -0.07, 0.55, 
                        '{}-{}'.format(y,y+80), 
                        va='bottom', 
                        ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                        fontweight='bold',
                        rotation='vertical', 
                        rotation_mode='anchor',
                        transform=ax.transAxes,
                    )
        for i,ax in enumerate(axes.flatten()):
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(
                letters[i],
                loc='left',
                fontweight='bold',
                fontsize=10
            )     
            
        cb = mpl.colorbar.ColorbarBase(
            ax=cbax, 
            cmap=cmap_list_frac,
            norm=norm_frac,
            orientation='vertical',
            spacing='uniform',
            ticks=tick_locs_frac,
            drawedges=False,
        )
        # cb_lu.set_label('LU trends (°C/5-years)',
        cb.set_label( '{} trends [fraction/year]'.format(flags['extr']))
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(
            labelcolor=col_cbticlbl,
            labelsize=tick_font,
            color=col_cbtic,
            length=cb_ticlen,
            width=cb_ticwid,
            direction='out'
        )
        cb.ax.set_yticklabels(
            tick_labels_frac,
            # rotation=45    
        )        
        cb.outline.set_edgecolor(col_cbedg)
        cb.outline.set_linewidth(cb_edgthic)
        f.savefig('./figures/{}_{}_regional_gs.png'.format(flags['extr'],k),dpi=800,bbox_inches='tight')
        plt.show()        
        
    
    
#%% ----------------------------------------------------------------
# plot timing and EMF of exceedence of pic-defined extreme
def spatial_emergence_plot(
    gdf_exposure_emergence_birth_year,
    flag_ext,
    flag_gmt,
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
    
    f.savefig('./figures/birth_year_emergence_{}_{}.png'.format(flag_ext,flag_gmt),dpi=300)


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
        ds_pop_frac_NDC['mean_unprec'].values / 1e6,
        lw=lw_mean,
        color=col_NDC,
        label='NDC',
        zorder=1,
    )
    ax1.fill_between(
        time,
        (ds_pop_frac_NDC['mean_unprec'].values / 1e6) + (ds_pop_frac_NDC['std_unprec'].values / 1e6),
        (ds_pop_frac_NDC['mean_unprec'].values / 1e6) - (ds_pop_frac_NDC['std_unprec'].values / 1e6),
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax1.plot(
        time,
        ds_pop_frac_20['mean_unprec'].values / 1e6,
        lw=lw_mean,
        color=col_20,
        label='2.0 °C',
        zorder=2,
    )
    ax1.fill_between(
        time,
        (ds_pop_frac_20['mean_unprec'].values / 1e6) + (ds_pop_frac_20['std_unprec'].values / 1e6),
        (ds_pop_frac_20['mean_unprec'].values / 1e6) - (ds_pop_frac_20['std_unprec'].values / 1e6),
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax1.plot(
        time,
        ds_pop_frac_15['mean_unprec'].values / 1e6,
        lw=lw_mean,
        color=col_15,
        label='1.5 °C',
        zorder=3,
    )
    ax1.fill_between(
        time,
        (ds_pop_frac_15['mean_unprec'].values / 1e6) + (ds_pop_frac_15['std_unprec'].values / 1e6),
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
        # plot unprecedented frac of pop, all run

        # NDC
        for run in ds_pop_frac_NDC.run:
            
            ax2.plot(
                time,
                ds_pop_frac_NDC['frac_unprec_{}'.format(cohort_type)].sel(run=run).values,
                lw=lw_mean,
                color=col_NDC,
                zorder=1,
            )
        # 2.0 degrees
        for run in ds_pop_frac_20.run:
            
            ax2.plot(
                time,
                ds_pop_frac_20['frac_unprec_{}'.format(cohort_type)].sel(run=run).values,
                lw=lw_mean,
                color=col_20,
                zorder=2,
            )
        # 1.5 degrees
        for run in ds_pop_frac_15.run:

            ax2.plot(
                time,
                ds_pop_frac_15['frac_unprec_{}'.format(cohort_type)].sel(run=run).values,
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
def plot_le_by_GMT_strj(
    ds_le,
    df_GMT_strj,
    ds_cohorts,
    flag_ext,
    flag_gmt,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=10
    y=6
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
    # time = year_range
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

    ax_ylab = 'Lifetime exposure'
    ax_xlab = 'GMT anomaly at 2100 [°C]'
        
    f,ax = plt.subplots(
        nrows=1,
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
    for i,by in enumerate(birth_years):
        
        ax.plot( # note that here we weight the age emergence for aggregation
            df_GMT_strj.loc[2100,:].values,
            ds_le['mmm'].\
                sel(birth_year=by).\
                    weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                        mean(dim='country').values,
            lw=lw_mean,
            color=line_colors[i],
            zorder=1,
        )

    ax.set_ylabel(
        ax_ylab, 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    ax.set_xlabel(
        ax_xlab, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    

    # for i,ax in enumerate([ax1,ax2]):
    # ax.set_title(letters[i],loc='left',fontsize=title_font,fontweight='bold')
    ax.set_xlim(xmin,xmax)
    ax.set_xticks(
        xticks,
        labels=xticklabels
    )
    ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
    ax.tick_params(labelsize=tick_font,axis="y",direction="in")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
    ax.xaxis.grid(color=col_grid, linestyle=style_grid, linewidth=lw_grid)
    ax.set_axisbelow(True) 
        # if i < 1:
        #     ax.tick_params(labelbottom=False)  
            
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
            
    # f.savefig(
    #     './figures/lifetime_exposure_birthyear_GMT_strj_annual_{}_{}.png'.format(flag_ext,flag_gmt),
    #     bbox_inches = "tight",
    #     dpi=300,
    # )

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
        # plot unprecedented frac of total pop, all run

        for run in ds_pop_frac_strj.run:
            
            for step in ds_pop_frac_strj.GMT.values:
            
                ax2.plot(
                    time,
                    ds_pop_frac_strj['frac_unprec_{}'.format(cohort_type)].sel(run=run,GMT=step,birth_year=time).values,
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
def plot_pf_ae_by_lines(
    ds_pf_strj,
    ds_ae_strj,
    df_GMT_strj,
    ds_cohorts,
    flags,
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
    # time = year_range
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
    
    # line version
    # for cohort_type in ['exposed','all']:
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
            ds_pf_strj['mean_frac_unprec_all_b_y0'].sel(birth_year=by).values,
            lw=lw_mean,
            color=line_colors[i],
            zorder=1,
        )
        
        ax2.plot( # note that here we weight the age emergence for aggregation
            df_GMT_strj.loc[2100,:].values,
            ds_ae_strj['age_emergence'].\
                sel(birth_year=by).\
                    weighted(ds_cohorts['by_y0_weights'].sel(birth_year=by)).\
                        mean(dim=('country','run')).values,
            lw=lw_mean,
            color=line_colors[i],
            zorder=1,
        )

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
        './figures/pf_ae_by_lines_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']),
        bbox_inches = "tight",
        dpi=300,
    )
    
#%% ----------------------------------------------------------------
# plotting pop frac
def plot_p_pf_ae_cs_heatmap(
    ds_pf_strj,
    ds_ae_strj,
    df_GMT_strj,
    ds_cohorts,
    flags,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    
    # --------------------------------------------------------------------
    # heatmap version
    
    # labels for GMT ticks
    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)
    
    # pop
    p = ds_pf_strj['mean_unprec_all_b_y0'] * 1000 / 10**6
    p = p.loc[{
        'birth_year':np.arange(1960,2021)
    }].plot(
        x='birth_year',
        y='GMT',
        add_colorbar=True,
        levels=10,
        cbar_kwargs={
            'label':'Millions unprecedented'
        }
    ) 
    p.axes.set_yticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    p.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p.axes.set_xlabel('Birth year')
    p.axes.figure.savefig('./figures/p_by_heatmap_cs_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))    
    plt.show()    
    
    # pop frac
    if flags['extr'] == 'heatwavedarea':
        levels = np.arange(0,1.01,0.1)
    else:
        levels = 10
    p2 = ds_pf_strj['mean_frac_unprec_all_b_y0'].loc[{
        'birth_year':np.arange(1960,2021)
    }].plot(
        x='birth_year',
        y='GMT',
        add_colorbar=True,
        levels=levels,
        cbar_kwargs={
            'label':'Population Fraction'
        }
    ) 
    p2.axes.set_yticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p2.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )
    p2.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p2.axes.set_xlabel('Birth year')
    p2.axes.figure.savefig('./figures/pf_by_heatmap_cs_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))    
    plt.show()
    
    # age emergence
    p3 = ds_ae_strj['age_emergence'].weighted(ds_cohorts['by_y0_weights']).mean(dim=('country','run')).plot(
        x='birth_year',
        y='GMT',
        levels=10,
        cbar_kwargs={
            'label':'Age Emergence'
        }        
    )
    p3.axes.set_yticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p3.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p3.axes.set_xlabel('Birth year')
    p3.axes.figure.savefig('./figures/ae_by_heatmap_cs_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))        

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_p_pf_ae_gs_heatmap(
    ds_pf_gs,
    ds_ae_gs,
    df_GMT_strj,
    da_gs_popdenom,
    flags,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    
    # --------------------------------------------------------------------
    # heatmap version
    GMT_indices_ticks=[6,11,15,19,24]
    # labels for GMT ticks
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)
    
    # pop
    p = ds_pf_gs['unprec'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]).astype('int')+1}].sum(dim='country')
    # summing across countries gives 0s for nan runs in some GMTs (should check country scale for this accidental feature, too)
    p = p.where(p!=0).mean(dim='run') / 10**6
    p = p.loc[{
        'birth_year':np.arange(1960,2021)
    }].plot(
        x='birth_year',
        y='GMT',
        add_colorbar=True,
        levels=10,
        cbar_kwargs={
            'label':'Millions unprecedented'
        }
    ) 
    p.axes.set_yticks(
        ticks=GMT_indices_ticks,
        labels=gmts2100
    )
    p.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    p.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p.axes.set_xlabel('Birth year')
    # p.axes.figure.savefig('./figures/p_by_heatmap_gs_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))    
    plt.show()    
    
    # pop frac
    if flags['extr'] == 'heatwavedarea':
        levels = np.arange(0,1.01,0.1)
    else:
        levels = 10
        
    p2 = ds_pf_gs['unprec'].loc[{
        'birth_year':np.arange(1960,2021),
        'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
    }].sum(dim='country')
    p2 = p2.where(p2!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country')
    p2 = p2.plot(
        x='birth_year',
        y='GMT',
        add_colorbar=True,
        levels=levels,
        cbar_kwargs={
            'label':'Population Fraction'
        }
    )
    p2.axes.set_yticks(
        ticks=GMT_indices_ticks,
        labels=gmts2100
    )
    p2.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    p2.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p2.axes.set_xlabel('Birth year')
    # p2.axes.figure.savefig('./figures/pf_by_heatmap_gs_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))    
    plt.show()
    
    # age emergence
    # p3 = ds_ae_gs['age_emergence_popweight'].weighted(ds_cohorts['by_y0_weights']).mean(dim=('country','run')).plot(
    p3 = ds_ae_gs['age_emergence_popweight'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}]
    p3 = p3.weighted(da_gs_popdenom / da_gs_popdenom.sum(dim='country')).mean(dim=('country','run')).plot(
        x='birth_year',
        y='GMT',
        levels=10,
        cbar_kwargs={
            'label':'Age Emergence'
        }        
    )
    p3.axes.set_yticks(
        ticks=GMT_indices_ticks,
        labels=gmts2100
    )
    p3.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p3.axes.set_xlabel('Birth year')
    # p3.axes.figure.savefig('./figures/ae_by_heatmap_gs_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))        

        
#%% ----------------------------------------------------------------
# plotting pop frac
def plot_pf_t_GMT_strj(
    ds_pf_strj,
    df_GMT_strj,
    flags,
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
    # time = year_range
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

    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)
    p = ds_pf_strj['mean_frac_unprec_all_t'].loc[{
        'time':np.arange(1960,2101)
    }].plot(
        x='time',
        y='GMT',
        levels=np.arange(0.6,1.01,0.05),
        cbar_kwargs={
            'label':'Population Fraction'
        }
    )
    p.axes.set_yticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p.axes.set_ylabel('GMT [°C]')
    p.axes.set_xlabel('Time')
    p.axes.figure.savefig('./figures/pf_time_{}_{}_{}.png'.format(flags['extr'],flags['gmt'],flags['rm']))


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
    # time = year_range
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
                            mean(dim=('country','run')).values,
                lw=lw_mean,
                color=col_NDC,
                zorder=1,
            )
            
            ax2.annotate(
                text=str(by), 
                xy=(df_GMT_strj.loc[2100,:].values[-1], ds_age_emergence_strj['age_emergence'].\
                    sel(birth_year=by).\
                        weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                            mean(dim=('country','run')).values[-1]),
                xytext=((df_GMT_strj.loc[2100,:].values[-1], ds_age_emergence_strj['age_emergence'].sel(birth_year=by).\
                    weighted(ds_cohorts['weights'].sel(birth_year=by)).\
                        mean(dim=('country','run')).values[-1])),
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
                                mean(dim=('country','run')),
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
    run,
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
    # plot unprecedented frac of total pop, all run
    for ax,gcm in zip((ax1,ax2,ax3,ax4),list(run.keys())):
        
        for run in run[gcm]:
            
            # NDC
            if run in ds_pop_frac_NDC['frac_all_unprec'].run.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_NDC['frac_all_unprec'].sel(run=run).values,
                    lw=lw_mean,
                    color=col_NDC,
                    zorder=1,
                )
                
            else:
                
                pass
            
            # 2.0 degrees
            if run in ds_pop_frac_20['frac_all_unprec'].run.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_20['frac_all_unprec'].sel(run=run).values,
                    lw=lw_mean,
                    color=col_20,
                    zorder=2,
                )
            
            else:
                
                pass
            
            # 1.5 degrees
            if run in ds_pop_frac_15['frac_all_unprec'].run.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_15['frac_all_unprec'].sel(run=run).values,
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
    run,
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
    xmin = 1960
    xmax = 2020

    ax1_ylab = 'Fraction unprecedented'
    ax2_ylab = 'Fraction unprecedented'
    ax2_xlab = 'Birth year'

    f,axes = plt.subplots(
        nrows=len(list(run.keys())),
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop, all run
    for ax,mod in zip(axes.flatten(),list(run.keys())):
        
        for run in run[mod]:
            
            # NDC
            if run in ds_pop_frac_NDC['frac_all_unprec'].run.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_NDC['frac_all_unprec'].sel(run=run).values,
                    lw=lw_mean,
                    color=col_NDC,
                    zorder=1,
                )
                
            else:
                
                pass
            
            # 2.0 degrees
            if run in ds_pop_frac_20['frac_all_unprec'].run.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_20['frac_all_unprec'].sel(run=run).values,
                    lw=lw_mean,
                    color=col_20,
                    zorder=2,
                )
            
            else:
                
                pass
            
            # 1.5 degrees
            if run in ds_pop_frac_15['frac_all_unprec'].run.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_15['frac_all_unprec'].sel(run=run).values,
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
            len(run[mod]),
            loc='right',
            fontweight='bold',
        )        
        
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
        da_age_emergence_NDC.mean(dim=('country','run')).values,
        lw=lw_mean,
        color=col_NDC,
        zorder=1,
    )
    ax1.fill_between(
        time,
        da_age_emergence_NDC.mean(dim=('country','run')).values + da_age_emergence_NDC.std(dim=('country','run')).values,
        da_age_emergence_NDC.mean(dim=('country','run')).values - da_age_emergence_NDC.std(dim=('country','run')).values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax1.plot(
        time,
        da_age_emergence_20.mean(dim=('country','run')).values,
        lw=lw_mean,
        color=col_20,
        zorder=2,
    )
    ax1.fill_between(
        time,
        da_age_emergence_20.mean(dim=('country','run')).values + da_age_emergence_20.std(dim=('country','run')).values,
        da_age_emergence_20.mean(dim=('country','run')).values - da_age_emergence_20.std(dim=('country','run')).values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax1.plot(
        time,
        da_age_emergence_15.mean(dim=('country','run')).values,
        lw=lw_mean,
        color=col_15,
        zorder=3,
    )
    ax1.fill_between(
        time,
        da_age_emergence_15.mean(dim=('country','run')).values + da_age_emergence_15.std(dim=('country','run')).values,
        da_age_emergence_15.mean(dim=('country','run')).values - da_age_emergence_15.std(dim=('country','run')).values,
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
    GMT_indices,
    d_isimip_meta,
):
    
    # --------------------------------------------------------------------
    # plotting utils
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    x=8
    y=6
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
    colors_rcp = {
        'rcp26': col_low,
        'rcp60': col_med,
        'rcp85': col_hi,
    }
    colors = {
        '4.0':'darkred',
        '3.0':'firebrick',
        'NDC':'darkorange',
        '2.0':'yellow',
        '1.5':'steelblue',
        'lb':'darkblue',
    }    
    legend_lw=3.5 # legend line width
    x0 = 0.15 # bbox for legend
    y0 = 0.25
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

    ax1_ylab = 'GMT [°C]'
    ax1_xlab = 'Time'

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
            
                

    f,ax1 = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot GMTs

    # plot all new scenarios in grey, then overlay marker scens
    df_GMT_strj.plot(
        ax=ax1,
        color='grey',
        zorder=1,
        lw=lw_mean,
    )
    
    # plot smooth gmts from RCPs
    for gcm in gcms:
        for rcp in rcps:  
            GMTs[gcm][rcp].plot(
                ax=ax1,
                color=colors_rcp[rcp],
                zorder=2,
                lw=lw_mean,
                style='--'
            )  
            
    # plot new ar6 marker scenarios in color
    df_GMT_lb = df_GMT_strj.loc[:,GMT_indices[0]]
    df_GMT_lb.plot(
        ax=ax1,
        color=colors['lb'],
        zorder=1,
        lw=lw_mean,
    )    
    df_GMT_15 = df_GMT_strj.loc[:,GMT_indices[1]]
    df_GMT_15.plot(
        ax=ax1,
        color=colors['1.5'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_20 = df_GMT_strj.loc[:,GMT_indices[2]]
    df_GMT_20.plot(
        ax=ax1,
        color=colors['2.0'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_NDC = df_GMT_strj.loc[:,GMT_indices[3]]
    df_GMT_NDC.plot(
        ax=ax1,
        color=colors['NDC'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_30 = df_GMT_strj.loc[:,GMT_indices[4]]
    df_GMT_30.plot(
        ax=ax1,
        color=colors['3.0'],
        zorder=1,
        lw=lw_mean,
    )
    df_GMT_40 = df_GMT_strj.loc[:,GMT_indices[5]]
    df_GMT_40.plot(
        ax=ax1,
        color=colors['4.0'],
        zorder=1,
        lw=2,
    )    
    df_GMT_40.loc[1960:2009].plot(
        ax=ax1,
        color='grey',
        zorder=3,
        lw=2,
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
    
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['lb']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['1.5']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['2.0']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['NDC']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['3.0']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=colors['4.0']),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=colors_rcp['rcp85']),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=colors_rcp['rcp60']),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=colors_rcp['rcp26'])         
    ]
    labels= [
        'lower bound',
        '1.5 °C',
        '2.0 °C',
        'NDC °C',
        '3.0 °C',
        '4.0 °C',
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
            
    f.savefig('./figures/GMT_trajectories_rcp_stylized.png',dpi=300)    
    

#%% ----------------------------------------------------------------
# plotting pop frac
def plot_age_emergence_strj(
    da_age_emergence_strj,
    df_GMT_strj,
    ds_cohorts,
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
                        mean(dim=('country','run')).values,
            lw=lw_mean,
            color=colors[floater(df_GMT_strj.loc[2100,step])],
            zorder=1,
        )
        ax1.annotate(
            text=str(round(df_GMT_strj.loc[2100,step],2)), 
            xy=(time[-1], da_age_emergence_strj['age_emergence'].\
                sel(GMT=step,birth_year=time).\
                    weighted(ds_cohorts['weights'].sel(birth_year=time)).\
                        mean(dim=('country','run')).values[-1]),
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
    
    for run in da_age_emergence_strj.run:
        
        for step in da_age_emergence_strj.GMT.values:
        
            ax2.plot(
                time,
                da_age_emergence_strj['age_emergence'].\
                    sel(run=run,GMT=step,birth_year=time).\
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
#%% ----------------------------------------------------------------
def wm_vs_gs_boxplots(
    d_isimip_meta,
    ds_ae_strj,
    ds_le,
    ds_pf_gs,
    ds_ae_gs,
    ds_le_gs,
):

    GMT_indices_plot = [0,10,19,28]
    # pop fraction dataset (sum of unprecedented exposure pixels' population per per country, run, GMT and birthyear)
    ds_pf_plot = xr.Dataset(
        data_vars={
            'unprec_exposed': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(sample_birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_exposed_fraction': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(sample_birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_all': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(sample_birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_all_fraction': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_indices_plot),len(sample_birth_years)),
                    fill_value=np.nan,
                ),
            )        
        },
        coords={
            'country': ('country', sample_countries),
            'birth_year': ('birth_year', sample_birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_indices_plot)
        }
    )

    # preparing weighted mean pop frac for our sample/comparison countries
    for i in list(d_isimip_meta.keys()):
        for step in GMT_indices_plot:
            if os.path.isfile('./data/pickles/{}/ds_exposure_cohort_aligned_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['extr'],i,step)):
                if d_isimip_meta[i]['GMT_strj_valid'][step]: # maybe an unecessary "if" since i probs didn't create it if the mapping wasn't right
                    with open('./data/pickles/{}/ds_exposure_cohort_aligned_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['extr'],i,step), 'rb') as f:
                        cohort_exposure_array = pk.load(f)
                    with open('./data/pickles/{}/da_exposure_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['gmt'],flags['extr'],i,step), 'rb') as f:
                        exposure_mask = pk.load(f)
                        birthyear_exposure_mask = xr.where(exposure_mask.sum(dim='time')>1,1,0)
                    for cntry in sample_countries:
                        ds_pf_plot['unprec_exposed'].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                        }] = cohort_exposure_array['exposure'].loc[{'birth_year':sample_birth_years,'country':cntry}].where(exposure_mask.loc[{'birth_year':sample_birth_years,'country':cntry}]==1).sum(dim='time')
                        ds_pf_plot['unprec_exposed_fraction'].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                        }] = cohort_exposure_array['exposure'].loc[{'birth_year':sample_birth_years,'country':cntry}].where(exposure_mask.loc[{'birth_year':sample_birth_years,'country':cntry}]==1).sum(dim='time') / ds_cohorts['population'].loc[{'birth_year':sample_birth_years,'country':cntry}]
                        ds_pf_plot['unprec_all'].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                        }] = ds_cohorts['population'].loc[{'birth_year':sample_birth_years,'country':cntry}].where(birthyear_exposure_mask.loc[{'birth_year':sample_birth_years,'country':cntry}]==1)
                        ds_pf_plot['unprec_all_fraction'].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                        }] = ds_cohorts['population'].loc[{'birth_year':sample_birth_years,'country':cntry}].where(birthyear_exposure_mask.loc[{'birth_year':sample_birth_years,'country':cntry}]==1) / ds_cohorts['population'].loc[{'birth_year':sample_birth_years,'country':cntry}]
                        
    # weighted mean age emergence
    da_ae_plot = ds_ae_strj['age_emergence'].loc[{'GMT':GMT_indices_plot,'country':sample_countries,'birth_year':sample_birth_years}]

    # weighted mean lifetime exposure
    da_le_plot = ds_le['lifetime_exposure'].loc[{'GMT':GMT_indices_plot,'country':sample_countries,'birth_year':sample_birth_years}]

    # gridscale datasets

    # pop frac, age emergence and lifetime exposure
    da_pf_gs_plot = ds_pf_gs['unprec_fraction'].loc[{'GMT':GMT_indices_plot,'birth_year':sample_birth_years}]
    da_ae_gs_plot = ds_ae_gs['age_emergence'].loc[{'GMT':GMT_indices_plot,'birth_year':sample_birth_years}]
    da_le_gs_plot = ds_le_gs['lifetime_exposure'].loc[{'GMT':GMT_indices_plot,'birth_year':sample_birth_years}]

    # lifetime exposure dataframes for plotting
    df_le_plot = da_le_plot.to_dataframe().drop(columns=['quantile']).reset_index() # so there are no series inside a cell (breaks up birth year and lifetime exposure to individual rows)
    df_le_plot['GMT'] = df_le_plot['GMT'].astype('str') # so hue is a string
    df_le_gs_plot = da_le_gs_plot.to_dataframe().reset_index() 
    df_le_gs_plot['GMT'] = df_le_gs_plot['GMT'].astype('str')
    
    # age emergence dataframes for plotting
    df_ae_plot = da_ae_plot.to_dataframe().reset_index()
    df_ae_plot['GMT'] = df_ae_plot['GMT'].astype('str')
    df_ae_gs_plot = da_ae_gs_plot.to_dataframe().reset_index()
    df_ae_gs_plot['GMT'] = df_ae_gs_plot['GMT'].astype('str')
    
    # pop frac dataframes for plotting
    df_pf_plot = ds_pf_plot['unprec_exposed_fraction'].to_dataframe().reset_index()
    df_pf_plot['GMT'] = df_pf_plot['GMT'].astype('str')
    df_pf_gs_plot = da_pf_gs_plot.to_dataframe().reset_index()
    df_pf_gs_plot['GMT'] = df_pf_gs_plot['GMT'].astype('str')

    import seaborn as sns

    # x=40
    # y=5

    # f,axes = plt.subplots(
    #     nrows=1, # variables
    #     ncols=4, # countries
    #     figsize=(x,y)
    # )
    x=20
    y=5
    f,ax = plt.subplots(
        nrows=1, # variables
        ncols=1, # countries
        figsize=(x,y)
    )
    colors = {
        '28':'darkred',
        '19':'firebrick',
        # 'NDC':'darkorange',
        '10':'yellow',
        # '1.5':'steelblue',
        '0':'darkblue',
    }

    # lifetime exposure
    ax = sns.boxplot(
        data=df_le_plot[df_le_plot['country']=='Russian Federation'],
        x='birth_year',
        y='lifetime_exposure',
        hue='GMT',
        palette=colors,
    )
    ax = sns.boxplot(
        data=df_le_gs_plot[df_le_gs_plot['country']=='Russian Federation'],
        x='birth_year',
        y='lifetime_exposure',
        hue='GMT',
        palette=colors,
    )
    
    # age emergence
    ax = sns.boxplot(
        data=df_ae_plot[df_ae_plot['country']=='Canada'],
        x='birth_year',
        y='age_emergence',
        hue='GMT',
        palette=colors,
    )
    ax = sns.boxplot(
        data=df_ae_gs_plot[df_ae_gs_plot['country']=='Canada'],
        x='birth_year',
        y='age_emergence',
        hue='GMT',
        palette=colors,
    )    
    
    # pop frac
    ax = sns.boxplot(
        data=df_pf_plot[df_pf_plot['country']=='Canada'],
        x='birth_year',
        y='unprec_exposed_fraction',
        hue='GMT',
        palette=colors,
    )
    ax = sns.boxplot(
        data=df_pf_gs_plot[df_pf_gs_plot['country']=='Canada'],
        x='birth_year',
        y='unprec_fraction',
        hue='GMT',
        palette=colors,
    )       

#%% ----------------------------------------------------------------
 
def gridscale_spatial(
    d_le_gs_spatial,
    analysis,
    countries_mask,
    countries_regions,
    flag_extr,
):
        
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
    cb_y0 = 0.2
    cb_xlen = 0.01
    cb_ylen = 0.8

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

    # colors = [cmap55,cmap45,cmap25,cmap5,cmap_25,cmap_35,cmap_45]
    # cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
    # cmap_list.set_over(cmap_55)
    # levels_i = np.arange(0,16,5)
    # levels_f = np.arange(20,51,10)
    # levels = np.concatenate((levels_i,levels_f))
    # norm = mpl.colors.BoundaryNorm(levels,cmap_list.N)   
        
    if analysis == 'lifetime_exposure':
        
        colors = [cmap55,cmap45,cmap25,cmap5,cmap_25,cmap_35,cmap_45]
        cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
        cmap_list.set_over(cmap_55)
        levels_i = np.arange(0,16,5)
        levels_f = np.arange(20,51,10)
        levels = np.concatenate((levels_i,levels_f))
        norm = mpl.colors.BoundaryNorm(levels,cmap_list.N)
        
    if analysis == 'age_emergence':
        
        colors = [cmap55,cmap45,cmap25,cmap5,cmap_25,cmap_35,cmap_45]
        cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
        cmap_list.set_over(cmap_55)
        levels = np.arange(10,81,10)
        norm = mpl.colors.BoundaryNorm(levels,cmap_list.N)
        
    if analysis == 'emergence_mask':
        
        colors = [cmap55,cmap45,cmap25,cmap_25,cmap_45]
        cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))
        # levels = np.arange(-0.5,1.6,1.0)
        levels = np.arange(0,1.1,0.2)
        norm = mpl.colors.BoundaryNorm(levels,cmap_list.N) 
        # ticks = [0,1]
                        

    f,axes = plt.subplots(
        nrows=len(GMT_indices_plot),
        ncols=len(birth_years_plot),
        figsize=(x,y),
        # subplot_kw={'projection':ccrs.Robinson()},
        subplot_kw={'projection':ccrs.PlateCarree()},
    )

    cbax = f.add_axes([cb_x0,cb_y0,cb_xlen,cb_ylen])

    i=0
    for row,step in zip(axes,GMT_indices_plot):
        
        for ax,by in zip(row,birth_years_plot):
        
            for cntry in sample_countries:
            
                plottable = d_le_gs_spatial[cntry][analysis].loc[{
                    'GMT':step,
                    'birth_year':by,
                }].mean(dim='run')
                
                if analysis == 'emergence_mask':
                    
                    da_cntry = xr.where( # grid cells/extent of sample country
                        countries_mask.where(countries_mask==countries_regions.map_keys(cntry),drop=True)==countries_regions.map_keys(cntry),
                        1,
                        0
                    )
                    plottable = plottable.where(da_cntry==1)                 

                plottable.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    levels=levels,
                    colors=colors,
                    cbar_ax=cbax,
                    add_labels=False,
                )
                
            ax.coastlines(linewidth=cstlin_lw)
            ax.add_feature(cr.feature.BORDERS,linewidth=cstlin_lw)
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
                    -0.07, 
                    0.55, 
                    step_labels[step], 
                    va='bottom', 
                    ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                    fontweight='bold',
                    fontsize=title_font,
                    rotation='vertical', 
                    rotation_mode='anchor',
                    transform=ax.transAxes
                )
    
    # if analysis != 'emergence_mask':    
        
    # colorbar
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
        
    # else:
        
    #     # colorbar
    #     cb = mpl.colorbar.ColorbarBase(
    #         ax=cbax, 
    #         cmap=cmap_list,
    #         norm=norm,
    #         spacing='proportional',
    #         orientation='vertical',
    #         extend='max',
    #         ticks=ticks,
    #         drawedges=False,
    #     )     
        # cb.ax.set_ticklabels([''])       
# cb = mpl.colorbar.ColorbarBase(ax=cbax, 
#                                cmap=cmap_list,
#                                norm=norm,
#                                spacing='uniform',
#                                orientation='horizontal',
#                                extend='neither',
#                                ticks=tick_locs,
#                                drawedges=False)
# cb.set_label('Correlation',
#              size=title_font)
# cb.ax.xaxis.set_label_position('top')
# cb.ax.tick_params(labelcolor=col_cbticlbl,
#                   labelsize=tick_font,
#                   color=col_cbtic,
#                   length=cb_ticlen,
#                   width=cb_ticwid,
#                   direction='out'); 
# cb.ax.set_xticklabels(tick_labels,
#                       rotation=45)
# cb.outline.set_edgecolor(col_cbedg)
# cb.outline.set_linewidth(cb_edgthic)        
    
    cb.ax.tick_params(
        labelsize=title_font,
    )
    
    plt.show()

    f.savefig('./figures/gridscale_{}_{}.png'.format(flag_extr,analysis),dpi=300)    
# %% ----------------------------------------------------------------
def boxplot_cs_vs_gs_p(
    ds_pf_strj,
    ds_pf_gs,
    df_GMT_strj,
    flags,
    sims_per_step,
    list_countries,
    ds_cohorts,
):
    
    # plot settings
    tick_font = 12
    axis_font = 12
    x=20
    y=5
    
    # country scale pop emerging
    da_p_plot = ds_pf_strj['unprec_country_b_y0'].loc[{
        'GMT':GMT_indices_plot,
        'country':list_countries,
        'birth_year':sample_birth_years,
    }] * 1000 / 10**6
    da_p_plot = da_p_plot.sum(dim='country')
    df_list = []
    for step in GMT_indices_plot:
        da_p_plot_step = da_p_plot.loc[{'run':sims_per_step[step],'GMT':step}]
        df_p_plot_step = da_p_plot_step.to_dataframe().reset_index()
        df_p_plot_step = df_p_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list.append(df_p_plot_step)
    df_p_plot = pd.concat(df_list)
    df_p_plot['unprec_country_b_y0'] = df_p_plot['unprec_country_b_y0'].fillna(0)

    # grid scale pop emerging
    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }] / 10**6
    df_list_gs = []
    da_p_gs_plot = da_p_gs_plot.sum(dim='country')
    for step in GMT_indices_plot:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}]
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe().reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['unprec'] = df_p_gs_plot['unprec'].fillna(0)
        
    f,(ax1,ax2) = plt.subplots(
        nrows=1, # variables
        ncols=2, # countries
        figsize=(x,y)
    )
    colors = dict(zip(np.round(df_GMT_strj.loc[2100,GMT_indices_plot],1).values.astype('str'),['steelblue','darkgoldenrod','darkred']))

    # pop
    sns.boxplot(
        data=df_p_plot,
        x='birth_year',
        y='unprec_country_b_y0',
        hue='GMT_label',
        palette=colors,
        ax=ax1,
    )
    sns.boxplot(
        data=df_p_gs_plot,
        x='birth_year',
        y='unprec',
        hue='GMT_label',
        palette=colors,
        ax=ax2,
    )        
    
    ax1.set_title(
        'country scale',
        loc='center',
        fontweight='bold'
    )
    ax2.set_title(
        'grid scale',
        loc='center',
        fontweight='bold'
    )    
    ax1.set_ylabel(
        'Millions living with unprecedented exposure', 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    ax2.set_ylabel(
        None, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    
    
    for ax in (ax1,ax2):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")     
        ax.set_ylim(
            0,
            np.around(ds_cohorts['by_population_y0'].sum(dim='country').sel(birth_year=2020).item()*1000 / 10**6,-1),
        )
        ax.set_xlabel(
            'Birth year', 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )    
        if ax == ax2:
            ax.yaxis.set_ticklabels([])
        
    f.savefig('./figures/testing/boxplots_p_cs_vs_gs_{}_selgmtruns.png'.format(flags['extr']))        
        
# %% ----------------------------------------------------------------
def boxplot_cs_vs_gs_pf(
    ds_cohorts,
    da_gs_popdenom,
    ds_pf_strj,
    ds_pf_gs,
    df_GMT_strj,
    flags,
    sims_per_step,
    list_countries,
):
    
    # plot settings
    tick_font = 12
    axis_font = 12
    x=20
    y=5
    
    # country scale pop emerging --------------------------------------------------------
    da_p_plot = ds_pf_strj['unprec_country_b_y0'].loc[{
        'GMT':GMT_indices_plot,
        'country':list_countries,
        'birth_year':sample_birth_years,
    }] * 1000
    df_list = []
    
    # this loop is done to make sure that for each GMT, we only include sims with valid mapping (and not unecessary nans that skew distribution and denominator when turned to 0)
    for step in GMT_indices_plot:
        da_pf_plot_step = da_p_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') \
            / (ds_cohorts['by_population_y0'].loc[{'country':list_countries,'birth_year':sample_birth_years}].sum(dim='country') * 1000)
        df_pf_plot_step = da_pf_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_plot_step = df_pf_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list.append(df_pf_plot_step)
    df_pf_plot = pd.concat(df_list)

    # grid scale pop emerging --------------------------------------------------------
    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }]
        
    df_list_gs = []
    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country')
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step = df_pf_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))        
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)
    
    
    f,(ax1,ax2) = plt.subplots(
        nrows=1, # variables
        ncols=2, # countries
        figsize=(x,y)
    )
    colors = dict(zip(np.round(df_GMT_strj.loc[2100,GMT_indices_plot],1).values.astype('str'),['steelblue','darkgoldenrod','darkred']))

    # pf
    sns.boxplot(
        data=df_pf_plot,
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        ax=ax1,
    )
    sns.boxplot(
        data=df_pf_gs_plot,
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        ax=ax2,
    )        
    
    ax1.set_title(
        'country scale',
        loc='center',
        fontweight='bold'
    )
    ax2.set_title(
        'grid scale',
        loc='center',
        fontweight='bold'
    )    
    ax1.set_ylabel(
        'Unprecedented population frac', 
        va='center', 
        rotation='vertical', 
        fontsize=axis_font, 
        labelpad=10,
    )
    ax2.set_ylabel(
        None, 
        va='center', 
        rotation='horizontal', 
        fontsize=axis_font, 
        labelpad=10,
    )    
    
    for ax in (ax1,ax2):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=tick_font,axis="x",direction="in", left="off",labelleft="on")
        ax.tick_params(labelsize=tick_font,axis="y",direction="in")    
        ax.set_ylim(
            0,
            1,
        )         
        ax.set_xlabel(
            'Birth year', 
            va='center', 
            rotation='horizontal', 
            fontsize=axis_font, 
            labelpad=10,
        )          
        if ax == ax2:
            ax.yaxis.set_ticklabels([])
        
    f.savefig('./figures/testing/boxplots_pf_cs_vs_gs_{}_selgmtruns.png'.format(flags['extr']))    
# %% ----------------------------------------------------------------
def scatter_pf_ae_cs(
    ds_ae_strj,
    ds_cohorts,
    df_GMT_strj,
    ds_pf_strj,
    flags,
):

    markersize=10
    # scatter plots of country-mean ae and global pf per run
    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)

    # for by in sample_birth_years:
    # for step in GMT_labels:
    by=2020
    step=24
            
    # initiate plotting axes
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10,7),
    )

    # age emergence in ax 1 and 2
    ds_plt = ds_ae_strj['age_emergence']                             
    ds_plt_gmt = ds_plt.loc[{'birth_year':by}]
    ds_plt_gmt = ds_plt_gmt.weighted(ds_cohorts['by_y0_weights'].loc[{'birth_year':by}]).mean(dim='country')
    p = ds_plt_gmt.to_dataframe().reset_index(level="run")
    x = p.index.values
    y = p['age_emergence'].values
    ax1.scatter(
        x,
        y,
        s=markersize,
    )
    ax1.plot(
        GMT_labels,
        ds_plt_gmt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax1.set_title(
        '{} birth cohort'.format(str(by)),
        loc='center',
        fontweight='bold',
    )
    ax1.set_ylabel(
        'age emergence', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )                                               
    ax1.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=None,
    )

    ds_plt_by = ds_plt.loc[{'GMT':step}]
    ds_plt_by = ds_plt_by.weighted(ds_cohorts['by_y0_weights']).mean(dim='country')
    p = ds_plt_by.to_dataframe().reset_index(level="run")
    x = p.index.values
    y = p['age_emergence'].values
    ax2.scatter(
        x,
        y,
        s=markersize,
    )
    ax2.plot(
        birth_years,
        ds_plt_by.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )   
    ax2.set_title(
        '{} @ 2100 [°C]'.format(str(np.round(df_GMT_strj.loc[2100,step],1))),
        loc='center',
        fontweight='bold',
    )       

    # pf in ax 3 and 4
    ds_plt = ds_pf_strj['frac_unprec_all_b_y0']                                                   
    ds_plt_gmt = ds_plt.loc[{'birth_year':by}]
    p = ds_plt_gmt.to_dataframe().reset_index(level="run")
    x = p.index.values
    y = p['frac_unprec_all_b_y0'].values
    ax3.scatter(
        x,
        y,
        s=markersize,
    )
    ax3.plot(
        GMT_labels,
        ds_plt_gmt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax3.set_ylabel(
        'population fraction', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )          
    ax3.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
    )                                           
    ax3.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100,
    )

    ds_plt_by = ds_plt.loc[{'GMT':step}]
    p = ds_plt_by.to_dataframe().reset_index(level="run")
    x = p.index.values
    y = p['frac_unprec_all_b_y0'].values
    ax4.scatter(
        x,
        y,
        s=markersize,
    )
    ax4.plot(
        birth_years,
        ds_plt_by.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )   
    ax4.set_xlabel(
        'Birth year', 
        va='center', 
        labelpad=10,
    )         
            
    # ax stuff
    for n,ax in enumerate((ax1,ax2,ax3,ax4)):
        ax.set_title(
            letters[n],
            loc='left',
            fontweight='bold',
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)                 
        if n < 2:
            ax.tick_params(labelbottom=False)        
    plt.show()
    f.savefig('./figures/testing/ae_pf_scatterplots_cs_{}_{}_{}.png'.format(step,by,flags['extr']),dpi=400)

# %% ----------------------------------------------------------------
def scatter_pf_ae_gs(
    ds_ae_gs,
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    flags,
):

    markersize=10

    # scatter plots of country-mean ae and global pf per run
    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)

    # for by in sample_birth_years:
    #     for step in GMT_labels:
    by=2020
    step = 24 # 3.5 deg
    

    # initiate plotting axes
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10,7),
    )

    # age emergence in ax 1 and 2
    da_plt = ds_ae_gs['age_emergence_popweight']                             
    da_plt_gmt = da_plt.loc[{'birth_year':by}]
    da_plt_gmt = da_plt_gmt.weighted((da_gs_popdenom.loc[{'birth_year':by}] / da_gs_popdenom.sum(dim='country')).loc[{'birth_year':by}]).mean(dim='country')
    p = da_plt_gmt.to_dataframe().reset_index(level="run")
    x = p.index.values
    y = p['age_emergence_popweight'].values
    ax1.scatter(
        x,
        y,
        s=markersize,
    )
    ax1.plot(
        GMT_labels,
        da_plt_gmt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax1.set_title(
        '{} birth cohort'.format(str(by)),
        loc='center',
        fontweight='bold',
    )
    ax1.set_ylabel(
        'age emergence', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )                                               
    ax1.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=None,
    )

    da_plt_by = da_plt.loc[{'GMT':step}]
    da_plt_by = da_plt_by.weighted((da_gs_popdenom.loc[{'birth_year':by}] / da_gs_popdenom.sum(dim='country')).loc[{'birth_year':by}]).mean(dim='country')
    p = da_plt_by.to_dataframe().reset_index(level="run")
    x = p.index.values
    y = p['age_emergence_popweight'].values
    ax2.scatter(
        x,
        y,
        s=markersize,
    )
    ax2.plot(
        birth_years,
        da_plt_by.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )   
    ax2.set_title(
        '{} @ 2100 [°C]'.format(str(np.round(df_GMT_strj.loc[2100,step],1))),
        loc='center',
        fontweight='bold',
    )       

    # pf in ax 3 and 4
    # da_plt = ds_pf_gs['frac_unprec_all_b_y0']
    da_plt = ds_pf_gs['unprec'].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
    da_plt_gmt = da_plt.loc[{'birth_year':by}].where(da_plt.loc[{'birth_year':by}]!=0)
    da_plt_gmt = da_plt_gmt / da_gs_popdenom.loc[{'birth_year':by}].sum(dim='country')
    p = da_plt_gmt.to_dataframe(name='pf').reset_index(level="run")
    x = p.index.values
    y = p['pf'].values
    ax3.scatter(
        x,
        y,
        s=markersize,
    )
    ax3.plot(
        GMT_labels,
        da_plt_gmt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax3.set_ylabel(
        'population fraction', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )          
    ax3.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
    )                                           
    ax3.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100,
    )

    # da_plt_by = da_plt.loc[{'GMT':step}]
    # p = da_plt_by.to_dataframe().reset_index(level="run")
    
    da_plt_by = da_plt.loc[{'GMT':step}].where(da_plt.loc[{'GMT':step}]!=0)
    da_plt_by = da_plt_by / da_gs_popdenom.sum(dim='country')      
    p = da_plt_by.to_dataframe(name='pf').reset_index(level="run")            
    
    x = p.index.values
    y = p['pf'].values
    ax4.scatter(
        x,
        y,
        s=markersize,
    )
    ax4.plot(
        birth_years,
        da_plt_by.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )   
    ax4.set_xlabel(
        'Birth year', 
        va='center', 
        labelpad=10,
    )         
            
    # ax stuff
    for n,ax in enumerate((ax1,ax2,ax3,ax4)):
        ax.set_title(
            letters[n],
            loc='left',
            fontweight='bold',
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)                 
        if n < 2:
            ax.tick_params(labelbottom=False)        
    plt.show()
    f.savefig('./figures/testing/ae_pf_scatterplots_gs_{}_{}_{}.png'.format(step,by,flags['extr']),dpi=400)


# %% ----------------------------------------------------------------

def lineplot_simcounts(
    d_isimip_meta,
    flags,
):
    # checking valid runs per GMT level:
    sim_counts = []
    for step in GMT_labels:
        c=0
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                c+=1
        sim_counts.append(c)
    da_sim_counts = xr.DataArray(
        data=sim_counts,
        coords={'GMT':GMT_labels},
    )
    p_sc = da_sim_counts.plot(marker='o')
    p_sc[0].axes.figure.savefig('./figures/testing/lineplot_simcounts_{}'.format(flags['extr']),dpi=400)            

# %% ----------------------------------------------------------------
            
def combined_plot_hw_p(
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
    tick_font = 12
    # cbar stuff
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    

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
        pos00.width,
        pos00.height*0.1
    ])
    pos01 = ax01.get_position()
    caxn1 = f.add_axes([
        pos01.x0-0.0775,
        pos00.y0+0.4,
        pos00.width,
        pos00.height*0.1
    ])    

    # pop frac heatmap ----------------------------------------------------------
    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)

    if flags['extr'] == 'heatwavedarea':
        levels = np.arange(0,1.01,0.1)
    else:
        levels = 10
        
    p2 = ds_pf_gs['unprec'].loc[{
        'birth_year':np.arange(1960,2021)
    }].sum(dim='country')
    p2 = p2.where(p2!=0).mean(dim='run') /  da_gs_popdenom.sum(dim='country')
    p2 = p2.plot(
        x='birth_year',
        y='GMT',
        ax=ax00,
        add_colorbar=True,
        levels=levels,
        cbar_kwargs={
            'label':'Population fraction',
            'cax':cax00,
            'orientation':'horizontal'
        }
    )
    p2.axes.set_yticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p2.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    p2.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p2.axes.set_xlabel('Birth year')
    cax00.xaxis.set_label_position('top')

    # add rectangle to 2020 series
    ax00.add_patch(Rectangle(
        (2020-0.5,0-0.5),1,29,
        facecolor='none',
        ec='k',
        lw=0.8
    ))
    # bracket connecting 2020 in heatmap to scatter plot panel

    # pop frac scatter ----------------------------------------------------------

    by=2020
    da_plt = ds_pf_gs['unprec'].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
    da_plt_gmt = da_plt.loc[{'birth_year':by}].where(da_plt.loc[{'birth_year':by}]!=0)
    da_plt_gmt = da_plt_gmt / da_gs_popdenom.loc[{'birth_year':by}].sum(dim='country')
    p = da_plt_gmt.to_dataframe(name='pf').reset_index(level="run")
    x = p.index.values
    y = p['pf'].values
    ax10.scatter(
        x,
        y,
        s=markersize,
        c='steelblue'
    )
    ax10.plot(
        GMT_labels,
        da_plt_gmt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax10.set_ylabel(
        'Population fraction', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )          
    ax10.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
    )                                           
    ax10.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100,
    )    
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)    

    handles = [
        Line2D([0],[0],linestyle='None',marker='o',color='steelblue'),
        Line2D([0],[0],marker='_',color='r'),
            
    ]
    labels= [
        'Simulations',
        'Mean',     
    ]    
    x0 = 0.55 # bbox for legend
    y0 = 0.25
    xlen = 0.2
    ylen = 0.2    
    legend_font = 10        
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

    # pop emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------

    cmap_whole = plt.cm.get_cmap('Reds')
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
    cmap_50 = cmap_whole(0.95)  # red
    cmap_55 = cmap_whole(0.99)

    colors = [
        cmap0, # gray for 0 unprecedented
        cmap45,cmap35,cmap25,cmap20,cmap5, # 100,000s
        cmap_5,cmap_20,cmap_25,cmap_35,cmap_45,cmap_55, # millions
    ]

    # declare list of colors for discrete colormap of colorbar
    cmap_list_p = mpl.colors.ListedColormap(colors,N=len(colors))

    # colorbar args
    values_p = [
        -0.1, 0.1,
        2*10**5,4*10**5,6*10**5,8*10**5,
        10**6,2*10**6,4*10**6,6*10**6,8*10**6,
        10**7,2*10**7,
    ]
    tick_locs_p = [
        0,1,
        2*10**5,4*10**5,6*10**5,8*10**5,
        10**6,2*10**6,4*10**6,6*10**6,8*10**6,
        10**7,2*10**7,        
    ]
    tick_labels_p = [
        '0',None,
        '200,000','400,000','600,000','800,000',
        '10e6','2x10e6','4x10e6','6x10e6','8x10e6',
        '10e7','2x10e7',
    ]
    norm_p = mpl.colors.BoundaryNorm(values_p,cmap_list_p.N)         

    gmt_indices_123 = [0,10,19]
    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':gmt_indices_123,
        'birth_year':by,
    }]
    
    # since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
    # so we only take sims or runs valid per GMT level and make sure nans are 0
    df_list_gs = []
    for step in gmt_indices_123:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe().reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['unprec'] = df_p_gs_plot['unprec'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_123):
        gdf_p['unprec']=df_p_gs_plot['unprec'][df_p_gs_plot['GMT']==step].values
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='unprec',
            cmap=cmap_list_p,
            norm=norm_p,
            cax=caxn1,
        )           
        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
        ) 
        
        cb = mpl.colorbar.ColorbarBase(
            ax=caxn1, 
            cmap=cmap_list_p,
            norm=norm_p,
            orientation='horizontal',
            spacing='uniform',
            drawedges=False,
            ticks=tick_locs_p,
        )

    cb.set_label('Unprecedented population')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )
    cb.ax.set_xticklabels(
        tick_labels_p,
        rotation=45,
    )    
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)                         

    f.savefig('./figures/combined_heatmap_scatter_mapsofp_{}.png'.format(flags['extr']),dpi=900)
    plt.show()        
    
# %% ----------------------------------------------------------------
            
def combined_plot_hw_pf_cs(
    df_GMT_strj,
    ds_pf_strj,
    ds_cohorts,
    gdf_country_borders,
    sims_per_step,
    flags,
):
    x=12
    y=10
    markersize=10
    # tick_font = 12
    # cbar stuff
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    

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
    pos01 = ax01.get_position()  
    
    i = 0 # letter indexing
    
    # colorbar stuff ------------------------------------------------------------
    
    cmap_whole = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,1.01,0.1)
    norm = mpl.colors.BoundaryNorm(levels,cmap_list_frac.N)   


    # pop frac heatmap ----------------------------------------------------------
    gmts2100 = np.round(df_GMT_strj.loc[2100,[0,5,10,15,20,25]].values,1)

    levels = np.arange(0,1.01,0.05)
      
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)
    p2 = ds_pf_strj['mean_frac_unprec_all_b_y0'].loc[{
        'birth_year':np.arange(1960,2021)
    }]
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
        ticks=[0,5,10,15,20,25],
        labels=gmts2100
    )
    p2.axes.set_xticks(
        ticks=np.arange(1960,2025,10),
    )    
    p2.axes.set_ylabel('GMT anomaly at 2100 [°C]')
    p2.axes.set_xlabel('Birth year')
    
    ax00.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    i+=1

    # add rectangle to 2020 series
    ax00.add_patch(Rectangle(
        (2020-0.5,0-0.5),1,29,
        facecolor='none',
        ec='gray',
        lw=0.8
    ))
    
    
    # bracket connecting 2020 in heatmap to scatter plot panel ------------------
    
    # vertical line
    x_h=2020
    y_h=-1
    # x_s=29.25
    # y_s=1.025
    x_s=0.995
    y_s=1.05    
    con = ConnectionPatch(
        xyA=(x_h,y_h),
        xyB=(x_s,y_s),
        coordsA=ax00.transData,
        # coordsB=ax10.transData,
        coordsB=ax10.transAxes,
        color='gray',
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
        color='gray'
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
        color='gray'
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
        color='gray'
    )
    ax10.add_artist(con)      

    # pop frac scatter ----------------------------------------------------------

    by=2020
    da_plt = ds_pf_strj['frac_unprec_all_b_y0'].loc[{'birth_year':by}]
    p = da_plt.to_dataframe(name='pf').reset_index(level="run")
    x = p.index.values
    y = p['pf'].values
    ax10.scatter(
        x,
        y,
        s=markersize,
        c='steelblue'
    )
    ax10.plot(
        GMT_labels,
        da_plt.mean(dim='run').values,
        marker='_',
        markersize=markersize/2,
        linestyle='',
        color='r'
    )
    ax10.set_ylabel(
        'Population fraction', 
        va='center', 
        rotation='vertical',
        labelpad=10,
    )          
    ax10.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
    )                                           
    ax10.set_xticks(
        ticks=[0,5,10,15,20,25],
        labels=gmts2100,
    )    
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)    

    handles = [
        Line2D([0],[0],linestyle='None',marker='o',color='steelblue'),
        Line2D([0],[0],marker='_',color='r'),
            
    ]
    labels= [
        'Simulations',
        'Mean',     
    ]    
    x0 = 0.55 # bbox for legend
    y0 = 0.25
    xlen = 0.2
    ylen = 0.2    
    legend_font = 10        
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
        letters[i],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    i+=1     

    # pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     

    gmt_indices_123 = [19,10,0]
    da_p_cs_plot = ds_pf_strj['unprec_country_b_y0'].loc[{
        'GMT':gmt_indices_123,
        'birth_year':by,
    }]
    
    # since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
    # so we only take sims or runs valid per GMT level and make sure nans are 0
    df_list_cs = []
    for step in gmt_indices_123:
        da_p_cs_plot_step = da_p_cs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).mean(dim='run')
        da_p_cs_plot_step = da_p_cs_plot_step / ds_cohorts['by_population_y0'].loc[{'birth_year':by}]
        df_p_cs_plot_step = da_p_cs_plot_step.to_dataframe(name='pf').reset_index()
        df_p_cs_plot_step = df_p_cs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_cs.append(df_p_cs_plot_step)
    df_p_cs_plot = pd.concat(df_list_cs)
    df_p_cs_plot['pf'] = df_p_cs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_123):
        gdf_p['pf']=df_p_cs_plot['pf'][df_p_cs_plot['GMT']==step].values
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='pf',
            cmap=cmap_list_frac,
            norm=norm,
            cax=cax00,
        )           

        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
        ) 
        
        ax.set_title(
            letters[i],
            loc='left',
            fontweight='bold',
            fontsize=10,
        )    
        i+=1
        
        ax.set_title(
            '{} °C'.format(str(np.round(df_GMT_strj.loc[2100,step],1))),
            loc='center',
            fontweight='bold',
            fontsize=10,       
        )
        
        # pointers connecting 2020, GMT step pixel in heatmap to map panels ------------------
        
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

    cb.set_label( 'Population fraction'.format(flags['extr']))
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        # labelsize=tick_font,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)   
    cax00.xaxis.set_label_position('top')                   

    f.savefig('./figures/combined_heatmap_scatter_mapsofpf_cs_{}.png'.format(flags['extr']),dpi=900)
    plt.show()            

# %% ----------------------------------------------------------------
            
def combined_plot_hw_pf_gs(
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
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
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
    
    p2.axes.set_ylabel('GMT warming by 2100 [°C]')
    p2.axes.set_xlabel('Birth year')
    
    ax00.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    i+=1

    # add rectangle to 2020 series
    ax00.add_patch(Rectangle(
        (2020-0.5,0-0.5),1,29,
        facecolor='none',
        ec='gray',
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
        color='gray',
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
        color='gray'
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
        color='gray'
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
        color='gray'
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
    )          
    ax10.set_xlabel(
        'GMT anomaly at 2100 [°C]', 
        va='center', 
        labelpad=10,
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
    
    ax10.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
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
        # ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))        
        
        ax.set_title(
            map_letters[step],
            loc='left',
            fontweight='bold',
            fontsize=10,
        )
        i+=1
        
        ax.set_title(
            # '{} °C'.format(str(np.round(df_GMT_strj.loc[2100,step],1))),
            '{} °C'.format(gmt_legend[step]),
            loc='center',
            fontweight='bold',
            fontsize=10,       
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

    cb.set_label('Population % living unprecedented exposure to heatwaves',fontsize=12)
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

    f.savefig('./figures/combined_heatmap_scatter_mapsofpf_gs_{}_panele.png'.format(flags['extr']),dpi=900)
    plt.show()            


#%% ----------------------------------------------------------------
def emergence_union_plot(
    grid_area,
    da_emergence_union,
    da_emergence_mean,
):
    x=18
    y=12
    markersize=10
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
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
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failure',
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
    gs0 = gridspec.GridSpec(3,2)
    gs0.update(wspace=0.25)

    # left side for 1960
    ax0 = f.add_subplot(gs0[0:1,0:1],projection=ccrs.Robinson()) # map of emergence union

    pos0 = ax0.get_position()
    cax = f.add_axes([
        pos0.x0+0.265,
        pos0.y0,
        pos0.width * 0.2,
        pos0.height*1
    ])

    gsn0 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[1:4,0:1],
        wspace=0,
        hspace=0,
    )
    ax00 = f.add_subplot(gsn0[0],projection=ccrs.Robinson())
    ax10 = f.add_subplot(gsn0[1],projection=ccrs.Robinson())
    ax20 = f.add_subplot(gsn0[2],projection=ccrs.Robinson()) 

    ax01 = f.add_subplot(gsn0[3],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gsn0[4],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gsn0[5],projection=ccrs.Robinson())    

    # right side for 2020
    ax1 = f.add_subplot(gs0[0:1,1:2],projection=ccrs.Robinson()) # map of emergence union
    gsn1 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[1:4,1:2],
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
    p_u3 = da_emergence_union.loc[{'GMT':19,'birth_year':1960}].where(mask.notnull())
    ax0.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax0.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u3.plot(
        ax=ax0,
        # cmap='Reds',
        cmap=cmap_list_union,
        levels=union_levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ax0.set_title(
        letters[i],
        loc='left',
        fontweight='bold',
    )
    ax0.set_title(
        # 'Emergence in 1960 cohort lifetimes',
        'All hazards',
        loc='center',
        fontweight='bold',
    )
    i+=1
    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p3 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':19,
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
        )
        ax.set_title(
            letters[i],
            loc='left',
            fontweight='bold',
        )
        i+=1
        
    # plot 2020
    # union_levels = np.arange(0.5,6.5,1)
    p_u3 = da_emergence_union.loc[{'GMT':19,'birth_year':2020}].where(mask.notnull())
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
        letters[i],
        loc='left',
        fontweight='bold',
    )
    ax1.set_title(
        # 'Emergence in 2020 cohort lifetimes',
        'All hazards',
        loc='center',
        fontweight='bold',
    )
    i+=1
    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p3 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':19,
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
        )
        ax.set_title(
            letters[i],
            loc='left',
            fontweight='bold',
        )
        i+=1    
        
        cb = mpl.colorbar.ColorbarBase(
            ax=cax, 
            cmap=cmap_list_union,
            norm=norm,
            orientation='vertical',
            spacing='uniform',
            ticks=np.arange(0,7).astype('int'),
            drawedges=False,
        )

        cb.set_label(
            'Number of emerged hazards',
            fontsize=12,
            labelpad=10,
        )
        cb.ax.yaxis.set_label_position('right')
        cb.ax.tick_params(
            labelcolor=col_cbticlbl,
            labelsize=12,
            color=col_cbtic,
            length=cb_ticlen,
            width=cb_ticwid,
            direction='out'
        )   
        
    # horizontal lines
    x_h1=0
    y_h1=1.2
    con = ConnectionPatch(
        xyA=(x_h1,y_h1),
        xyB=(x_h1+1,y_h1),
        coordsA=ax0.transAxes,
        coordsB=ax0.transAxes,
        color='gray'
    )
    ax0.add_artist(con)    
    ax0.annotate(
        '1960 birth cohort emergence',
        (0.23,1.25),
        xycoords='axes fraction',
        fontweight='bold',
        fontsize=12,
    )

    con = ConnectionPatch(
        xyA=(x_h1,y_h1),
        xyB=(x_h1+1,y_h1),
        coordsA=ax1.transAxes,
        coordsB=ax1.transAxes,
        color='gray'
    )
    ax1.add_artist(con)
    ax1.annotate(
        '2020 birth cohort emergence',
        (0.23,1.25),
        xycoords='axes fraction',
        fontweight='bold',
        fontsize=12,
    )

    # brace outliers
    for ax in (ax0,ax1):
        # left 
        con = ConnectionPatch(
            xyA=(x_h1,y_h1),
            xyB=(x_h1-0.025,y_h1-0.05),
            coordsA=ax.transAxes,
            coordsB=ax.transAxes,
            color='gray'
        )
        ax.add_artist(con)       

        # right 
        con = ConnectionPatch(
            xyA=(x_h1+1,y_h1),
            xyB=(x_h1+1+0.025,y_h1-0.05),
            coordsA=ax.transAxes,
            coordsB=ax.transAxes,
            color='gray'
        )
        ax.add_artist(con)
        
    f.savefig('./figures/emergence_locations.png',dpi=900)

#%% ----------------------------------------------------------------
# plotting pop and pop frac for grid scale across hazards
def plot_p_pf_gs_heatmap_combined(
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
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs_extr = pk.load(f)
        p = ds_pf_gs_extr['unprec'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}].sum(dim='country')
        p = p.where(p!=0).mean(dim='run') / 10**6
        # p.plot(x='birth_year',y='GMT')
        # plt.show()
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})
    
    # plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        ds_pf_gs_extrs.loc[{
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
                      
    f.savefig('./figures/gs_p_heatmap_combined.png',dpi=800)
    plt.show()    
    
    # --------------------------------------------------------------------
    # population fractions
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs_extr = pk.load(f)
        p = ds_pf_gs_extr['unprec'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}].sum(dim='country')
        p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
        # p.plot(x='birth_year',y='GMT')
        # plt.show()
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})
    
    # plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        ds_pf_gs_extrs.loc[{
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
 
    f.savefig('./figures/gs_pf_heatmap_combined.png',dpi=800)
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

def plot_sims_per_gmt(
    GMT_indices_plot,    
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
        'cropfailedarea': 'Crop failure',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }
    gmt_x_axis={
        str(GMT_indices_plot[0]):1.5,
        str(GMT_indices_plot[1]):2.5,
        str(GMT_indices_plot[2]):3.5,
    }
    # frame = {k:v.values for k,v in d_exposure_peryear_percountry.items()}
    frame = {
        'Simulations':[],
        'GMT':[],
        'Hazard':[],
    }
    # df_exposure = pd.DataFrame(frame,index=year_range)               
    for extr in extremes:
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)           
        sims_per_step = {}
        for step in GMT_indices_plot:
            sims_per_step[step] = 0
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step] +=1
            frame['Simulations'].append(sims_per_step[step])
            frame['GMT'].append(gmt_x_axis[str(step)])
            frame['Hazard'].append(extremes_labels[extr])
    df_sims_per_step = pd.DataFrame(frame)
    sns.barplot(
        data=df_sims_per_step,
        x='GMT',
        y='Simulations',
        hue='Hazard'
    )

# %% ======================================================================================================
def boxplot_combined_gs_pf(
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
    
    f.savefig('./figures/gs_boxplots_combined_pf.png',dpi=800,bbox_inches='tight')

# %% ======================================================================================================
def boxplot_heatwave_gs_pf(
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
        # loc=9,   #bbox: (x, y, width, height)
        ncol=1,
        fontsize=legend_font, 
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )              
    
    f.savefig('./figures/gs_boxplots_heatwave_pf.png',dpi=800,bbox_inches='tight')
    
    
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
def boxplot_combined_cs_p(
    flags,   
    gridscale_countries,
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
        }] * 1000 / 10**6
        
        # this loop is done to make sure that for each GMT, we only include sims with valid mapping (and not unecessary nans that skew distribution and denominator when turned to 0)
        for step in GMT_indices_plot:
            da_p_plot_step = da_p_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country')
            df_p_plot_step = da_p_plot_step.to_dataframe(name='p').reset_index()
            df_p_plot_step['GMT_label'] = df_p_plot_step['GMT'].map(gmt_legend)       
            df_p_plot_step['hazard'] = extr
            df_list_cs.append(df_p_plot_step)
            
    df_p_plot = pd.concat(df_list_cs)
    
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
            data=df_p_plot[df_p_plot['hazard']==extr],
            x='birth_year',
            y='p',
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
    
    f.savefig('./figures/cs_boxplots_combined_p.png',dpi=800,bbox_inches='tight')


# %% ======================================================================================================
def plot_ae_gs_heatmap_combined(
    da_gs_popdenom
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
    
    # loop through extremes and concat pop and pop frac
    list_extrs_ae = []
    for extr in extremes:
        with open('./data/pickles/{}/gridscale_aggregated_age_emergence_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_ae_gs = pk.load(f)
        ae = ds_ae_gs['age_emergence_popweight'].loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}]
        ae = ae.weighted(da_gs_popdenom / da_gs_popdenom.sum(dim='country')).mean(dim=('country','run'))
        list_extrs_ae.append(ae)
        
    ds_ae_gs_extrs = xr.concat(list_extrs_ae,dim='hazard').assign_coords({'hazard':extremes})       
    
    # plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        ds_ae_gs_extrs.loc[{
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
                      
    f.savefig('./figures/gs_ae_heatmap_combined.png',dpi=800)
    plt.show()         

# %% ======================================================================================================
def plot_ae_cs_heatmap_combined(
    ds_cohorts,
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
    
    # loop through extremes and concat pop and pop frac
    list_extrs_ae = []
    for extr in extremes:
        # age emergence           
        with open('./data/pickles/{}/age_emergence_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_ae_strj = pk.load(f)                
        ae = ds_ae_strj['age_emergence'].weighted(ds_cohorts['by_y0_weights']).mean(dim=('country','run'))
        ae = ae.loc[{'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int')}]
        list_extrs_ae.append(ae)
        
    ds_ae_cs_extrs = xr.concat(list_extrs_ae,dim='hazard').assign_coords({'hazard':extremes})       
    
    # plot
    x=16
    y=8
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        ds_ae_cs_extrs.loc[{
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
                      
    f.savefig('./figures/cs_ae_heatmap_combined.png',dpi=800)
    plt.show()         

# %% ======================================================================================================

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
    # cntry='Canada'
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
    # saint john coords
    # city_lat = 45.2733
    # city_lon = -66.0633
    # Tokyo coords  
    # city_lat = 35.6762
    # city_lon = 139.6503      
    # Delhi coords  
    # city_lat = 28.7041
    # city_lon = 77.1025
    # Cairo coords  
    # city_lat = 30.0444
    # city_lon = 31.2357
    # Lagos coords  
    # city_lat = 6.5244
    # city_lon = 3.3792

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
            linewidth=1
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
    ax.annotate(
        'Born in 1960',
        (1962,ax.get_ylim()[-1]+1),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )
    # ax.tick_params(colors='gray')
    ax.set_xlim(
        1960,
        end_year,
    )
    ax.set_ylim(
        0,
        # np.round(da_test_city.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max())+1,
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
    # sns.kdeplot(
    #     data=df_pic_city,
    #     y='lifetime_exposure',
    #     fill=True,
    #     color='grey',
    #     bw_adjust=5,
    #     ax=ax_pdf
    # )
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
        # xmax=0.4,
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
            # xmax=0.4,
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
    ax.tick_params(labelleft=False)    
    ax_pdf.tick_params(colors='gray')
    ax_pdf.spines['left'].set_color('gray')
    ax_pdf.spines['bottom'].set_color('gray')
        
    # ------------------------------------------------------------------       
    # 1990 time series
    ax2_l = 1960
    # ax2_b = np.round(da_test_city.loc[{'birth_year':1960,'GMT':GMT_indices_plot[-1]}].max()) *2
    ax2_b = da_pic_city_9999 *2
    ax2_w = 1990-1960+np.floor(df_life_expectancy_5.loc[1990,cntry])
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
    ax2.set_xlim(
        1960,
        end_year,
    )
    ax2.set_ylim(
        0,
        np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)  
    ax2.spines['left'].set_position(('data',1990))
    ax2.tick_params(labelleft=False)    
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
        # xmax=da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[0]}].time.\
        #     where(np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}])==np.round(da_pic_city_9999)).min()-3.1,
        xmax=end_year, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    ax2.annotate(
        'Born in 1990',
        (1992,ax2.get_ylim()[-1]-4),
        xycoords=ax2.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )

    # 1990 pdf
    ax2_pdf_l = end_year+5
    ax2_pdf_b = -2
    ax2_pdf_w = 20
    # ax2_pdf_h = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()+2)
    ax2_pdf_h = ax2.get_ylim()[-1]+2
    ax2_pdf = ax2.inset_axes(
        bounds=(ax2_pdf_l, ax2_pdf_b, ax2_pdf_w, ax2_pdf_h),
        transform=ax2.transData,
    )
    # sns.kdeplot(
    #     data=df_pic_city,
    #     y='lifetime_exposure',
    #     fill=True,
    #     color='grey',
    #     bw_adjust=5,
    #     ax=ax2_pdf
    # )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        # fill=True,
        color='lightgrey',
        discrete = True,
        ax=ax2_pdf
    )
    ax2_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        # xmax=0.4,
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
            # xmax=0.4, 
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
    ax3_l = 1960
    ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    ax3_w = 2020-1960+np.floor(df_life_expectancy_5.loc[2020,cntry])
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
    ax3.set_title(None)
    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.set_xlim(
        1960,
        end_year,
    )
    ax3.set_ylim(
        0,
        np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)  
    ax3.spines['left'].set_position(('data',2020))
    ax3.tick_params(labelleft=False)    
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
        # xmax=da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[0]}].time.\
        #     where(np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}])==np.round(da_pic_city_9999)).min()-3.1,
        xmax=end_year, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )

    # ax3.hlines(
    #     y=da_pic_city_9999, 
    #     xmin=2020, 
    #     # xmax=da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}].time.where(), 
    #     xmax=da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}].time.\
    #         where(np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[0]}])==np.round(da_pic_city_9999)).min()+1,
    #     colors='grey', 
    #     linewidth=1, 
    #     linestyle='--', 
    #     label='99.99%', 
    #     zorder=1
    # )
    ax3.annotate(
        'Born in 2020',
        (2022,ax3.get_ylim()[-1]-10),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='gray',
    )

    # 2020 pdf
    ax3_pdf_l = end_year+5
    ax3_pdf_b = -2
    ax3_pdf_w = 20
    # ax3_pdf_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max()+2)
    ax3_pdf_h = ax3.get_ylim()[-1]+2
    ax3_pdf = ax3.inset_axes(
        bounds=(ax3_pdf_l, ax3_pdf_b, ax3_pdf_w, ax3_pdf_h),
        transform=ax3.transData,
    )
    # sns.kdeplot(
    #     data=df_pic_city,
    #     y='lifetime_exposure',
    #     fill=True,
    #     color='grey',
    #     bw_adjust=5,
    #     cut=0,
    #     ax=ax3_pdf
    # )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        # fill=True,
        color='lightgrey',
        discrete = True,
        ax=ax3_pdf
    )
    ax3_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        # xmax=0.4, 
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
            # xmax=0.4, 
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
    y_f=y_i + 50
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
    y0 = -1.3
    xlen = 0.5
    ylen = 0.5

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 10
    legend_lw=1
    
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
        loc=3,
        ncol=1,
        fontsize=legend_font, 
        mode="expand", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    f.savefig('./figures/concept_{}_{}.png'.format(city_name,cntry),dpi=900,bbox_inches='tight')

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest')

# %%
