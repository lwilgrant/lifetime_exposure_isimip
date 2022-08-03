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
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate

#%% --------------------------------------------------------------------
# test colors for plotting

def c(x):
    col = plt.cm.OrRd(x)
    fig, ax = plt.subplots(figsize=(1,1))
    fig.set_facecolor(col)
    ax.axis("off")
    plt.show()
    
#%% ----------------------------------------------------------------
# load Wittgenstein Center population size per age cohort (source: http://dataexplorer.wittgensteincentre.org/wcde-v2/)
def load_wcde_data():

    wcde_years          = np.arange(1950,2105,5)       # hard coded from 'wcde_data_orig.xls' len is 31
    wcde_ages           = np.arange(2,102+5,5)         # hard coded from 'wcde_data_orig.xls' not that we assume +100 to be equal to 100-104, len is 21

    df_wcde             =  pd.read_excel('./data/Wittgenstein_Centre/wcde_data.xlsx',header=None)
    wcde_country_data   = df_wcde.values[:,4:]
    df_wcde_region      =  pd.read_excel(
        './data/Wittgenstein_Centre/wcde_data.xlsx', 
        'world regions', 
        header=None
    )
    wcde_region_data    = df_wcde_region.values[:,2:]

    return wcde_years, wcde_ages, wcde_country_data, wcde_region_data    
    
#%% ----------------------------------------------------------------
# interpolate cohortsize per country
def get_all_cohorts(
    wcde, 
    df_countries, 
    df_GMT_15,
): 

    # unpack loaded wcde values
    wcde = load_wcde_data() 
    wcde_years, wcde_ages, wcde_country_data, unused = wcde 
    # 31 year ranges, 21 age categories
    # target ages 
    # ages = np.arange(0,61)
    new_ages = np.arange(104,-1,-1)
    # new_years = np.concatenate((wcde_years, [np.max(df_GMT_15.index)]))
    # ages_og = np.concatenate(([np.min(ages)], wcde_ages))
    # years_og = np.concatenate((wcde_years, [np.max(df_GMT_15.index)]))
    # initialise dictionary to store cohort sizes dataframes per country with years as rows and ages as columns
    d_all_cohorts = {}

    for i,name in enumerate(df_countries.index):
    # i=25
    # name='Canada'    
    # extract population size per age cohort data from WCDE file and
    # linearly interpolate from 5-year WCDE blocks to pre-defined birth year
    # ! this gives slightly different values than MATLAB at some interpolation points inherent to the interpolation
        # wcde_country_data_reshape = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))).transpose()
        # wcde_per_country = np.hstack((
        #     np.expand_dims(wcde_country_data_reshape[:,0],axis=1)/2,
        #     np.expand_dims(wcde_country_data_reshape[:,0],axis=1)/2,
        #     wcde_country_data_reshape[:,1:]
        # ))
        # wcde_per_country = np.array(np.vstack([wcde_per_country,wcde_per_country[-1,:]]), dtype='float64')
        # [Xorig, Yorig] = np.meshgrid(np.concatenate(([np.min(ages)], wcde_ages)),np.concatenate((wcde_years, [np.max(df_GMT_15.index)]))) 
        # [Xnew, Ynew] = np.meshgrid(ages, np.array(df_GMT_15.index)) # prepare for 2D interpolation
        # wcde_country_data_raw = interpolate.griddata(
        #     (Xorig.ravel(),Yorig.ravel()),
        #     wcde_per_country.ravel(),
        #     (Xnew.ravel(),Ynew.ravel()),
        # )
        # wcde_country_data_interp = wcde_country_data_raw.reshape( len(df_GMT_15.index),len(ages))
        # d_all_cohorts[name] = pd.DataFrame(
        #     (wcde_country_data_interp), # cchanged from '/5' to '/25' because i'm imagining that the spread of cohort sizes needs to be accounted for by years and ages (and data matches better now to original size)
        #     columns=ages, 
        #     index=df_GMT_15.index
        # )
        
        wcde_country_data_reshape = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))).transpose()
        # below is using the original scheme but adding years at end too so that up to 104 has data (as original 102 was actually 100-104)
        # wcde_per_country = np.hstack((
        #     np.expand_dims(wcde_country_data_reshape[:,0],axis=1),
        #     wcde_country_data_reshape,
        #     np.expand_dims(wcde_country_data_reshape[:,-1],axis=1)
        # )) 
        # but now we test below by saying, since we're adding the 0-4 weights at front to interpolate to 0, we need to make cohort size adjustments so we don't inflate population size
        # therefore, we take half of the additional 0-4 in the front position 0, half of the original 0-4 at position 1, and use less of main data
        wcde_per_country = np.hstack((
            np.expand_dims(wcde_country_data_reshape[:,0],axis=1)/4,
            np.expand_dims(wcde_country_data_reshape[:,0],axis=1)*3/4,
            wcde_country_data_reshape[:,1:],
            np.expand_dims(wcde_country_data_reshape[:,-1],axis=1)
        ))         
        wcde_per_country = np.array(np.vstack([wcde_per_country,wcde_per_country[-1,:]]), dtype='float64')
        [Xorig, Yorig] = np.meshgrid(np.concatenate(([np.min(ages)], np.append(wcde_ages,107))),np.concatenate((wcde_years, [np.max(df_GMT_15.index)]))) 
        [Xnew, Ynew] = np.meshgrid(new_ages, np.array(df_GMT_15.index)) # prepare for 2D interpolation
        wcde_country_data_raw = interpolate.griddata(
            (Xorig.ravel(),Yorig.ravel()),
            wcde_per_country.ravel(),
            (Xnew.ravel(),Ynew.ravel()),
        )
        wcde_country_data_interp = wcde_country_data_raw.reshape( len(df_GMT_15.index),len(new_ages))
        d_all_cohorts[name] = pd.DataFrame(
            (wcde_country_data_interp /5), 
            columns=new_ages, 
            index=df_GMT_15.index
        )        
        
    # og_df = pd.DataFrame(
    #     wcde_per_country,
    #     columns=np.concatenate(([np.min(ages)], np.append(wcde_ages,107))),
    #     index=np.concatenate((wcde_years, [np.max(df_GMT_15.index)]))
    # ) 
    # allyears = np.arange(1950,2114)
    # og_df_empty = pd.DataFrame(
    #     columns=np.concatenate(([np.min(ages)], np.append(wcde_ages,107))),
    #     index=allyears
    # )
    # test_df = pd.concat([og_df,og_df_empty]).sort_index()
    # keeplist = []
    # for i in list(test_df.index.values):
    #     rows = test_df.loc[i,:]
    #     if len(rows.shape) > 1:
    #         keeplist.append(rows.dropna(axis=0))
    #     else:
    #         keeplist.append(pd.DataFrame(rows,columns=test_df.columns,index=[i]))
    # new_test_df = pd.concat(keeplist)
    # new_test_df = new_test_df[~new_test_df.index.duplicated()]
    # new_og_df = new_test_df.astype('float').interpolate(
    #     method='slinear', # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d
    #     limit_direction='both',
    #     fill_value='extrapolate',
    # )
    # new_og_df.loc[1960:,0:102].sum().sum() # after interpolating years in original data but maintaining cohorts, have better check against cohort extraction
    
    # df_concat = df_concat[~df_concat.index.duplicated(keep='last')]
    
    return d_all_cohorts  

#%% ----------------------------------------------------------------
# *improved function to compute extreme event exposure across a person's lifetime
def calc_cohort_emergence(
    da_exposure_cohort,
    # df_life_expectancy
    df_life_expectancy_5,
    col,
):

    country_list = []
    for country in da_exposure_cohort.country.values:
        
        # country='Canada'
        birthyear_list = []
        
        for i, birth_year in enumerate(df_life_expectancy_5.index):
            
            death_year = birth_year + np.floor(df_life_expectancy_5.loc[birth_year,country])
            
            time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
            ages = xr.DataArray(np.arange(0,len(time)),dims='age')
            # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
            # new data points from paired coords will be under new dim called ages, to be converted
            test = da_exposure_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
            # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
            test = test.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
            test = test.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
            yr = 1960+i
            test = test.assign_coords({'birth_year':yr}).drop_vars('ages')
            birthyear_list.append(test)
        
        cohort_exposure_test = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(cohort_exposure_test)
    da_exposure_cohort_all = xr.concat(country_list,dim='country')
    da_exposure_cohort_all_cumsum = da_exposure_cohort_all.cumsum(dim='time')
    ds_exposure_cohort = xr.Dataset(
        data_vars={
            'exposure': (da_exposure_cohort_all.dims,da_exposure_cohort_all.data),
            'exposure_cumulative': (da_exposure_cohort_all.dims,da_exposure_cohort_all.data)
        },
        coords={
            'country': ('country',da_exposure_cohort_all.country.data),
            'birth_year': ('birth_year',da_exposure_cohort_all.birth_year.data),
            'runs': ('runs',da_exposure_cohort_all.runs.data),
            'time': ('time',da_exposure_cohort_all.time.data),
        },
    )
     
    return exposure_birthyears_percountry  
    
#%% ----------------------------------------------------------------
# get timing and EMF of exceedence of pic-defined extreme
def calc_exposure_emergence(
    ds_exposure,
    ds_exposure_pic,
    gdf_country_borders,
):

    mmm_subset = [
        'mmm_RCP',
        'mmm_15',
        'mmm_20',
        'mmm_NDC',
    ]

    EMF_subset = [
        'mmm_EMF_RCP',
        'mmm_EMF_15',
        'mmm_EMF_20',
        'mmm_EMF_NDC',
    ]

    # get years where mmm exposures under different trajectories exceed pic 99.99%
    ds_exposure_emergence = ds_exposure[mmm_subset].where(ds_exposure[mmm_subset] > ds_exposure_pic.ext)
    ds_exposure_emergence_birth_year = ds_exposure_emergence.birth_year.where(ds_exposure_emergence.notnull()).min(dim='birth_year',skipna=True)
    
    # for same years, get EMF
    for var in EMF_subset:
    
        ds_exposure_emergence_birth_year[var] = ds_exposure[var].where(ds_exposure[var].birth_year==ds_exposure_emergence_birth_year[var.replace('_EMF','')]).min(dim='birth_year',skipna=True)
    
    # move emergene birth years and EMFs to gdf for plotting
    gdf_exposure_emergence_birth_year = gpd.GeoDataFrame(ds_exposure_emergence_birth_year.to_dataframe().join(gdf_country_borders))
    
    return gdf_exposure_emergence_birth_year

#%% ----------------------------------------------------------------
# get timing and EMF of exceedence of pic-defined extreme
def emergence_plot(
    gdf_exposure_emergence_birth_year,
):
    
    # plot
    f,axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(20,16)
    )

    # letters
    letters = ['a','b','c','d','e','f','g','h','i','j','k']

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

    colors_by = [
        cmap55,cmap45,cmap35,cmap25,cmap10,cmap5, # 6 dark colors for 1960 - 1990
        cmap_5,cmap_10,cmap_25,cmap_35,cmap_45,cmap_55, # 6 light colors for 1990-2020
    ]

    # declare list of colors for discrete colormap of colorbar for birth years
    cmap_list_by = mpl.colors.ListedColormap(colors_by,N=len(colors_by))

    # colorbar args for birth years
    values_by = [1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
    tick_locs_by = [1960,1970,1980,1990,2000,2010,2020]
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
    for trj in ['RCP','15','20','NDC']:
        data = np.append(data,gdf_exposure_emergence_birth_year.loc[:,'mmm_EMF_{}'.format(trj)].values)        
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
    for row,trj in zip(axes,['RCP','15','20','NDC']):
        
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
                    trj, 
                    va='bottom', 
                    ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                    fontweight='bold',
                    fontsize=16,
                    rotation='vertical', 
                    rotation_mode='anchor',
                    transform=ax.transAxes
                )            
            
            # plot associated EMF
            else:
                
                gdf_exposure_emergence_birth_year.plot(
                    column='mmm_EMF_{}'.format(trj),
                    ax=ax,
                    norm=norm_emf,
                    legend=False,
                    cmap=cmap_list_emf,
                    cax=cbax_emf,
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
            l += 1
            
                
    # birth year colorbar
    cb_by = mpl.colorbar.ColorbarBase(
        ax=cbax_by, 
        cmap=cmap_list_by,
        norm=norm_by,
        spacing='uniform',
        orientation='horizontal',
        extend='neither',
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
        'EMF of emergence',
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
