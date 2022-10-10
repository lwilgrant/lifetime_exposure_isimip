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
    new_ages = np.arange(104,-1,-1)

    d_all_cohorts = {}

    for i,name in enumerate(df_countries.index):

        wcde_country_data_reshape = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))).transpose()
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
    
    return d_all_cohorts  

#%% ----------------------------------------------------------------
# function to compute extreme event exposure per cohort per year, annual and annual cummulative
def calc_cohort_emergence(
    da_exposure_cohort,
    df_life_expectancy,
    year_start,
    year_end,
    year_ref,
):

    country_list = []
    for country in da_exposure_cohort.country.values:
        
        # country='Canada'
        birthyear_list = []
        
        # for i, birth_year in enumerate(df_life_expectancy.index):
        # instead of iterating from 1960-2020, we want to do 1960 to year_end (2113)
        for birth_year in np.arange(year_start,year_end+1):
            
            # use life expectancy information where available (until 2020)
            if birth_year <= year_ref:
                
                death_year = birth_year + np.floor(df_life_expectancy.loc[birth_year,country])
                time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
                ages = xr.DataArray(np.arange(0,len(time)),dims='age')
                # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
                # new data points from paired coords will be under new dim called ages, to be converted
                data = da_exposure_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
                # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
                data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)
            
            # after 2020, assume constant life expectancy    
            elif birth_year > year_ref and birth_year < year_end:
                
                death_year = birth_year + np.floor(df_life_expectancy.loc[year_ref,country]) #for years after 2020, just take 2020 life expectancy
                
                if death_year > year_end:
                    
                    death_year = year_end
                
                time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
                ages = xr.DataArray(np.arange(0,len(time)),dims='age')
                # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
                # new data points from paired coords will be under new dim called ages, to be converted
                data = da_exposure_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
                # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
                data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)
            
            # for 2113, use single year of exposure    
            elif birth_year == year_end:
                
                time = xr.DataArray([year_end],dims='age')
                ages = xr.DataArray([0],dims='age')
                # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
                # new data points from paired coords will be under new dim called ages, to be converted
                data = da_exposure_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
                # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
                data = data.rename({'age':'time'}).assign_coords({'time':[year_end]})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)                
        
        cohort_exposure_data = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(cohort_exposure_data)
        
    da_exposure_cohort_all = xr.concat(country_list,dim='country')
    da_exposure_cohort_all_cumsum = da_exposure_cohort_all.cumsum(dim='time')
    ds_exposure_cohort = xr.Dataset(
        data_vars={
            'exposure': (da_exposure_cohort_all.dims,da_exposure_cohort_all.data),
            'exposure_cumulative': (da_exposure_cohort_all_cumsum.dims,da_exposure_cohort_all_cumsum.data)
        },
        coords={
            'country': ('country',da_exposure_cohort_all.country.data),
            'birth_year': ('birth_year',da_exposure_cohort_all.birth_year.data),
            'runs': ('runs',da_exposure_cohort_all.runs.data),
            'time': ('time',da_exposure_cohort_all.time.data),
        },
    )
     
    return ds_exposure_cohort

#%% ----------------------------------------------------------------
# function to compute extreme event exposure per cohort per year, annual and annual cummulative
def calc_birthyear_cohortsizes(
    da_cohort,
    df_life_expectancy,
    year_start,
    year_end,
    year_ref,
):

    country_list = []
    for country in da_cohort.country.values:
        
        # country='Canada'
        birthyear_list = []
        
        # for i, birth_year in enumerate(df_life_expectancy.index):
        # instead of iterating from 1960-2020, we want to do 1960 to year_end (2113)
        for birth_year in np.arange(year_start,year_end+1):
            
            # use life expectancy information where available (until 2020)
            if birth_year <= year_ref:
                
                death_year = birth_year + np.floor(df_life_expectancy.loc[birth_year,country])
                time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
                ages = xr.DataArray(np.arange(0,len(time)),dims='age')
                # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
                # new data points from paired coords will be under new dim called ages, to be converted
                data = da_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
                # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
                data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)
            
            # after 2020, assume constant life expectancy    
            elif birth_year > year_ref and birth_year < year_end:
                
                death_year = birth_year + np.floor(df_life_expectancy.loc[year_ref,country]) #for years after 2020, just take 2020 life expectancy
                
                if death_year > year_end:
                    
                    death_year = year_end
                
                time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
                ages = xr.DataArray(np.arange(0,len(time)),dims='age')
                # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
                # new data points from paired coords will be under new dim called ages, to be converted
                data = da_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
                # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
                data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)
            
            # for 2113, use single year of pop
            elif birth_year == year_end:
                
                time = xr.DataArray([year_end],dims='age')
                ages = xr.DataArray([0],dims='age')
                # for birth year 1960, we want paired coord selections of (1960, age 0), (1961, age 1), (1962, age 2) & (1963, age 3) ... until death year/age
                # new data points from paired coords will be under new dim called ages, to be converted
                data = da_cohort.sel(country=country,time=time,ages=ages)#.cumsum(dim='age') # cumulative sum for each year to show progress of exposure
                # but do we want the above cum sum? maybe we rather want a copy of the final data array with this cum sum for checking against 99% from pic? Removed it for this reason
                data = data.rename({'age':'time'}).assign_coords({'time':[year_end]})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)                
        
        cohort_data = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(cohort_data)
        
    da_cohort_all = xr.concat(country_list,dim='country')
    da_cohort_all= da_cohort_all.sum(dim='time')
    ds_cohort_all = xr.Dataset(
        data_vars={
            'population': (da_cohort_all.dims,da_cohort_all.data),
        },
        coords={
            'country': ('country',da_cohort_all.country.data),
            'birth_year': ('birth_year',da_cohort_all.birth_year.data),
        },
    )
     
    return ds_cohort_all

#%% ----------------------------------------------------------------
# function to generate mask of unprecedented timesteps per birth year and age of emergence
def exposure_pic_masking(
    ds_exposure_mask,
    ds_exposure_pic,
):
    # generate exposure mask for timesteps after reaching pic extreme to find age of emergence
    ds_exposure_pic['ext'] = ds_exposure_pic['ext'].where(ds_exposure_pic['ext']>0)
    da_exposure_mask = xr.where(ds_exposure_mask['exposure_cumulative'] >= ds_exposure_pic['ext'],1,0)
    da_age_emergence = da_exposure_mask * (da_exposure_mask.time - da_exposure_mask.birth_year)
    da_age_emergence = da_age_emergence.where(da_age_emergence!=0).min(dim='time',skipna=True)
    
    # adjust exposure mask; for any birth cohorts that crossed extreme, keep 1s at all time steps and 0 for other birth cohorts
    da_exposure_mask = da_exposure_mask.where(da_exposure_mask==1).bfill(dim='time')
    da_exposure_mask = xr.where(da_exposure_mask==1,1,0)
    
    return da_exposure_mask,da_age_emergence

#%% ----------------------------------------------------------------
# function to compute extreme event exposure across a person's lifetime
def calc_unprec_exposure(
    ds_exposure_cohort,
    da_exposure_mask,
    d_all_cohorts,
    year_range,
    df_countries,
    df_life_expectancy,
    year_start,
    year_end,
    year_ref,
):

    # new empty dataset with variables for population experiencing unprecedented exposure or not
    ds_pop_frac = xr.Dataset(
        data_vars={
            'unprec': (['runs','birth_year'], np.empty((len(ds_exposure_cohort.runs.data),len(ds_exposure_cohort.birth_year.data)))),
            'normal': (['runs','birth_year'], np.empty((len(ds_exposure_cohort.runs.data),len(ds_exposure_cohort.birth_year.data)))),
        },
        coords={
            'runs': ('runs',ds_exposure_cohort.runs.data),
            'birth_year': ('birth_year',ds_exposure_cohort.birth_year.data),
            # 'time': ('time',ds_exposure_cohort.time.data),
        }
    )
    
    # cohort conversion to data array
    # now that x axis will be birth year, need to reconsider cohort for denominator of fraction
    # cohorts currently ages and time, need to do paired seleciton like in calc_cohort_emergence() to get full cohort pops for 1960 birth year (or do it elsewhere in separate function)
    # need new cohort array that has total population per birth year (using life expectancy info; each country has a different end point)
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
    
    ds_cohorts = calc_birthyear_cohortsizes(
        da_cohort_size,
        df_life_expectancy,
        year_start,
        year_end,
        year_ref,
    )
    
    # keep only timesteps/values where cumulative exposure exceeds pic defined extreme
    unprec = ds_exposure_cohort['exposure'].where(da_exposure_mask == 1)
    # unprec = unprec.sum(dim=['birth_year','country'])
    unprec = unprec.sum(dim=['time','country'])
    normal = ds_exposure_cohort['exposure'].where(da_exposure_mask == 0)
    # normal = normal.sum(dim=['birth_year','country'])
    normal = normal.sum(dim=['time','country'])
    
    # assign aggregated unprecedented/normal exposure to ds_pop_frac
    ds_pop_frac['unprec'] = unprec
    ds_pop_frac['normal'] = normal
    
    # stats on exposure types
    ds_pop_frac['mean_unprec'] = ds_pop_frac['unprec'].mean(dim='runs')
    ds_pop_frac['max_unprec'] = ds_pop_frac['unprec'].max(dim='runs')
    ds_pop_frac['min_unprec'] = ds_pop_frac['unprec'].min(dim='runs')
    ds_pop_frac['std_unprec'] = ds_pop_frac['unprec'].std(dim='runs')

    ds_pop_frac['mean_normal'] = ds_pop_frac['normal'].mean(dim='runs')
    ds_pop_frac['max_normal'] = ds_pop_frac['normal'].max(dim='runs')
    ds_pop_frac['min_normal'] = ds_pop_frac['normal'].min(dim='runs')
    ds_pop_frac['std_normal'] = ds_pop_frac['normal'].std(dim='runs')
    
    # unprecedented exposure as fraction of total population estimate
    # ds_pop_frac['frac_all_unprec'] = ds_pop_frac['unprec'] / da_cohort_size.sum(dim=['ages','country'])
    ds_pop_frac['frac_all_unprec'] = ds_pop_frac['unprec'] / ds_cohorts['population'].sum(dim=['country'])
    ds_pop_frac['mean_frac_all_unprec'] = ds_pop_frac['frac_all_unprec'].mean(dim='runs')
    ds_pop_frac['std_frac_all_unprec'] = ds_pop_frac['frac_all_unprec'].std(dim='runs')

    # fraction of all exposed
    ds_pop_frac['frac_exposed_unprec'] = ds_pop_frac['unprec'] / (ds_pop_frac['unprec'] + ds_pop_frac['normal'])
    ds_pop_frac['mean_frac_exposed_unprec'] = ds_pop_frac['frac_exposed_unprec'].mean(dim='runs')
    ds_pop_frac['std_frac_exposed_unprec'] = ds_pop_frac['frac_exposed_unprec'].std(dim='runs')
     
    return ds_pop_frac
        
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
    ds_exposure_emergence_birth_year = ds_exposure_emergence.birth_year.where(ds_exposure_emergence.notnull()).min(dim='birth_year',skipna=True)#.astype('int')
    # ds_exposure_emergence_birth_year = xr.where(ds_exposure_emergence_birth_year > 0, ds_exposure_emergence_birth_year, 0)
    
    # for same years, get EMF
    for var in EMF_subset:
    
        ds_exposure_emergence_birth_year[var] = ds_exposure[var].where(ds_exposure[var].birth_year==ds_exposure_emergence_birth_year[var.replace('_EMF','')]).min(dim='birth_year',skipna=True)
    
    # move emergene birth years and EMFs to gdf for plotting
    gdf_exposure_emergence_birth_year = gpd.GeoDataFrame(ds_exposure_emergence_birth_year.to_dataframe().join(gdf_country_borders))
    
    return gdf_exposure_emergence_birth_year,ds_exposure_emergence_birth_year

#%% ----------------------------------------------------------------
def all_emergence(
    da_exposure_peryear_perage_percountry,
    da_exposure_cohort,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
    ds_exposure_pic,
    d_all_cohorts,
    year_range,
    df_countries,
    flag,
    traject,
):
    
    # exposure mask setup
    ds_exposure_mask = calc_cohort_emergence(
        da_exposure_peryear_perage_percountry,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )
    
    # age emergence
    da_exposure_mask,da_age_emergence = exposure_pic_masking(
        ds_exposure_mask,
        ds_exposure_pic,
    )
    
    # cohort exposure emergence
    ds_exposure_cohort = calc_cohort_emergence(
        da_exposure_cohort,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )
    
    # population experiencing normal vs unprecedented exposure
    ds_pop_frac = calc_unprec_exposure(
        ds_exposure_cohort,
        da_exposure_mask,
        d_all_cohorts,
        year_range,
        df_countries,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,        
    )
    
    # pickle pop frac
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flag,traject), 'wb') as f:
        pk.dump(ds_pop_frac,f)
        
    # pickle age emergence
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flag,traject), 'wb') as f:
        pk.dump(da_age_emergence,f)        

    return da_age_emergence, ds_pop_frac

#%% ----------------------------------------------------------------
# plot timing and EMF of exceedence of pic-defined extreme
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
        # label='Population unprecedented',
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
        # label='Population unprecedented',
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
        # label='Population unprecedented',
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
    # plot unprecedented frac of exposed pop

    # NDC
    ax3.plot(
        time,
        ds_pop_frac_NDC['mean_frac_exposed_unprec'].values,
        lw=lw_mean,
        color=col_NDC,
        # label='Population unprecedented',
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
        # label='Population unprecedented',
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
        # label='Population unprecedented',
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

    f,(ax1,ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(x,y),
    )

    # --------------------------------------------------------------------
    # plot unprecedented frac of total pop, ax1 for mean +/- std

    # NDC
    ax1.plot(
        time,
        ds_pop_frac_NDC['mean_frac_all_unprec'].values,
        lw=lw_mean,
        color=col_NDC,
        # label='Population unprecedented',
        zorder=1,
    )
    ax1.fill_between(
        time,
        ds_pop_frac_NDC['mean_frac_all_unprec'].values + ds_pop_frac_NDC['std_frac_all_unprec'].values,
        ds_pop_frac_NDC['mean_frac_all_unprec'].values - ds_pop_frac_NDC['std_frac_all_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_NDC_fill,
        zorder=1,
    )

    # 2.0 degrees
    ax1.plot(
        time,
        ds_pop_frac_20['mean_frac_all_unprec'].values,
        lw=lw_mean,
        color=col_20,
        # label='Population unprecedented',
        zorder=2,
    )
    ax1.fill_between(
        time,
        ds_pop_frac_20['mean_frac_all_unprec'].values + ds_pop_frac_20['std_frac_all_unprec'].values,
        ds_pop_frac_20['mean_frac_all_unprec'].values - ds_pop_frac_20['std_frac_all_unprec'].values,
        lw=lw_fill,
        alpha=ub_alpha,
        color=col_20_fill,
        zorder=2,
    )

    # 1.5 degrees
    ax1.plot(
        time,
        ds_pop_frac_15['mean_frac_all_unprec'].values,
        lw=lw_mean,
        color=col_15,
        # label='Population unprecedented',
        zorder=3,
    )
    ax1.fill_between(
        time,
        ds_pop_frac_15['mean_frac_all_unprec'].values + ds_pop_frac_15['std_frac_all_unprec'].values,
        ds_pop_frac_15['mean_frac_all_unprec'].values - ds_pop_frac_15['std_frac_all_unprec'].values,
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
    # plot unprecedented frac of exposed pop, all runs

    # NDC
    for run in ds_pop_frac_NDC.runs:
        
        ax2.plot(
            time,
            ds_pop_frac_NDC['frac_all_unprec'].sel(runs=run).values,
            lw=lw_mean,
            color=col_NDC,
            # label='Population unprecedented',
            zorder=1,
        )
    # 2.0 degrees
    for run in ds_pop_frac_20.runs:
        
        ax2.plot(
            time,
            ds_pop_frac_20['frac_all_unprec'].sel(runs=run).values,
            lw=lw_mean,
            color=col_20,
            # label='Population unprecedented',
            zorder=2,
        )
    # 1.5 degrees
    for run in ds_pop_frac_15.runs:

        ax2.plot(
            time,
            ds_pop_frac_15['frac_all_unprec'].sel(runs=run).values,
            lw=lw_mean,
            color=col_15,
            # label='Population unprecedented',
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
            
    f.savefig('./figures/pop_frac_birthyear.png',dpi=300)

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
    # plot unprecedented frac of exposed pop, all runs
    for ax,gcm in zip((ax1,ax2,ax3,ax4),list(runs.keys())):
        
        for run in runs[gcm]:
            
            # NDC
            if run in ds_pop_frac_NDC['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_NDC['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_NDC,
                    # label='Population unprecedented',
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
                    # label='Population unprecedented',
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
                    # label='Population unprecedented',
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
    # plot unprecedented frac of exposed pop, all runs
    for ax,mod in zip(axes.flatten(),list(runs.keys())):
        
        for run in runs[mod]:
            
            # NDC
            if run in ds_pop_frac_NDC['frac_all_unprec'].runs.values:
                
                ax.plot(
                    time,
                    ds_pop_frac_NDC['frac_all_unprec'].sel(runs=run).values,
                    lw=lw_mean,
                    color=col_NDC,
                    # label='Population unprecedented',
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
                    # label='Population unprecedented',
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
                    # label='Population unprecedented',
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
# %%
