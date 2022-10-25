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
# make age+time selections to align exposure, cohort exposure and cohort sizes along birth year and time
def calc_birthyear_align(
    da,
    df_life_expectancy,
    year_start,
    year_end,
    year_ref,
):
    
    country_list = []
    
    # loop through countries
    for country in da.country.values:
        
        birthyear_list = []
        
        # per birth year, make (year,age) selections
        for birth_year in np.arange(year_start,year_end+1):
            
            # use life expectancy information where available (until 2020)
            if birth_year <= year_ref:
                
                death_year = birth_year + np.floor(df_life_expectancy.loc[birth_year,country])
                time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
                ages = xr.DataArray(np.arange(0,len(time)),dims='age')
                data = da.sel(country=country,time=time,ages=ages) # paired selections
                data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)
            
            # after 2020, assume constant life expectancy    
            elif birth_year > year_ref and birth_year < year_end:
                
                death_year = birth_year + np.floor(df_life_expectancy.loc[year_ref,country]) #for years after 2020, just take 2020 life expectancy
                
                # if lifespan not encompassed by 2113, set death to 2113
                if death_year > year_end:
                    
                    death_year = year_end
                
                time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
                ages = xr.DataArray(np.arange(0,len(time)),dims='age')
                data = da.sel(country=country,time=time,ages=ages)
                data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)
            
            # for 2113, use single year of exposure    
            elif birth_year == year_end:
                
                time = xr.DataArray([year_end],dims='age')
                ages = xr.DataArray([0],dims='age')
                data = da.sel(country=country,time=time,ages=ages)
                data = data.rename({'age':'time'}).assign_coords({'time':[year_end]})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
                data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
                birthyear_list.append(data)                
        
        da_data = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(da_data)
        
    da_all = xr.concat(country_list,dim='country')
    
    return da_all


#%% ----------------------------------------------------------------
# create dataset out of birthyear aligned cohort exposure and regular rexposure, add cumulative exposure, 
def ds_exposure_align(
    da,
    traject,
):

    if traject != 'strj':
        
        da_cumsum = da.cumsum(dim='time').where(da>0)
        ds_exposure_cohort = xr.Dataset(
            data_vars={
                'exposure': (da.dims,da.data),
                'exposure_cumulative': (da_cumsum.dims,da_cumsum.data)
            },
            coords={
                'country': ('country',da.country.data),
                'birth_year': ('birth_year',da.birth_year.data),
                'runs': ('runs',da.runs.data),
                'time': ('time',da.time.data),
            },
        )
        
    else:
        
        da_cumsum = da.cumsum(dim='time').where(da>0)
        ds_exposure_cohort = xr.Dataset(
            data_vars={
                'exposure': (da.dims,da.data),
                'exposure_cumulative': (da_cumsum.dims,da_cumsum.data)
            },
            coords={
                'country': ('country',da.country.data),
                'birth_year': ('birth_year',da.birth_year.data),
                'runs': ('runs',da.runs.data),
                # 'GMT': ('GMT',da.GMT.data),
                'time': ('time',da.time.data),
            },
        )        
     
    return ds_exposure_cohort

#%% ----------------------------------------------------------------
# create dataset out of birthyear aligned cohort sizes
def ds_cohort_align(
    da,
):

    da = da.sum(dim='time') # for each birth year, get total number of people (population in ds)
    ds_cohort_sizes = xr.Dataset(
        data_vars={
            'population': (da.dims,da.data),
        },
        coords={
            'country': ('country',da.country.data),
            'birth_year': ('birth_year',da.birth_year.data),
        },
    )
     
    return ds_cohort_sizes

#%% ----------------------------------------------------------------
# function to generate mask of unprecedented timesteps per birth year and age of emergence
def exposure_pic_masking(
    ds_exposure_mask,
    ds_exposure_pic,
):
    # generate exposure mask for timesteps after reaching pic extreme to find age of emergence
    ds_exposure_pic['ext'] = ds_exposure_pic['ext'].where(ds_exposure_pic['ext']>0)
    da_age_exposure_mask = xr.where(ds_exposure_mask['exposure_cumulative'] >= ds_exposure_pic['ext'],1,0)
    da_age_emergence = da_age_exposure_mask * (da_age_exposure_mask.time - da_age_exposure_mask.birth_year)
    da_age_emergence = da_age_emergence.where(da_age_emergence!=0).min(dim='time',skipna=True)
    
    # adjust exposure mask; for any birth cohorts that crossed extreme, keep 1s at all lived time steps and 0 for other birth cohorts
    da_birthyear_exposure_mask = xr.where(da_age_exposure_mask.sum(dim='time')>0,1,0) # find birth years crossing threshold
    da_birthyear_exposure_mask = xr.where(ds_exposure_mask['exposure']>0,1,0).where(da_birthyear_exposure_mask==1) # 
    da_exposure_mask = xr.where(da_birthyear_exposure_mask==1,1,0)
    
    return da_exposure_mask,da_age_emergence

#%% ----------------------------------------------------------------
# function to find number of people with unprecedented exposure 
# (original) that summed across time and countries before taking means across runs, creating another copy that does the opposite
def calc_unprec_exposure(
    ds_exposure_cohort,
    da_exposure_mask,
    ds_cohorts,
    year_range,
    df_countries,
    df_life_expectancy,
    year_start,
    year_end,
    year_ref,
    traject,
):

    # new empty dataset with variables for population experiencing unprecedented exposure
    if traject != 'strj':
        
        ds_pop_frac = xr.Dataset(
            data_vars={
                'unprec': (['runs','birth_year'], np.empty((len(ds_exposure_cohort.runs.data),len(ds_exposure_cohort.birth_year.data)))),
            },
            coords={
                'runs': ('runs',ds_exposure_cohort.runs.data),
                'birth_year': ('birth_year',ds_exposure_cohort.birth_year.data),
            }
        )
        
    else:
        
        ds_pop_frac = xr.Dataset(
            data_vars={
                'unprec': (['runs','birth_year'], np.empty((len(ds_exposure_cohort.runs.data),len(ds_exposure_cohort.birth_year.data)))),
            },
            coords={
                'runs': ('runs',ds_exposure_cohort.runs.data),
                'birth_year': ('birth_year',ds_exposure_cohort.birth_year.data),
                # 'GMT': ('GMT',ds_exposure_cohort.GMT.data),
            }
        )
    
    # keep only timesteps/values where cumulative exposure exceeds pic defined extreme
    unprec = ds_exposure_cohort['exposure'].where(da_exposure_mask == 1)
    unprec = unprec.sum(dim=['time','country'])
    if traject == 'strj':
        unprec = unprec.where(unprec!= 0)
    
    # assign aggregated unprecedented/normal exposure to ds_pop_frac
    ds_pop_frac['unprec'] = unprec
    
    # stats on exposure types
    ds_pop_frac['mean_unprec'] = ds_pop_frac['unprec'].mean(dim='runs')
    ds_pop_frac['max_unprec'] = ds_pop_frac['unprec'].max(dim='runs')
    ds_pop_frac['min_unprec'] = ds_pop_frac['unprec'].min(dim='runs')
    ds_pop_frac['std_unprec'] = ds_pop_frac['unprec'].std(dim='runs')
    
    # unprecedented exposure as fraction of total population estimate
    ds_pop_frac['frac_all_unprec'] = ds_pop_frac['unprec'] / ds_cohorts['population'].sum(dim=['country'])
    ds_pop_frac['mean_frac_all_unprec'] = ds_pop_frac['frac_all_unprec'].mean(dim='runs')
    ds_pop_frac['std_frac_all_unprec'] = ds_pop_frac['frac_all_unprec'].std(dim='runs')
     
    return ds_pop_frac

#%% ----------------------------------------------------------------
# get timing and EMF of exceedence of pic-defined extreme
def calc_exposure_emergence(
    ds_exposure,
    ds_exposure_pic,
    ds_age_emergence,
    gdf_country_borders,
):

    mmm_subset = [
        'mmm_RCP',
        'mmm_15',
        'mmm_20',
        'mmm_NDC',
    ]

    # get years where mmm exposures under different trajectories exceed pic 99.99%
    ds_exposure_emergence = ds_exposure[mmm_subset].where(ds_exposure[mmm_subset] > ds_exposure_pic['ext'])
    ds_exposure_emergence_birth_year = ds_exposure_emergence.birth_year.where(ds_exposure_emergence.notnull()).min(dim='birth_year',skipna=True)#.astype('int')
    
    scen_subset = [
        'RCP',
        '15',
        '20',
        'NDC',
    ]
    
    for scen in scen_subset:
        
        ds_exposure_emergence_birth_year['birth_year_age_{}'.format(scen)] = ds_age_emergence['age_{}'.format(scen)].where(
            ds_age_emergence['age_{}'.format(scen)].birth_year==ds_exposure_emergence_birth_year['mmm_{}'.format(scen)]
        ).mean(dim='runs').min(dim='birth_year',skipna=True)    
    
    # move emergene birth years and EMFs to gdf for plotting
    gdf_exposure_emergence_birth_year = gpd.GeoDataFrame(ds_exposure_emergence_birth_year.to_dataframe().join(gdf_country_borders))
    
    return gdf_exposure_emergence_birth_year

#%% ----------------------------------------------------------------
def all_emergence(
    da_exposure_peryear_perage_percountry,
    da_exposure_cohort,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
    ds_exposure_pic,
    ds_cohorts,
    year_range,
    df_countries,
    flag,
    traject,
):
    start_time = time.time()
    
    # align age + time selections of annual mean exposure along birthyears + time per country, birth cohort, run to act as mask for birthyears when pic threshold is passed
    da_exposure_aligned = calc_birthyear_align(
        da_exposure_peryear_perage_percountry,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )
    
    # dataset of birthyear aligned exposure, add cumulative exposure 
    ds_exposure_aligned = ds_exposure_align(
        da_exposure_aligned,
        traject,
    )    
    
    # use birthyear aligned (cumulative) exposure and pic extreme to extract age of emergence and get mask to include all lived timesteps per birthyear that passed pic threshold
    da_exposure_mask,da_age_emergence = exposure_pic_masking(
        ds_exposure_aligned,
        ds_exposure_pic,
    )
    
    # aligned cohort exposure age+time selections
    da_exposure_cohort_aligned = calc_birthyear_align(
        da_exposure_cohort,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
    )
    
    # convert aligned cohort exposure to dataset
    ds_exposure_cohort_aligned = ds_exposure_align(
        da_exposure_cohort_aligned,
        traject,
    )
    
    # population experiencing normal vs unprecedented exposure
    ds_pop_frac = calc_unprec_exposure(
        ds_exposure_cohort_aligned,
        da_exposure_mask,
        ds_cohorts,
        year_range,
        df_countries,
        df_life_expectancy_5,
        year_start,
        year_end,
        year_ref,
        traject,
    )
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )        
    
    # pickle pop frac
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flag,traject), 'wb') as f:
        pk.dump(ds_pop_frac,f)
        
    # pickle age emergence
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flag,traject), 'wb') as f:
        pk.dump(da_age_emergence,f)        

    return da_age_emergence, ds_pop_frac

#%% ----------------------------------------------------------------
def strj_emergence(
    da_exposure_peryear_perage_percountry,
    da_exposure_cohort,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
    ds_exposure_pic,
    ds_cohorts,
    year_range,
    df_countries,
    flag,
    traject,
):
    start_time = time.time()
    exposure_ages = []
    pop_fracs = []  
    
    for step in da_exposure_peryear_perage_percountry.GMT.values:
        
        print('Processing GMT step {} of {}'.format(step,len(da_exposure_peryear_perage_percountry.GMT.values)))
    
        # align age + time selections of annual mean exposure along birthyears + time per country, birth cohort, run to act as mask for birthyears when pic threshold is passed
        da_exposure_aligned = calc_birthyear_align(
            da_exposure_peryear_perage_percountry.sel(GMT=step),
            df_life_expectancy_5,
            year_start,
            year_end,
            year_ref,
        )
        
        # dataset of birthyear aligned exposure, add cumulative exposure 
        ds_exposure_aligned = ds_exposure_align(
            da_exposure_aligned,
            traject,
        )
        
        # use birthyear aligned (cumulative) exposure and pic extreme to extract age of emergence and get mask to include all lived timesteps per birthyear that passed pic threshold
        da_exposure_mask,da_age_emergence = exposure_pic_masking(
            ds_exposure_aligned,
            ds_exposure_pic,
        )
        
        # aligned cohort exposure age+time selections
        da_exposure_cohort_aligned = calc_birthyear_align(
            da_exposure_cohort.sel(GMT=step),
            df_life_expectancy_5,
            year_start,
            year_end,
            year_ref,
        )
        
        # convert aligned cohort exposure to dataset
        ds_exposure_cohort_aligned = ds_exposure_align(
            da_exposure_cohort_aligned,
            traject,
        )
        
        # population experiencing normal vs unprecedented exposure
        ds_pop_frac = calc_unprec_exposure(
            ds_exposure_cohort_aligned,
            da_exposure_mask,
            ds_cohorts,
            year_range,
            df_countries,
            df_life_expectancy_5,
            year_start,
            year_end,
            year_ref,
            traject,
        )
        
        
        exposure_ages.append(da_age_emergence)
        pop_fracs.append(ds_pop_frac)
        
    # testing concat on datasets while loop is small (3)
    ds_pop_frac = xr.concat(pop_fracs,dim='GMT').assign_coords({'GMT':da_exposure_peryear_perage_percountry.GMT.values})
    da_age_emergence = xr.concat(exposure_ages,dim='GMT').assign_coords({'GMT':da_exposure_peryear_perage_percountry.GMT.values})
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )        
    
    # pickle pop frac
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flag,traject), 'wb') as f:
        pk.dump(ds_pop_frac,f)
        
    # pickle age emergence
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flag,traject), 'wb') as f:
        pk.dump(da_age_emergence,f)        

    return da_age_emergence, ds_pop_frac

