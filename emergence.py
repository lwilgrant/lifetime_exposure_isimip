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
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot = init()

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

# #%% ----------------------------------------------------------------
# # make age+time selections to align exposure, cohort exposure and cohort sizes along birth year and time
# def calc_birthyear_align(
#     da,
#     df_life_expectancy,
#     year_start,
#     year_end,
#     year_ref,
# ):
    
#     country_list = []
    
#     # loop through countries
#     for country in da.country.values:
        
#         birthyear_list = []
        
#         # per birth year, make (year,age) selections
#         for birth_year in np.arange(year_start,year_end+1):
            
#             # use life expectancy information where available (until 2020)
#             if birth_year <= year_ref:
                
#                 death_year = birth_year + np.floor(df_life_expectancy.loc[birth_year,country])
#                 time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
#                 ages = xr.DataArray(np.arange(0,len(time)),dims='age')
#                 data = da.sel(country=country,time=time,ages=ages) # paired selections
#                 data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
#                 data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
#                 data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
#                 birthyear_list.append(data)
            
#             # after 2020, assume constant life expectancy    
#             elif birth_year > year_ref and birth_year < year_end:
                
#                 death_year = birth_year + np.floor(df_life_expectancy.loc[year_ref,country]) #for years after 2020, just take 2020 life expectancy
                
#                 # if lifespan not encompassed by 2113, set death to 2113
#                 if death_year > year_end:
                    
#                     death_year = year_end
                
#                 time = xr.DataArray(np.arange(birth_year,death_year),dims='age')
#                 ages = xr.DataArray(np.arange(0,len(time)),dims='age')
#                 data = da.sel(country=country,time=time,ages=ages)
#                 data = data.rename({'age':'time'}).assign_coords({'time':np.arange(birth_year,death_year,dtype='int')})
#                 data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
#                 data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
#                 birthyear_list.append(data)
            
#             # for 2113, use single year of exposure    
#             elif birth_year == year_end:
                
#                 time = xr.DataArray([year_end],dims='age')
#                 ages = xr.DataArray([0],dims='age')
#                 data = da.sel(country=country,time=time,ages=ages)
#                 data = data.rename({'age':'time'}).assign_coords({'time':[year_end]})
#                 data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze()
#                 data = data.assign_coords({'birth_year':birth_year}).drop_vars('ages')
#                 birthyear_list.append(data)                
        
#         da_data = xr.concat(birthyear_list,dim='birth_year')
#         country_list.append(da_data)
        
#     da_all = xr.concat(country_list,dim='country')
    
#     return da_all

#%% ----------------------------------------------------------------
# make age+time selections to align exposure, cohort exposure and cohort sizes along birth year and time
# this is new version that should be faster because I limit the alignment to 1960 to 2020 (birth_years instead of year range)
def calc_birthyear_align(
    da,
    df_life_expectancy,
):
    
    country_list = []
    
    # loop through countries
    for country in da.country.values:
        
        birthyear_list = []
        
        # per birth year, make (year,age) selections
        for by in birth_years:
                
            death_year = by + np.ceil(df_life_expectancy.loc[by,country]) # since we don't have AFA, best to round life expec up and then multiply last year of exposure by fraction of final year lived
            time = xr.DataArray(np.arange(by,death_year+1),dims='cohort')
            ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
            data = da.sel(country=country,time=time,ages=ages) # paired selections
            data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,death_year+1,dtype='int')})
            data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
            data = data.assign_coords({'birth_year':by}).drop_vars('ages')
            data.loc[{'time':death_year}] = data.loc[{'time':death_year}] * (df_life_expectancy.loc[by,country] - np.floor(df_life_expectancy.loc[by,country]))
            birthyear_list.append(data)
        
        da_data = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(da_data)
        
    da_all = xr.concat(country_list,dim='country')
    
    return da_all
#%% ----------------------------------------------------------------
# create dataset out of birthyear aligned cohort exposure and regular rexposure, add cumulative exposure, 
def ds_exposure_align(
    da,
):
        
    da_cumsum = da.cumsum(dim='time').where(da>0)
    ds_exposure_cohort = xr.Dataset(
        data_vars={
            'exposure': (da.dims, da.data),
            'exposure_cumulative': (da_cumsum.dims, da_cumsum.data)
        },
        coords={
            'country': ('country', da.country.data),
            'birth_year': ('birth_year', da.birth_year.data),
            'time': ('time', da.time.data),
        },
    )
     
    return ds_exposure_cohort

#%% ----------------------------------------------------------------
# create dataset out of birthyear aligned cohort sizes
def ds_cohort_align(
    da,
    da_aligned,
):
    da_t = da.sum(dim='ages') 
    da_by = da_aligned.sum(dim='time') 
    ds_cohort_sizes = xr.Dataset(
        data_vars={
            'by_population': (da_by.dims,da_by.data), # for each birth year, get total number of people for by_weights (population in ds; duplicate counting risk with time sum)
            'population': (da_aligned.dims,da_aligned.data), # population per birth year distributed over time for emergence mask
            't_population': (da_t.dims,da_t.data) # population per timestep across countries for t_weights
        },
        coords={
            'country': ('country',da_aligned.country.data),
            'birth_year': ('birth_year',da_aligned.birth_year.data),
            'time': ('time',da_aligned.time.data),
            'ages': ('ages',da.ages.data)
        },
    )
    
    ds_cohort_sizes['by_weights'] = (ds_cohort_sizes['by_population'] / ds_cohort_sizes['by_population'].sum(dim='country')) # add cohort weights to dataset    
    ds_cohort_sizes['t_weights'] = (ds_cohort_sizes['t_population'] / ds_cohort_sizes['t_population'].sum(dim='country')) # add cohort weights to dataset    
    
     
    return ds_cohort_sizes

#%% ----------------------------------------------------------------
# function to generate mask of unprecedented timesteps per birth year and age of emergence
def exposure_pic_masking(
    ds_emergence_mask,
    ds_exposure_pic,
):
    
    age_emergence_list = []
    emergence_mask_list = []
    ds_exposure_pic['ext'] = ds_exposure_pic['ext'].where(ds_exposure_pic['ext']>0)
    
    for c in ds_emergence_mask.country.data:
        
        # generate exposure mask for timesteps after reaching pic extreme to find age of emergence
        da_age_emergence_mask = xr.where(
            ds_emergence_mask['exposure_cumulative'].sel(country=c) >= ds_exposure_pic['ext'].sel(country=c),
            1,
            0,
        )
        da_age_emergence = da_age_emergence_mask * (da_age_emergence_mask.time - da_age_emergence_mask.birth_year)
        da_age_emergence = da_age_emergence.where(da_age_emergence!=0).min(dim='time',skipna=True)
        age_emergence_list.append(da_age_emergence)
            
        # adjust exposure mask; for any birth cohorts that crossed extreme, keep 1s at all lived time steps and 0 for other birth cohorts
        da_birthyear_emergence_mask = xr.where(da_age_emergence_mask.sum(dim='time')>0,1,0) # find birth years crossing threshold
        da_birthyear_emergence_mask = xr.where(ds_emergence_mask['exposure'].sel(country=c).notnull(),1,0).where(da_birthyear_emergence_mask==1) # first get array of 1s for timesteps when lifetimes exist in aligned exposure array, then only keep birthyears crossing threshold in emergence mask
        da_emergence_mask = xr.where(da_birthyear_emergence_mask==1,1,0) # turn missing values to 0
        emergence_mask_list.append(da_emergence_mask)
        
    # concat across countries
    da_age_emergence = xr.concat(
        age_emergence_list,
        dim='country',
    )
    da_emergence_mask = xr.concat(
        emergence_mask_list,
        dim='country',
    )    
    
    # turn age emergence into dataset
    ds_age_emergence = xr.Dataset(
        data_vars={
            'age_emergence': (da_age_emergence.dims, da_age_emergence.data)
        },
        coords={
            'country': ('country', da_age_emergence.country.data),
            'birth_year': ('birth_year', da_age_emergence.birth_year.data),
        },        
    )        
    
    return da_emergence_mask,ds_age_emergence

#%% ----------------------------------------------------------------
# function to find number of people with unprecedented exposure 
def calc_unprec_exposure(
    ds_exposure_cohort,
    da_emergence_mask,
    ds_cohorts,
):

    # new empty dataset with variables for population experiencing unprecedented exposure
    ds_pop_frac = xr.Dataset(
        data_vars={
            'unprec_exposed_b': (['birth_year'], np.empty((len(ds_exposure_cohort.birth_year.data)))),
            'unprec_exposed_t': (['time'], np.empty((len(ds_exposure_cohort.time.data)))),
            'unprec_all_b': (['birth_year'], np.empty((len(ds_exposure_cohort.birth_year.data)))),
            'unprec_all_t': (['time'], np.empty((len(ds_exposure_cohort.time.data)))),
        },
        coords={
            'birth_year': ('birth_year', ds_exposure_cohort.birth_year.data),
            'time': ('time', ds_exposure_cohort.time.data),
        }
    )
    
    # -------------------------------------------
    # birth year based pop frac
    
    # keep people exposed during birth year's timesteps if cumulative exposure exceeds pic defined extreme
    unprec_exposed = ds_exposure_cohort['exposure'].where(da_emergence_mask==1)
    unprec_exposed = unprec_exposed.sum(dim=['time','country'])
    
    # keep all people/members of birth year cohort if birth year's timesteps cum exposure exceeds pic extreme
    da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0)
    unprec_all = ds_cohorts['by_population'].where(da_birthyear_emergence_mask==1)
    unprec_all = unprec_all.sum(dim='country') # want to do sums over time and country before reading in pickles for ensemble stats
    
    # for stylized trajectories this avoids including a bunch of 0 pop frac runs that got calc'd from nans from checking maxdiff criteria
    unprec_exposed = unprec_exposed.where(unprec_exposed!=0)
    unprec_all = unprec_all.where(unprec_all!=0)
    
    # assign aggregated unprecedented exposure to ds_pop_frac
    ds_pop_frac['unprec_exposed_b'] = unprec_exposed
    ds_pop_frac['unprec_all_b'] = unprec_all
    
    # -------------------------------------------
    # time based pop frac
    
    # keep people exposed during birth year's timesteps if cumulative exposure exceeds pic defined extreme
    unprec_exposed = ds_exposure_cohort['exposure'].where(da_emergence_mask==1)
    unprec_exposed = unprec_exposed.sum(dim=['time','country'])
    
    # keep all people/members of birth year cohort if birth year's timesteps cum exposure exceeds pic extreme
    da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0)
    unprec_all = ds_cohorts['by_population'].where(da_birthyear_emergence_mask==1)
    unprec_all = unprec_all.sum(dim='country') # want to do sums over time and country before reading in pickles for ensemble stats
    
    # for stylized trajectories this avoids including a bunch of 0 pop frac runs that got calc'd from nans from checking maxdiff criteria
    unprec_exposed = unprec_exposed.where(unprec_exposed!=0)
    unprec_all = unprec_all.where(unprec_all!=0)
    
    # assign aggregated unprecedented exposure to ds_pop_frac
    ds_pop_frac['unprec_exposed_b'] = unprec_exposed
    ds_pop_frac['unprec_all_b'] = unprec_all    
     
    return ds_pop_frac

#%% ----------------------------------------------------------------
# function to run stats on pop frac across runs
def pop_frac_stats(
    ds_pop_frac,
    ds_cohorts,
):
    
    # stats on exposure types
    ds_pop_frac['mean_unprec_exposed'] = ds_pop_frac['unprec_exposed'].mean(dim='run')
    ds_pop_frac['max_unprec_exposed'] = ds_pop_frac['unprec_exposed'].max(dim='run')
    ds_pop_frac['min_unprec_exposed'] = ds_pop_frac['unprec_exposed'].min(dim='run')
    ds_pop_frac['std_unprec_exposed'] = ds_pop_frac['unprec_exposed'].std(dim='run')
    ds_pop_frac['mean_unprec_all'] = ds_pop_frac['unprec_all'].mean(dim='run')
    ds_pop_frac['max_unprec_all'] = ds_pop_frac['unprec_all'].max(dim='run')
    ds_pop_frac['min_unprec_all'] = ds_pop_frac['unprec_all'].min(dim='run')
    ds_pop_frac['std_unprec_all'] = ds_pop_frac['unprec_all'].std(dim='run')
    
    # unprecedented exposure as fraction of total population estimate
    ds_pop_frac['frac_unprec_exposed'] = ds_pop_frac['unprec_exposed'] / ds_cohorts['population'].sum(dim=['country'])
    ds_pop_frac['mean_frac_unprec_exposed'] = ds_pop_frac['frac_unprec_exposed'].mean(dim='run')
    ds_pop_frac['std_frac_unprec_exposed'] = ds_pop_frac['frac_unprec_exposed'].std(dim='run')
    ds_pop_frac['frac_unprec_all'] = ds_pop_frac['unprec_all'] / ds_cohorts['population'].sum(dim=['country'])
    ds_pop_frac['mean_frac_unprec_all'] = ds_pop_frac['frac_unprec_all'].mean(dim='run')
    ds_pop_frac['std_frac_unprec_all'] = ds_pop_frac['frac_unprec_all'].std(dim='run')    
    
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
        'mmm_15',
        'mmm_20',
        'mmm_NDC',
    ]

    # get years where mmm exposures under different trajectories exceed pic 99.99%
    ds_exposure_emergence = ds_exposure[mmm_subset].where(ds_exposure[mmm_subset] > ds_exposure_pic['ext'])
    ds_exposure_emergence_birth_year = ds_exposure_emergence.birth_year.where(ds_exposure_emergence.notnull()).min(dim='birth_year',skipna=True)#.astype('int')
    
    scen_subset = [
        '15',
        '20',
        'NDC',
    ]
    
    for scen in scen_subset:
        
        ds_exposure_emergence_birth_year['birth_year_age_{}'.format(scen)] = ds_age_emergence['age_emergence_{}'.format(scen)].where(
            ds_age_emergence['age_emergence_{}'.format(scen)].birth_year==ds_exposure_emergence_birth_year['mmm_{}'.format(scen)]
        ).mean(dim='run').min(dim='birth_year',skipna=True)    
    
    # move emergence birth years and EMFs to gdf for plotting
    gdf_exposure_emergence_birth_year = gpd.GeoDataFrame(ds_exposure_emergence_birth_year.to_dataframe().join(gdf_country_borders))
    
    return gdf_exposure_emergence_birth_year

#%% ----------------------------------------------------------------
def all_emergence(
    d_isimip_meta,
    df_life_expectancy_5,
    year_start,
    year_end,
    year_ref,
    ds_exposure_pic,
    ds_cohorts,
    flag_extr,
    flag_gmt,
    traject,
):
    start_time = time.time()
    
    # create new template datasets for pop_frac and age_emergence, make assignments for each run
    ds_pop_frac = xr.Dataset(
        data_vars={
            'unprec_exposed': (
                ['run','birth_year'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(year_range)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_all': (
                ['run','birth_year'], 
                np.full(
                    (len(list(d_isimip_meta.keys())),len(year_range)),
                    fill_value=np.nan,
                ),
            ),
        },
        coords={
            'run': ('run', list(d_isimip_meta.keys())),
            'birth_year': ('birth_year', year_range),
        }
    )
    ds_age_emergence = xr.Dataset(
        data_vars={
            'age_emergence': (
                ['country','birth_year','run'], 
                np.full(
                    (len(ds_cohorts.country.data),len(year_range),len(list(d_isimip_meta.keys()))), 
                    fill_value=np.nan,
                )
            ),
        },
        coords={
            'country': ('country', ds_cohorts.country.data),
            'birth_year': ('birth_year', year_range),
            'run': ('run', list(d_isimip_meta.keys())),
        },        
    )
    
    # loop through sims and run emergence analysis
    for i in list(d_isimip_meta.keys()):
        
        if d_isimip_meta[i]['GMT_{}_valid'.format(traject)]:
            
            with open('./data/pickles/exposure_cohort_{}_{}_{}_{}.pkl'.format(traject,d_isimip_meta[i]['extreme'],flag_gmt,i), 'rb') as f:
                da_exposure_cohort = pk.load(f)
            with open('./data/pickles/exposure_peryear_perage_percountry_{}_{}_{}_{}.pkl'.format(traject,d_isimip_meta[i]['extreme'],flag_gmt,i), 'rb') as f:
                da_exposure_peryear_perage_percountry = pk.load(f)
            
            # align age + time selections of annual mean exposure along birthyears + time per country, birth cohort, run to act as mask for birthyears when pic threshold is passed
            da_exposure_aligned = calc_birthyear_align(
                da_exposure_peryear_perage_percountry,
                df_life_expectancy_5,
            )
            
            # dataset of birthyear aligned exposure, add cumulative exposure 
            ds_exposure_aligned = ds_exposure_align(
                da_exposure_aligned,
                traject,
            )
            
            # use birthyear aligned (cumulative) exposure and pic extreme to extract age of emergence and get mask to include all lived timesteps per birthyear that passed pic threshold
            da_exposure_mask, ds_age_emergence.loc[{
                'country': ds_cohorts.country.data,
                'birth_year': year_range,
                'run': i,
            }] = exposure_pic_masking(
                    ds_exposure_aligned,
                    ds_exposure_pic,
                )
            
            # aligned cohort exposure age+time selections
            da_exposure_cohort_aligned = calc_birthyear_align(
                da_exposure_cohort,
                df_life_expectancy_5,
            )
            
            # convert aligned cohort exposure to dataset
            ds_exposure_cohort_aligned = ds_exposure_align(
                da_exposure_cohort_aligned,
                traject,
            )
            
            # population experiencing normal vs unprecedented exposure
            ds_pop_frac.loc[{
                'run': i,
                'birth_year': year_range,
            }] = calc_unprec_exposure(
                    ds_exposure_cohort_aligned,
                    da_exposure_mask,
                    ds_cohorts,
                    traject,
                )
    
    # run ensemble stats across runs
    ds_pop_frac = pop_frac_stats(
        ds_pop_frac,
        ds_cohorts,
    )
        
    # pickle pop frac for individual run
    with open('./data/pickles/pop_frac_{}_{}_{}.pkl'.format(flag_ext,flag_gmt,traject), 'wb') as f:
        pk.dump(ds_pop_frac,f)
    
    # pickle age emergence for individual run
    with open('./data/pickles/age_emergence_{}_{}_{}.pkl'.format(flag_ext,flag_gmt,traject), 'wb') as f:
        pk.dump(ds_age_emergence,f)
            
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
    )

    return ds_age_emergence, ds_pop_frac

#%% ----------------------------------------------------------------
def strj_emergence(
    d_isimip_meta,
    df_life_expectancy_5,
    ds_exposure_pic,
    ds_cohorts,
    flag_extr,
    flag_gmt,
):
    start_time = time.time()
    
    # create new template datasets for pop_frac and age_emergence, make assignments for each run and GMT trajectory
    ds_pop_frac = xr.Dataset(
        data_vars={
            'unprec_exposed': (
                ['run','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(birth_years),len(GMT_labels)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_all': (
                ['run','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(birth_years),len(GMT_labels)),
                    fill_value=np.nan
                ),
            ),
        },
        coords={
            'run': ('run', list(d_isimip_meta.keys())),
            'birth_year': ('birth_year', birth_years),
            'GMT': ('GMT', np.arange(len(GMT_labels))),
        }
    )
    
    ds_age_emergence = xr.Dataset(
        data_vars={
            'age_emergence': (
                ['country','birth_year','run','GMT'], 
                np.full(
                    (len(ds_cohorts.country.data),len(birth_years),len(list(d_isimip_meta.keys())),len(GMT_labels)),
                    fill_value=np.nan,
                )
            ),
        },
        coords={
            'country': ('country', ds_cohorts.country.data),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', list(d_isimip_meta.keys())),
            'GMT': ('GMT', np.arange(len(GMT_labels))),
        },        
    )
    
    # loop through sims and run emergence analysis
    for i in list(d_isimip_meta.keys()):
        
        with open('./data/pickles/exposure_cohort_{}_{}_{}.pkl'.format(d_isimip_meta[i]['extreme'],flag_gmt,i), 'rb') as f:
            da_exposure_cohort = pk.load(f)
        with open('./data/pickles/exposure_peryear_perage_percountry_{}_{}_{}.pkl'.format(d_isimip_meta[i]['extreme'],flag_gmt,i), 'rb') as f:
            da_exposure_peryear_perage_percountry = pk.load(f)
    
        for step in da_exposure_peryear_perage_percountry.GMT.values:
            
            print('Processing GMT step {} of {}, run {}'.format(step,len(GMT_labels),i))
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                
                # check for pickles before running full proc
                if not os.path.isfile('./data/pickles/ds_exposure_aligned_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step)):
                    
                    # align age + time selections of annual mean exposure along birthyears + time per country, birth cohort, run; to act as mask for birthyears when pic threshold is passed
                    da_exposure_aligned = calc_birthyear_align(
                        da_exposure_peryear_perage_percountry.sel(GMT=step),
                        df_life_expectancy_5,
                    )
                    
                    # dataset of birthyear aligned exposure, add cumulative exposure 
                    ds_exposure_aligned_run_step = ds_exposure_align(
                        da_exposure_aligned,
                    )
                    
                    # pickle
                    with open('./data/pickles/ds_exposure_aligned_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'wb') as f:
                        pk.dump(ds_exposure_aligned_run_step,f)
                    
                else:
                    
                    # load pickle
                    with open('./data/pickles/ds_exposure_aligned_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'rb') as f:
                        ds_exposure_aligned_run_step = pk.load(f)
                
                # check for pickles before running
                if not os.path.isfile('./data/pickles/da_emergence_mask_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step)) or not os.path.isfile('./data/pickles/ds_age_emergence_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step)):
                    
                    # use birthyear aligned (cumulative) exposure and pic extreme to extract age of emergence and get mask to include all lived timesteps per birthyear that passed pic threshold
                    da_emergence_mask_run_step, ds_age_emergence_run_step = exposure_pic_masking(
                        ds_exposure_aligned_run_step,
                        ds_exposure_pic,
                    )
                    
                    # pickle
                    with open('./data/pickles/da_emergence_mask_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'wb') as f:
                        pk.dump(da_emergence_mask_run_step,f)
                    with open('./data/pickles/ds_age_emergence_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'wb') as f:
                        pk.dump(ds_age_emergence_run_step,f)
                    
                else:
                    
                    # load pickle
                    with open('./data/pickles/da_exposure_mask_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'rb') as f:
                        da_emergence_mask_run_step = pk.load(f)
                
                # check for pickles before running
                if not os.path.isfile('./data/pickles/ds_exposure_cohort_aligned_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step)):
                
                    # aligned cohort exposure age+time selections
                    da_exposure_cohort_aligned_run_step = calc_birthyear_align(
                        da_exposure_cohort.sel(GMT=step),
                        df_life_expectancy_5,
                    )
                    
                    # convert aligned cohort exposure to dataset
                    ds_exposure_cohort_aligned_run_step = ds_exposure_align(
                        da_exposure_cohort_aligned_run_step,
                    )
                    
                    # pickle
                    with open('./data/pickles/ds_exposure_cohort_aligned_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'wb') as f:
                        pk.dump(ds_exposure_cohort_aligned_run_step,f)
                        
                else:
                    
                    # load pickle
                    with open('./data/pickles/ds_exposure_cohort_aligned_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'rb') as f:
                        ds_exposure_cohort_aligned_run_step = pk.load(f)
                
                # check for pop frac before running; don't load on else since loading at end for total dataset of all runs/GMT steps
                if not os.path.isfile('./data/pickles/ds_pop_frac_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step)):
                    
                    # population experiencing normal vs unprecedented exposure
                    ds_pop_frac_run_step = calc_unprec_exposure(
                        ds_exposure_cohort_aligned_run_step,
                        da_emergence_mask_run_step,
                        ds_cohorts,
                    )
                    
                    # pickle
                    with open('./data/pickles/ds_pop_frac_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'wb') as f:
                        pk.dump(ds_pop_frac_run_step,f)                
    
    # loop through sims and and GMT levels and assign age emergence and pop frac to full datasets
    for i in list(d_isimip_meta.keys()):
    
        for step in da_exposure_peryear_perage_percountry.GMT.values:   
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]: 
            
                with open('./data/pickles/ds_age_emergence_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'rb') as f:
                    ds_age_emergence_run_step = pk.load(f)
                with open('./data/pickles/ds_pop_frac_{}_{}_{}_{}.pkl'.format(flag_gmt,flag_extr,i,step), 'rb') as f:
                    ds_pop_frac_run_step = pk.load(f)                     
                
                # assign run/GMT to larger dataset
                ds_age_emergence.loc[{
                    'country': ds_cohorts.country.data,
                    'birth_year': birth_years,
                    'run': i,
                    'GMT': step,
                }] = ds_age_emergence_run_step    
                
                # population experiencing normal vs unprecedented exposure
                ds_pop_frac.loc[{
                    'run':i,
                    'birth_year': birth_years,
                    'GMT': step,
                }] = ds_pop_frac_run_step
                              
    # run ensemble stats across runs
    ds_pop_frac = pop_frac_stats(
        ds_pop_frac,
        ds_cohorts,
    )
        
    # pickle pop frac for individual run
    with open('./data/pickles/pop_frac_{}_{}.pkl'.format(flag_extr,flag_gmt), 'wb') as f:
        pk.dump(ds_pop_frac,f)
    
    # pickle age emergence for individual run
    with open('./data/pickles/age_emergence_{}_{}.pkl'.format(flag_extr,flag_gmt), 'wb') as f:
        pk.dump(ds_age_emergence,f)
            
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
    )

    return ds_age_emergence, ds_pop_frac

