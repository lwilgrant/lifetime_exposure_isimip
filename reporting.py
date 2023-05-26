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
import pickle as pk
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import regionmask as rm
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

#%% ----------------------------------------------------------------
# sample analytics for paper
# ------------------------------------------------------------------

#%% ----------------------------------------------------------------
# multi-hazard emergence estimates
# ------------------------------------------------------------------

def multi_hazard_emergence(
    grid_area,
    da_emergence_mean,
    da_gs_popdenom,
):
    with open('./data/pickles/gridscale_cohort_global.pkl', 'rb') as file:
        da_gridscale_cohortsize = pk.load(file)   

    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
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

    density=6
    sim_frac=0.25
    gmt_indices_152535 = [24,15,6]
    gmt = 17 # gmt index to compare multihazard pf
    multiextrn = 3 #number of extremes for multihazard pf comparison
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)

    # pf for 1960 birth cohort in multi hazard case
    template_1960 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=17,birth_year=1960),
        False
    )

    for extr in extremes:

        p1960 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':1960,
        }]
        template_1960 = template_1960+p1960.where(p1960>sim_frac).notnull()

    p_u1960 = template_1960.where(mask.notnull())
    pf_3extr_1960 = da_gridscale_cohortsize.loc[{
        'birth_year':1960,
    }].where(p_u1960>=multiextrn).sum(dim=('lat','lon')) / da_gs_popdenom.loc[{'birth_year':1960}].sum(dim='country') * 100
    print('1960 {}-hazard pf is {} for GMT {}'.format(multiextrn,pf_3extr_1960['cohort_size'].item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))

    # pf for 2020 birth cohort in multi hazard case
    template_2020 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=17,birth_year=2020),
        False
    )

    for extr in extremes:

        p2020 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':2020,
        }]
        template_2020 = template_2020+p2020.where(p2020>sim_frac).notnull()

    p_u2020 = template_2020.where(mask.notnull())
    pf_3extr_2020 = da_gridscale_cohortsize.loc[{
        'birth_year':2020,
    }].where(p_u2020>=multiextrn).sum(dim=('lat','lon')) / da_gs_popdenom.loc[{'birth_year':2020}].sum(dim='country') * 100
    print('2020 {}-hazard pf is {} for GMT {}'.format(multiextrn,pf_3extr_2020['cohort_size'].item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))


    # land area
    la_frac_eu_gteq3_2020 = xr.where(p_u2020>=multiextrn,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100
    la_frac_eu_gteq3_1960 = xr.where(p_u1960>=multiextrn,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100


    print('1960 percentage of land area \n with emergence of {} extremes \n is {} in a {} GMT pathway'.format(multiextrn,la_frac_eu_gteq3_1960.item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))  
    print('2020 percentage of land area \n with emergence of {} extremes \n is {} in a {} GMT pathway'.format(multiextrn,la_frac_eu_gteq3_2020.item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))    

#%% ----------------------------------------------------------------
# grid scale cohort sizes per birth year (copied from jupyter and only useable there)
# ------------------------------------------------------------------

def gridscale_cohort_sizes(
    da_population,
    gridscale_countries,   
):
    # dataset of by_py0
    ds_gridscale_cohortsize = xr.Dataset(
        data_vars={
            'cohort_size': (
                ['birth_year','lat','lon'],
                np.full(
                    (len(birth_years),len(da_population.lat.data),len(da_population.lon.data)),
                    fill_value=np.nan,
                ),
            ),
        },
        coords={
            'lat': ('lat', da_population.lat.data),
            'lon': ('lon', da_population.lon.data),
            'birth_year': ('birth_year', birth_years),
        }
    )

    # loop through countries and assign birth cohort size to dataset
    for cntry in gridscale_countries:
        print(cntry)
        # load demography pickle
        with open('./data/pickles/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
            ds_dmg = pk.load(f)   
        # get population used in analysis
        da_cohort_cntry = ds_dmg['by_population_y0']
        # assign to bigger dataset
        ds_gridscale_cohortsize['cohort_size'].loc[{
            'birth_year':birth_years,
            'lat':da_cohort_cntry.lat.data,
            'lon':da_cohort_cntry.lon.data,
        }] = xr.where(
            da_cohort_cntry.notnull(),
            da_cohort_cntry,
            ds_gridscale_cohortsize['cohort_size'].loc[{'birth_year':birth_years,'lat':da_cohort_cntry.lat.data,'lon':da_cohort_cntry.lon.data}],
        )
    print('countries merged')
    with open('./data/pickles/gridscale_cohort_global.pkl', 'wb') as f:
        pk.dump(ds_gridscale_cohortsize,f)   
        
#%% ----------------------------------------------------------------
# grid exposure locations for all sims 
# ------------------------------------------------------------------        
        
def exposure_locs(
    grid_area,
):        
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]

    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)

    for extr in extremes:
        
        with open('./data/pickles/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)     
            
        n = 0
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/pickles/{}/isimip_AFA_{}_{}.pkl'.format(extr,extr,str(i)), 'rb') as f:
                da_AFA = pk.load(f)           
            
            if n == 0:    
                da_sum = da_AFA.sum(dim='time').where(mask.notnull())
            else:
                da_sum = da_sum + da_AFA.sum(dim='time').where(mask.notnull())
            
            n+=1
            
        da_exposure_occurence = xr.where(da_sum>0,1,0)
        
        with open('./data/pickles/{}/exposure_occurrence_{}.pkl'.format(extr,extr), 'wb') as file:
            pk.dump(da_exposure_occurence,file)      
            
            
#%% ----------------------------------------------------------------
# emergence locations in specific runs for 1.5, 2.5, 2.7 and 3.5
# used for geographically constrained PF to recompute with different 
# denominator (used for numerator with grid scale pop to compare against
# pop of regions pinged by exposure in da_exposure_occurrence)
# ------------------------------------------------------------------                    

def emergence_locs_perrun(
    flags,
    grid_area,
    gridscale_countries,
    countries_mask,
    countries_regions,
):

    gmt_indices_sample = [24,17,15,6]
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    da_mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]    
            
    # loop through extremes
    for extr in extremes:
        
        start_time = time.time()
        
        # get metadata for extreme
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)
            
        sims_per_step = {}
        for step in gmt_indices_sample:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)                           
        
        for step in gmt_indices_sample:
            
            ds_global_emergence = xr.Dataset(
                data_vars={
                    'emergence': (
                        ['run','birth_year','lat','lon'],
                        np.full(
                            (len(sims_per_step[step]),len(birth_years),len(da_mask.lat.data),len(da_mask.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                              
                },
                coords={
                    'lat': ('lat', da_mask.lat.data),
                    'lon': ('lon', da_mask.lon.data),
                    'birth_year': ('birth_year', birth_years),
                    'run': ('run', sims_per_step[step]),
                    'GMT': ('GMT', GMT_labels),
                }
            )        
            
            # loop through countries
            for cntry in gridscale_countries:
                
                da_cntry = xr.DataArray(
                    np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                    dims=countries_mask.dims,
                    coords=countries_mask.coords,
                )
                da_cntry = da_cntry.where(da_cntry,drop=True)                  
                
                # dataset for extreme - country - GMT
                ds_cntry_emergence = xr.Dataset(
                    data_vars={
                        'emergence': (
                            ['run','birth_year','lat','lon'],
                            np.full(
                                (len(sims_per_step[step]),len(birth_years),len(da_cntry.lat.data),len(da_cntry.lon.data)),
                                fill_value=np.nan,
                            ),
                        ),                          
                    },
                    coords={
                        'lat': ('lat', da_cntry.lat.data),
                        'lon': ('lon', da_cntry.lon.data),
                        'birth_year': ('birth_year', birth_years),
                        'run': ('run', sims_per_step[step]),
                    }
                )                      
            
                # loop through sims and pick emergence masks for sims that are valid
                for i in sims_per_step[step]: 
                    
                    if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    
                        with open('./data/pickles/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(extr,extr,cntry,i,step), 'rb') as f:
                            da_birthyear_emergence_mask = pk.load(f)
                            
                        ds_cntry_emergence['emergence'].loc[{
                            'run':i,
                            'birth_year':birth_years,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,
                        }] = da_birthyear_emergence_mask      
                        
                
                ds_global_emergence['emergence'].loc[{
                    'run':sims_per_step[step],
                    'birth_year':birth_years,
                    'lat':da_cntry.lat.data,
                    'lon':da_cntry.lon.data,
                }] = xr.where(
                    ds_cntry_emergence['emergence'].notnull(),
                    ds_cntry_emergence['emergence'],
                    ds_global_emergence['emergence'].loc[{
                        'run':sims_per_step[step],'birth_year':birth_years,'lat':da_cntry.lat.data,'lon':da_cntry.lon.data
                    }],
                )             
                
            with open('./data/pickles/{}/emergence_locs_perrun_{}_{}.pkl'.format(extr,extr,step), 'wb') as f:
                pk.dump(ds_global_emergence['emergence'],f)        
                
        print("--- {} minutes for {} emergence loc ---".format(
            np.floor((time.time() - start_time) / 60),
            extr
            )
            )    
                   