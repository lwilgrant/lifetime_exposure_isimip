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
                        
                ds_cntry_emergence['emergence'] = ds_cntry_emergence['emergence'].where(da_cntry == 1)
                
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
        
#%% ----------------------------------------------------------------
# population fraction estimates per run and for selected GMTs 
# when constraining denominator by geography ie exposed locations
# in our dataset
# ------------------------------------------------------------------            

def pf_geoconstrained(
    flags,
    countries_mask,
):

    gmt_indices_sample = [6,15,17,24]

    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]    

    with open('./data/pickles/gridscale_cohort_global.pkl', 'rb') as file:
        ds_gridscale_cohortsize = pk.load(file)   
        
    da_gridscale_cohortsize = ds_gridscale_cohortsize['cohort_size']

    # loop through extremes
    for extr in extremes:

        start_time = time.time()

        # first get all regions that have exposure to extr in ensemble
        with open('./data/pickles/{}/exposure_occurrence_{}.pkl'.format(extr,extr), 'rb') as file:
            da_exposure_occurrence = pk.load(file)          

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
                    
        ds_pf_geoconstrained = xr.Dataset(
            data_vars={
                'p_perrun': (
                    ['GMT','run','birth_year'],
                    np.full(
                        (len(gmt_indices_sample),len(sims_per_step[gmt_indices_sample[0]]),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),                
                'pf_perrun': (
                    ['GMT','run','birth_year'],
                    np.full(
                        (len(gmt_indices_sample),len(sims_per_step[gmt_indices_sample[0]]),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),                                        
            },
            coords={
                'birth_year': ('birth_year', birth_years),
                'run': ('run', sims_per_step[gmt_indices_sample[0]]),
                'GMT': ('GMT', gmt_indices_sample),
            }
        )       
        # numerator to exposure constrained PF
        for step in gmt_indices_sample:
            
            with open('./data/pickles/{}/emergence_locs_perrun_{}_{}.pkl'.format(extr,extr,step), 'rb') as f:
                da_global_emergence = pk.load(f)
                
            da_global_emergence = xr.where(da_global_emergence==1,1,0)    

            for r in da_global_emergence.run.data:
                
                da_global_emergence.loc[{'run':r}] = da_global_emergence.loc[{'run':r}] * da_gridscale_cohortsize

            da_unprec_p = da_global_emergence.sum(dim=('lat','lon'))

            da_total_p = da_exposure_occurrence.where(countries_mask.notnull()) * da_gridscale_cohortsize
            da_total_p = da_total_p.sum(dim=('lat','lon'))

            da_pf = da_unprec_p / da_total_p

            ds_pf_geoconstrained['p_perrun'].loc[{
                'GMT':step,
                'run':da_pf.run.data,
                'birth_year':birth_years,
            }] = da_unprec_p
            
            ds_pf_geoconstrained['pf_perrun'].loc[{
                'GMT':step,
                'run':da_pf.run.data,
                'birth_year':birth_years,
            }] = da_pf        
        
        with open('./data/pickles/{}/pf_geoconstrained_{}.pkl'.format(extr,extr), 'wb') as f:
            pk.dump(ds_pf_geoconstrained,f)  
        
        print("--- {} minutes for {} pf in under geo constraints ---".format(
            np.floor((time.time() - start_time) / 60),
            extr
            )
            )                

#%% ----------------------------------------------------------------
# read in geoconstrained pf and print for 1960 and 2020 across GMTs
# ------------------------------------------------------------------                           
                   
def print_pf_geoconstrained(
    flags,    
    da_gs_popdenom,
):

    gmt_indices_sample = [6,15,17,24]
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]

    for extr in extremes:
        
        with open('./data/pickles/{}/pf_geoconstrained_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_geoconstrained = pk.load(f)      
            
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)      
            
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
        
        for step in gmt_indices_sample:
            
            pf_geo = ds_pf_geoconstrained['pf_perrun'].loc[{'GMT':step,'run':sims_per_step[step]}].mean(dim='run') * 100
            pf = ds_pf_gs['unprec'].loc[{'GMT':step,'run':sims_per_step[step]}].fillna(0).sum(dim='country').mean(dim='run') / da_gs_popdenom.sum(dim='country') * 100
            
            print('{} under GMT step {} has geoconstrained pf of {} for 1960 and {} for 2020'.format(extr,step,pf_geo.loc[{'birth_year':1960}].item(),pf_geo.loc[{'birth_year':2020}].item()))
            print('{} under GMT step {} has regular pf of {} for 1960 and {} for 2020'.format(extr,step,pf.loc[{'birth_year':1960}].item(),pf.loc[{'birth_year':2020}].item()))
            
#%% ----------------------------------------------------------------
# checking for signifiance of change in means between 1960 and 2020 pf per event and for a GMT level
# low sensitivity to ttest_ind() or ttest_rel() choice
# ------------------------------------------------------------------        
    
def paired_ttest(
    flags,
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

    for extr in extremes:
        
        # coefficient of  of variation
        mean = ds_pf_gs_extrs.sel(hazard=extr).mean(dim=('run','birth_year')).item()
        std = ds_pf_gs_extrs.sel(hazard=extr).std(dim=('run','birth_year')).item()
        cv = std / mean
        print('CV is {}'.format(cv))
        mean_1960 = ds_pf_gs_extrs.sel(hazard=extr,birth_year=1960).mean(dim=('run')).item()
        mean_2020 = ds_pf_gs_extrs.sel(hazard=extr,birth_year=2020).mean(dim=('run')).item()
        delta_mean = mean_2020 - mean_1960
        delta_ratio = delta_mean / mean
        print('delta mean ratio is {}'.format(delta_ratio))
        
        # 2 sample t test
        extr_1960=ds_pf_gs_extrs.sel(hazard=extr,birth_year=1960).values
        extr_2020=ds_pf_gs_extrs.sel(hazard=extr,birth_year=2020).values
        result = sts.ttest_rel(
            extr_1960, 
            extr_2020,
            nan_policy='omit',
        )
        print('{} p value for difference of means: {}'.format(extr,result.pvalue))
        print('')
            
#%% ----------------------------------------------------------------
# print latex table of ensemble members per hazard and gmt pathway
# ------------------------------------------------------------------      

def print_latex_table(
    flags,
    df_GMT_strj,
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

    gmts = np.arange(6,25).astype('int')
    gmts2100 = np.round(df_GMT_strj.loc[2100,gmts].values,1)   
    gmt_dict = dict(zip(gmts,gmts2100))

    sims_per_step = {}
    for extr in extremes:
        
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)        

        sims_per_step[extr] = {}
        for step in gmts:
            sims_per_step[extr][step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[extr][step].append(i)      
                    
    headers = list(extremes_labels.values())
    data = {}
    for step in gmts:
        data[str(gmt_dict[step])] = [len(sims_per_step[extr][step]) for extr in extremes]

    textabular = f"l|{'r'*len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texdata = "\\hline\n"
    for label in sorted(data):
        if label == "z":
            texdata += "\\hline\n"
        texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

    print("\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata,end="")
    print("\\end{tabular}")                

#%% ----------------------------------------------------------------
# millions excess children between 1.5 and 2.7 deg warming by 2100
# living unprecedented exposure to events
# ------------------------------------------------------------------  

def print_millions_excess(
    flags,
    df_GMT_strj,
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

    gmts = np.arange(6,25).astype('int')
    gmts2100 = np.round(df_GMT_strj.loc[2100,gmts].values,1)   
    gmt_dict = dict(zip(gmts,gmts2100))
    sumlist=[]
    for extr in extremes:
        
        print(extr)
        with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)    
        with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)            

        sims_per_step = {}
        for step in gmts:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)  
        # millions children unprecedented in 1.5 pathway
        step=6
        unprec_15 = ds_pf_gs['unprec'].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].sum(dim='birth_year').mean(dim='run') / 10**6
        print('in 1.5 degree pathway, {} chidren live unprecedented exposure to {}'.format(np.around(unprec_15.item()),extr))
        
        # millions children unprecedented in 2.7 pathway
        step=17
        unprec_27 = ds_pf_gs['unprec'].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].sum(dim='birth_year').mean(dim='run') / 10**6
        print('in 2.7 degree pathway, {} children live unprecedented exposure to {}'.format(np.around(unprec_27.item()),extr))    
            
        # difference between 1.5 and 2.7 deg pathways
        print('{} more million children will live through unprecedented exposure to {}'.format(np.around((unprec_27.item()-unprec_15.item())),extr))
        print('')
        
        sumlist.append(np.around((unprec_27.item()-unprec_15.item())))
        
    print(np.sum(sumlist))
    
#%% ----------------------------------------------------------------
# ratio of pfs reporting
# ------------------------------------------------------------------  
def print_pf_ratios(
    df_GMT_strj,
    da_gs_popdenom,
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

    for extr in extremes:
        
        # looking across birth years (1960 to 2020 for current policies)
        pf_1960 = ds_pf_gs_extrs.loc[{
            'birth_year':1960,
            'GMT':17,
            'hazard':extr
        }].item()
        
        pf_2020 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':17,
            'hazard':extr
        }].item()    
        
        pf_2020_1960_ratio = np.around(pf_2020 / pf_1960,1)
        
        print('change in pf for {} in 2.7 degree \n scenario between 2020 and 1960 is {}'.format(extr,pf_2020_1960_ratio))
        
        # looking across GMTs for 2020
        pf15 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':6,
            'hazard':extr
        }].item()
        
        pf27 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':17,
            'hazard':extr
        }].item()    
        
        pf_27_15_ratio = np.around(pf27 / pf15,1)    
        
        print('change in pf for {} and 2020 cohort \n between 1.5 and 2.7 pathways is {}'.format(extr,pf_27_15_ratio))
        
        print('')    