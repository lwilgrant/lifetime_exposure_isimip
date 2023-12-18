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
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

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
    with open('./data/pickles_v2/gridscale_cohort_global.pkl', 'rb') as file:
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
        with open('./data/pickles_v2/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
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
    with open('./data/pickles_v2/gridscale_cohort_global.pkl', 'wb') as f:
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
        
        with open('./data/pickles_v2/{}/isimip_metadata_{}_ar6_rm.pkl'.format(extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)     
            
        n = 0
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/pickles_v2/{}/isimip_AFA_{}_{}.pkl'.format(extr,extr,str(i)), 'rb') as f:
                da_AFA = pk.load(f)           
            
            if n == 0:    
                da_sum = da_AFA.sum(dim='time').where(mask.notnull())
            else:
                da_sum = da_sum + da_AFA.sum(dim='time').where(mask.notnull())
            
            n+=1
            
        da_exposure_occurence = xr.where(da_sum>0,1,0)
        
        with open('./data/pickles_v2/{}/exposure_occurrence_{}.pkl'.format(extr,extr), 'wb') as file:
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
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
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
                    
                        with open('./data/pickles_v2/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(extr,extr,cntry,i,step), 'rb') as f:
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
                
            with open('./data/pickles_v2/{}/emergence_locs_perrun_{}_{}.pkl'.format(extr,extr,step), 'wb') as f:
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

    with open('./data/pickles_v2/gridscale_cohort_global.pkl', 'rb') as file:
        ds_gridscale_cohortsize = pk.load(file)   
        
    da_gridscale_cohortsize = ds_gridscale_cohortsize['cohort_size']

    # loop through extremes
    for extr in extremes:

        start_time = time.time()

        # first get all regions that have exposure to extr in ensemble
        with open('./data/pickles_v2/{}/exposure_occurrence_{}.pkl'.format(extr,extr), 'rb') as file:
            da_exposure_occurrence = pk.load(file)          

        # get metadata for extreme
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
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
            
            with open('./data/pickles_v2/{}/emergence_locs_perrun_{}_{}.pkl'.format(extr,extr,step), 'rb') as f:
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
        
        with open('./data/pickles_v2/{}/pf_geoconstrained_{}.pkl'.format(extr,extr), 'wb') as f:
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
        
        with open('./data/pickles_v2/{}/pf_geoconstrained_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_geoconstrained = pk.load(f)      
            
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)      
            
        # get metadata for extreme
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
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
        
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)               
        
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)         
        
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
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

def print_latex_table_ensemble_sizes(
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
        
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
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
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)    
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
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
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
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
        

#%% ----------------------------------------------------------------
# print number of unprecedented people
# ------------------------------------------------------------------          
        
def print_absolute_unprecedented(
    ds_pf_gs,     
):
    
    step=6
    by=2020
    unprec=ds_pf_gs['unprec'].sum(dim='country').loc[{'run':sims_per_step[step],'GMT':step, 'birth_year':by}].median(dim='run')
    print('{} million'.format(unprec.item()/10**6))               
    

#%% ----------------------------------------------------------------
# get pickle of cities that are valid for f1 concept plot
# ------------------------------------------------------------------  

def find_valid_cities(
     df_countries,
     da_cohort_size,
     countries_mask,
     countries_regions,
     d_isimip_meta,
     flags,
):
    if not os.path.isfile('./data/pickles_v2/valid_cities.pkl'):
        # excel file of cities, their coords and population
        df_cities = pd.read_excel('./data/city_locations/worldcities.xlsx')
        df_cities = df_cities.drop(columns=['city_ascii','iso2','iso3','admin_name','capital','id']).nlargest(n=200,columns=['population'])
        concept_bys = np.arange(1960,2021,30)

        # loop through countries
        cntry_concat = []
        for cntry in list(df_countries.index):
            
            print(cntry)
            da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
            da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
            da_cntry = xr.DataArray(
                np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                dims=countries_mask.dims,
                coords=countries_mask.coords,
            )
            da_cntry = da_cntry.where(da_cntry,drop=True)
            lat_weights = np.cos(np.deg2rad(da_cntry.lat))
            lat_weights.name = "weights" 

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
            with open('./data/pickles_v2/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
                ds_dmg = pk.load(f)   

            # load PIC pickle
            with open('./data/pickles_v2/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
                ds_pic = pk.load(f)                   

            # loop over simulations
            for i in list(d_isimip_meta.keys()): 

                # print('simulation {} of {}'.format(i,len(d_isimip_meta)))

                # load AFA data of that run
                with open('./data/pickles_v2/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
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

            # select country from excel database of city coords
            df_cntry = df_cities.loc[df_cities['country']==cntry].copy()
            df_cntry['valid'] = np.nan

            # loop through cities in country
            for city_i in list(df_cntry.index):   

                # get city info from 
                # city = df_cntry.loc[city_i,'city']
                city_lat = df_cntry.loc[city_i,'lat']
                city_lon = df_cntry.loc[city_i,'lng']

                # pic
                da_pic_city_9999 = ds_pic['99.99'].sel({'lat':city_lat,'lon':city_lon},method='nearest').item()            

                # mean for city            
                da_test_city = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').mean(dim='run')
                da_test_city = da_test_city.rolling(time=5,min_periods=5).mean()   

                # sequence booleans for showing that gmt+1 is greater than gmt (relevant for 2.5 and 3.5)
                sequence_bools = {}
                for i,gmt in enumerate(GMT_indices_plot):

                    sequence_bools[gmt] = []

                    for by in da_test_city.birth_year:

                        da_by = da_test_city.sel(birth_year=by).max(dim='time')
                        bool_entry = da_by.sel(GMT=gmt) > da_by.sel(GMT=GMT_indices_plot[i-1])
                        sequence_bools[gmt].append(bool_entry.item())

                # pre-industrial comparison to make sure 1960 lifetime exposure for 1.5, 2.5 and 3.5 is below pic 99.99
                pic_bools = []
                by=1960
                for i,gmt in enumerate(GMT_indices_plot):

                    da_by = da_test_city.sel(birth_year=by,GMT=gmt).max(dim='time')
                    bool_entry = da_by < da_pic_city_9999
                    pic_bools.append(bool_entry.item())        

                # check that sequence bools for 2.5 and 3.5 and pic bools are all True
                sequence_bools_highgmts = sequence_bools[15]+sequence_bools[24]
                all_bools = sequence_bools_highgmts + pic_bools
                if np.all(all_bools):
                    df_cntry.loc[city_i,'valid'] = True
                else:
                    df_cntry.loc[city_i,'valid'] = False

            # only keep cities that match criteria
            df_cntry = df_cntry.drop(df_cntry.index[df_cntry['valid']==False])
            cntry_concat.append(df_cntry)    

        df_valid_cities = pd.concat(cntry_concat)
        df_valid_cities = df_valid_cities.sort_values(by=['population'],ascending=False)
        print(df_valid_cities)    
        # pickle selection of cities
        with open('./data/pickles_v2/valid_cities.pkl', 'wb') as f:
            pk.dump(df_valid_cities,f)   
            
    else:
        
        with open('./data/pickles_v2/valid_cities.pkl', 'rb') as f:
            df_valid_cities = pk.load(f)        
            
            
#%% ----------------------------------------------------------------
# generating large latex tables on CF data per country, birth year and 1.5, 2.5 and 3.5 degree scenario
# ------------------------------------------------------------------

def print_latex_table_unprecedented(
    flags,
    da_gs_popdenom,
):

    # input
    bys=np.arange(1960,2021,10)
    # bys=np.arange(1960,2021,1)
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'wildfires',
        'cropfailedarea': 'crop failures',
        'driedarea': 'droughts',
        'floodedarea': 'floods',
        'heatwavedarea': 'heatwaves',
        'tropicalcyclonedarea': 'tropical cyclones',
    }  

    # data
    for extr in extremes:
        
        # open dictionary of metadata for sim means and CF data per extreme
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)     
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)           

        sims_per_step = {}
        sims_per_step[extr] = {}
        for step in GMT_indices_plot:
            sims_per_step[extr][step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[extr][step].append(i)      
        
        da_p_gs_plot = ds_pf_gs['unprec'].loc[{
            'GMT':GMT_indices_plot,
        }]
        df_list_gs = []
        for step in GMT_indices_plot:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[extr][step],'GMT':step}].mean(dim='run')
            da_cf_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom * 100
            df_cf_gs_plot_step = da_cf_gs_plot_step.to_dataframe(name='CF').reset_index()
            df_cf_gs_plot_step['P'] = da_p_gs_plot_step.to_dataframe(name='P').reset_index().loc[:,['P']] / 1000
            df_list_gs.append(df_cf_gs_plot_step)
        df_cf_gs_plot = pd.concat(df_list_gs)

        df_cf_gs_plot['CF'] = df_cf_gs_plot['CF'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P'] = df_cf_gs_plot['P'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P (CF)'] = df_cf_gs_plot.apply(lambda x: '{} ({})'.format(str(x.P),str(x.CF)), axis=1)

        # print latex per step
        for step in GMT_indices_plot:
            
            print('')
            print('Running latex print of CF for {} under {} pathway'.format(extremes_labels[extr],gmt_legend[step]))
            print('')
            
            df_latex = df_cf_gs_plot[df_cf_gs_plot['GMT']==step].copy()
            df_cntry_by = df_latex.loc[:,['country','birth_year','P (CF)']].set_index('country')

            for by in bys:
                df_cntry_by[by] = df_cntry_by[df_cntry_by['birth_year']==by].loc[:,['P (CF)']]
                
            df_cntry_by = df_cntry_by.drop(columns=['birth_year','P (CF)']).drop_duplicates() 

            # latex
            caption = '\\caption{{\\textbf{{Absolute population (in thousands) of cohorts living unprecedented exposure to {0} and CF\\textsubscript{{{0}}} (\\%) per country and birth year in a {1}\\degree C pathway}}}}\\\\'.format(extremes_labels[extr],gmt_legend[step])
            headers = list(df_cntry_by.columns.astype('str'))
            headers = ['Country'] + headers
            data = {}
            for row in list(df_cntry_by.index):
                if len(str(row).split()) > 1:
                    newrow = ' \\\ '.join(str(row).split())
                    newrow = '\makecell[l]{{{}}}'.format(newrow)    
                    data[str(newrow)] = list(df_cntry_by.loc[row,:].values)
                else:
                    data[str(row)] = list(df_cntry_by.loc[row,:].values)

            textabular = f" l |{' c '*(len(headers)-1)}"
            texheader = " & ".join(headers) + "\\\\"
            texdata = "\\hline\n"

            for label in data:
                if label == "z":
                    texdata += "\\hline\n"
                texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

            print('\\small')
            print('\\begin{longtable}{'+textabular+'}')
            print(caption)
            print(texheader)
            print(texdata,end='')
            print('\\end{longtable}')
            print('\\normalsize') 
            print('\\clearpage')             
#%% ----------------------------------------------------------------
# generating large latex tables on CF data per country, birth year and 1.5, 2.5 and 3.5 degree scenario
# ------------------------------------------------------------------

def print_latex_table_unprecedented_sideways(
    flags,
    da_gs_popdenom,
):

    # input
    # bys=np.arange(1960,2021,10)
    bys=np.arange(1960,2021,1)
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'wildfires',
        'cropfailedarea': 'crop failures',
        'driedarea': 'droughts',
        'floodedarea': 'floods',
        'heatwavedarea': 'heatwaves',
        'tropicalcyclonedarea': 'tropical cyclones',
    }  

    # data
    for extr in extremes:
        
        # open dictionary of metadata for sim means and CF data per extreme
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)     
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)           

        sims_per_step = {}
        sims_per_step[extr] = {}
        for step in GMT_indices_plot:
            sims_per_step[extr][step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[extr][step].append(i)      
        
        da_p_gs_plot = ds_pf_gs['unprec'].loc[{
            'GMT':GMT_indices_plot,
        }]
        df_list_gs = []
        for step in GMT_indices_plot:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[extr][step],'GMT':step}].mean(dim='run')
            da_cf_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom * 100
            df_cf_gs_plot_step = da_cf_gs_plot_step.to_dataframe(name='CF').reset_index()
            df_cf_gs_plot_step['P'] = da_p_gs_plot_step.to_dataframe(name='P').reset_index().loc[:,['P']] / 1000
            df_list_gs.append(df_cf_gs_plot_step)
        df_cf_gs_plot = pd.concat(df_list_gs)

        df_cf_gs_plot['CF'] = df_cf_gs_plot['CF'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P'] = df_cf_gs_plot['P'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P (CF)'] = df_cf_gs_plot.apply(lambda x: '{} ({})'.format(str(x.P),str(x.CF)), axis=1)

        # print latex per step
        for step in GMT_indices_plot:
            
            print('')
            print('Running latex print of CF for {} under {} pathway'.format(extremes_labels[extr],gmt_legend[step]))
            print('')
            
            df_latex = df_cf_gs_plot[df_cf_gs_plot['GMT']==step].copy()
            df_cntry_by = df_latex.loc[:,['country','birth_year','P (CF)']].set_index('country')

            for by in bys:
                df_cntry_by[by] = df_cntry_by[df_cntry_by['birth_year']==by].loc[:,['P (CF)']]
                
            df_cntry_by = df_cntry_by.drop(columns=['birth_year','P (CF)']).drop_duplicates() 

            # latex
            caption = '\\caption{{\\textbf{{Absolute population (in thousands) of cohorts living unprecedented exposure to {0} and CF\\textsubscript{{{0}}} (\\%) per country and birth year in a {1}\\degree C pathway}}}}\\\\'.format(extremes_labels[extr],gmt_legend[step])
            headers = list(df_cntry_by.columns.astype('str'))
            headers = ['\'{}'.format(y[2:]) for y in headers]
            headers = ['Country'] + headers
            data = {}
            for row in list(df_cntry_by.index):
                if len(str(row).split()) > 1:
                    newrow = ' \\\ '.join(str(row).split())
                    newrow = '\makecell[l]{{{}}}'.format(newrow)    
                    data[str(newrow)] = list(df_cntry_by.loc[row,:].values)
                else:
                    data[str(row)] = list(df_cntry_by.loc[row,:].values)

            textabular = f" l |{' c '*(len(headers)-1)}"
            texheader = " & ".join(headers) + "\\\\"
            texdata = "\\hline\n"

            for label in data:
                if label == "z":
                    texdata += "\\hline\n"
                texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

            print('\\small')
            print('\\begin{longtable}{'+textabular+'}')
            print(caption)
            print(texheader)
            print(texdata,end='')
            print('\\end{longtable}')
            print('\\normalsize') 
            print('\\clearpage')             
# %%

#%% ----------------------------------------------------------------
# testing f1 vs f4 inconsistency for belgium
# ------------------------------------------------------------------
def testing_f1_v_f4():
    cntry='Belgium'
    extr='heatwavedarea'
    step=6
    # brussels coords  
    city_lat = 50.8476
    city_lon = 4.3572   

    # get metadata for extreme
    with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
        d_isimip_meta = pk.load(f)
        
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        print('step {}'.format(step))
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)                

    step=6
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
                ['run','birth_year'],
                np.full(
                    (len(sims_per_step[step]),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),                          
        },
        coords={
            'birth_year': ('birth_year', birth_years),
            'run': ('run', sims_per_step[step]),
        }
    )                      

    # loop through sims and pick emergence masks for sims that are valid
    for i in sims_per_step[step]: 
        
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
        
            with open('./data/pickles_v2/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(extr,extr,cntry,i,step), 'rb') as f:
                da_birthyear_emergence_mask = pk.load(f)
                
            ds_cntry_emergence['emergence'].loc[{
                'run':i,
                'birth_year':birth_years,
            }] = da_birthyear_emergence_mask.sel({'lat':city_lat,'lon':city_lon},method='nearest')
            
    # compute mean for extreme - country - GMT, assign into greater dataset for eventual union
    da_loc_mean = ds_cntry_emergence['emergence'].loc[{
        'run':sims_per_step[step],
    }].mean(dim='run')
    
    da_emergence_mean.loc[{'hazard':'heatwavedarea','GMT':step}].sel({'lat':city_lat,'lon':city_lon},method='nearest')
    ds_cntry_emergence['emergence'].plot()
    # da_loc_mean.plot()


    # get metadata for extreme
    with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
        d_isimip_meta = pk.load(f)
        
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        print('step {}'.format(step))
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)   

    # get data
    cntry='Belgium'
    city_name='Brussels'
    # concept_bys = np.arange(1960,2021,30)
    concept_bys = np.arange(1960,2021,1)
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
    with open('./data/pickles_v2/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
        ds_dmg = pk.load(f)                  

    # loop over simulations
    for i in list(d_isimip_meta.keys()): 

        print('simulation {} of {}'.format(i,len(d_isimip_meta)))

        # load AFA data of that run
        with open('./data/pickles_v2/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
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
    da_test = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest')
    da_test = da_test.rolling(time=5,min_periods=5).mean()

    # fill in 1st 4 years with 1s
    # first for mean
    for by in da_test.birth_year.data:
        for step in GMT_indices_plot:
            for run in sims_per_step[step]:
                da_test.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5),'run':run}] = da_test.loc[{'birth_year':by,'GMT':step,'run':run}].min(dim='time')      
                
    # load PIC pickle
    with open('./data/pickles_v2/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
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
        for run in sims_per_step[step]:
            da_test.loc[{'birth_year':1960,'GMT':step,'run':run}].plot.line(
                ax=ax,
                color=colors[step],
                linewidth=1,
            )
        # # bold line for emergence
        # da = da_test.loc[{'birth_year':1960,'GMT':step}]
        # da = da.where(da>da_pic_city_9999)
        # da.plot.line(
        #     ax=ax,
        #     color=colors[step],
        #     linewidth=3,
        #     zorder=4,
        # )
                
    end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])     
    ax.annotate(
        'Born in 1960',
        (1965,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )    
    ax.set_title(None)
    ax.annotate(
        letters[l],
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
        xmax=da_test.loc[{'birth_year':1960}].time.max()+10, 
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
        for run in sims_per_step[step]:
            ax_pdf.hlines(
                y=da_test.loc[{'birth_year':1960,'GMT':step,'run':run}].max(), 
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
    ax2_l = 1990
    ax2_b = da_pic_city_9999 *2
    ax2_w = np.floor(df_life_expectancy_5.loc[1990,cntry])
    ax2_h = np.round(da_test.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())
    ax2 = ax.inset_axes(
        bounds=(ax2_l, ax2_b, ax2_w, ax2_h),
        transform=ax.transData,
    )

    for step in GMT_indices_plot:
        for run in sims_per_step[step]:
            da_test.loc[{'birth_year':1990,'GMT':step,'run':run}].plot.line(
                ax=ax2,
                color=colors[step],
                linewidth=1,
            )
        # # bold line for emergence
        # da = da_test.loc[{'birth_year':1990,'GMT':step}]
        # da = da.where(da>da_pic_city_9999)
        # da.plot.line(
        #     ax=ax2,
        #     color=colors[step],
        #     linewidth=3,
        #     zorder=4,
        # )    
                    
    end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
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
        np.round(da_test.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)  
    ax2.spines['left'].set_position(('data',1990)) 
    ax2.tick_params(colors='gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['bottom'].set_color('gray')

    ax2.annotate(
        'Born in 1990',
        (1995,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )
    ax2.set_title(None)
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

    # get time of first line to cross PIC thresh
    # emergences = []
    # for step in GMT_indices_plot:
    #     da = da_test.loc[{'birth_year':1990,'GMT':step}]
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
        for run in sims_per_step[step]:
            ax2_pdf.hlines(
                y=da_test.loc[{'birth_year':1990,'GMT':step,'run':run}].max(), 
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
        'Unprecedented\nlifetime\nexposure\nfor {} people'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':1990,'lat':city_lat,'lon':city_lon},method='nearest').item())))),
        (1.1,0.3),
        xycoords=ax2_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        # fontweight='bold',
    )             

    # ------------------------------------------------------------------   
    # 2020 time series
    ax3_l = 2020
    ax3_b = np.round(da_test.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    ax3_w = np.floor(df_life_expectancy_5.loc[2020,cntry])
    ax3_h = np.round(da_test.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())
    ax3 = ax2.inset_axes(
        bounds=(ax3_l, ax3_b, ax3_w, ax3_h),
        transform=ax2.transData,
    )
    # plot mean lines
    for step in GMT_indices_plot:
        for run in sims_per_step[step]:
            da_test.loc[{'birth_year':2020,'GMT':step,'run':run}].plot.line(
                ax=ax3,
                color=colors[step],
                linewidth=1,
            )
        # # bold line for emergence
        # da = da_test.loc[{'birth_year':2020,'GMT':step}]
        # da = da.where(da>da_pic_city_9999)
        # da.plot.line(
        #     ax=ax3,
        #     color=colors[step],
        #     linewidth=3,
        #     zorder=4,
        # )    

    end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])      

    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.set_yticks([0,5,10,15,20,25])
    ax3.set_yticklabels([None,5,10,15,20,25])   
    ax3.set_xticks(np.arange(2020,2101,10))      
    ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    ax3.set_xlim(
        2020,
        end_year,
    )
    ax3.set_ylim(
        0,
        np.round(da_test.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)  
    ax3.spines['left'].set_position(('data',2020))  
    ax3.tick_params(colors='gray')
    ax3.spines['left'].set_color('gray')
    ax3.spines['bottom'].set_color('gray')

    # # get time of first line to cross PIC thresh
    # emergences = []
    # for step in GMT_indices_plot:
    #     da = da_test.loc[{'birth_year':2020,'GMT':step}]
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
    ax3.annotate(
        'Born in 2020',
        (2025,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )
    ax3.set_title(None)
    ax3.annotate(
        letters[l],
        (2020,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    ) 
    l+=1      

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
        for run in sims_per_step[step]:
            ax3_pdf.hlines(
                y=da_test.loc[{'birth_year':2020,'GMT':step,'run':run}].max(), 
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
        'Unprecedented\nlifetime\nexposure\nfor {} people'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item())))),
        (1.1,0.6),
        xycoords=ax3_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        # fontweight='bold',
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
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    # f.savefig('./ms_figures/concept_{}_{}.png'.format(city_name,cntry),dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/concept_{}_{}.pdf'.format(city_name,cntry),dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/concept_{}_{}.eps'.format(city_name,cntry),format='eps',bbox_inches='tight')

    # population estimates
    # ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    # ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item()

    # # getting estimate of all birth years that emerge in 1.5 and 3.5 pathways and how many these cohorts sum to
    # valid_bys=da_test.birth_year.where(da_test.loc[{'GMT':6}].max(dim='time')>da_pic_city_9999)
    # y1 = valid_bys.min(dim='birth_year')
    # y2 = valid_bys.max(dim='birth_year')
    # unprecedented=ds_dmg['by_population_y0'].sel(birth_year=np.arange(y1,y2+1),lat=city_lat,lon=city_lon,method='nearest').sum(dim='birth_year').round().item()    
    # print('{} thousand unprecedented born in {} and later under pathway {}'.format(unprecedented/10**3,y1,6))

    # valid_bys=da_test.birth_year.where(da_test.loc[{'GMT':24}].max(dim='time')>da_pic_city_9999)
    # y1 = valid_bys.min(dim='birth_year')
    # y2 = valid_bys.max(dim='birth_year')
    # unprecedented=ds_dmg['by_population_y0'].sel(birth_year=np.arange(y1,y2+1),lat=city_lat,lon=city_lon,method='nearest').sum(dim='birth_year').round().item()    
    # print('{} thousand unprecedented born in {} and later under pathway {}'.format(unprecedented/10**3,y1,24))        

