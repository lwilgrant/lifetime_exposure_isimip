# ---------------------------------------------------------------
# Functions to compute exposure
# ----------------------------------------------------------------


import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle as pk
from scipy import interpolate
import regionmask
import glob
import time
import matplotlib.pyplot as plt
from settings import *

#%% ----------------------------------------------------------------
# function to compute extreme event exposure across a person's lifetime (mf_exposure.m)
def calc_life_exposure(
    df_life_expectancy_5, 
    df_countries, 
    df_birthyears, 
    d_exposure_peryear,
):

    # to store exposure per life for every run
    exposure_birthyears = []

    for j, country in enumerate(df_countries['name']):

        # initialise birth years 
        exposure_birthyears_percountry = np.empty(len(df_birthyears))

        for i, birth_year in enumerate(df_birthyears.index): 

            # ugly solution to deal with similar years of life expectancy - to be solved more elegantly. 
            life_expectancy_5 = df_life_expectancy_5.loc[birth_year, country] 
            # if np.size(life_expectancy_5) > 1:  # note fr Luke; not necessary now because duplicate indices were removed from df_life_ex... before interpolating (kept native unwpp years)
            #     life_expectancy_5 = life_expectancy_5.iloc[0]

            # define death year based on life expectancy
            death_year = birth_year + np.floor(life_expectancy_5)

            # integrate exposure over full years lived
            exposure_birthyears_percountry[i] = d_exposure_peryear[country].sel(time=slice(birth_year,death_year)).sum().values
            
            # add exposure during last (partial) year
            exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + \
                d_exposure_peryear[country].sel(time=death_year+1).sum().values * \
                    (life_expectancy_5 - np.floor(life_expectancy_5))
        

        if j == 0: # ugly - solve better!
            exposure_birthyears = exposure_birthyears_percountry
        else: 
            exposure_birthyears = np.vstack([exposure_birthyears, exposure_birthyears_percountry])

    df_exposure_perlife = pd.DataFrame(
        exposure_birthyears.transpose(),
        index=df_birthyears.index, 
        columns=df_countries.index,
    )
    
                # df_life_expectancy_5,
                # df_countries,
                # df_birthyears,
                # {country: da[ind_RCP2GMT_15] for country, da in d_exposure_peryear_percountry.items()},
    
    return df_exposure_perlife

#%% ----------------------------------------------------------------
# calculated weighted fieldmean per country mask
def calc_weighted_fldmean(
    da, 
    weights, 
    countries_mask, 
    ind_country, 
    flag_region,
):

    # one country provided, easy masking
    if not flag_region : 
        da_masked = da.where(countries_mask == ind_country)
    
    # if more countries are provided, combine the different masks - THIS IS VERY TIME-INEFFICIENT!
    else: 
        # initialise the data array with the first country mask
        # comment from Luke: this doesn't need the numpy conversion and the np.nan_to_num conversion (read online that its costly because of infinity conversions)
        # mask_sum = da.where(countries_mask == ind_country[0]).values

        # # then add the masks for the other countries
        # for ind in ind_country[1:]: 

        #     mask_sum = np.nan_to_num(mask_sum) + np.nan_to_num(da.where(countries_mask == ind).values)
        
        # # turn mask_sum into data array
        # da_masked = xr.DataArray(mask_sum,coords=da.coords,dims=da.dims)
        # da_masked = da_masked.where(da_masked>0, np.nan)
        
        # proposed substitute to the np.nan_to_num approach
        if len(ind_country) > 1:
        
            da_masked = xr.DataArray(
                np.in1d(da,ind_country).reshape(da.shape),
                dims=da.dims,
                coords=da.coords,
            )
    
    da_weighted_fldmean = da_masked.weighted(weights).mean(dim=("lat", "lon"))
    
    return da_weighted_fldmean

                    # da_AFA_pic, 
                    # da_population[0,:,:], 
                    # countries_mask, 
                    # ind_country, 
                    # flag_region= False

#%% ----------------------------------------------------------------
# calculated weighted fieldmean combining country masks per region
def calc_weighted_fldmean_region(
    da, 
    weights, 
    member_countries, 
    countries_mask, 
    ind_country,
):

    da_masked = da.where(countries_mask == ind_country)
    da_weighted_fldmean = da_masked.weighted(weights).mean(dim=("lat", "lon"))
    
    return da_weighted_fldmean

#%% ----------------------------------------------------------------
# get member countries per region
def get_countries_of_region(
    region, 
    df_countries,
): 

    # Get list of member countries from region
    member_countries = df_countries.loc[df_countries['region']==region]['name'].values

    # not region but income group
    if len(member_countries) == 0: 
        member_countries = df_countries.loc[df_countries['incomegroup']==region]['name'].values

    # get all countries for the world
    if region == 'World':
        member_countries = df_countries['name'].values

    return member_countries    

#%% ----------------------------------------------------------------
# function to compute multi-model mean across ISIMIP simulations based on mf_exposure_mmm.m
def calc_exposure_mmm(
    exposure, 
    extremes, 
    d_isimip_meta,
):


    # define dictionaries
    d_exposure_mmm = {}
    d_exposure_mms = {}
    d_exposure_q25 = {}
    d_exposure_q75 = {}

    # turn meta dictionary into list
    l_isimip_meta =  list(d_isimip_meta.values())


    # loop over extremes
    for extreme in extremes:
        
        # get indices of runs for extreme
        indices_extremes = []
        for metadata in l_isimip_meta:
            if metadata['extreme'] == extreme: 
                indices_extremes.append(l_isimip_meta.index(metadata)+1)


        # Stack all dictionary values of different extremes
        to_stack = []

        for ind_run in [1,2]: #indices_extremes: 
            to_stack.append(exposure[ind_run].values)

        stacked_perextreme = np.stack(to_stack, axis=0)


        # compute multi-model mean (mmm) along different runs and save as dataframe in dictionary
        d_exposure_mmm[extreme] = pd.DataFrame(
            np.nanmean(stacked_perextreme, axis=0), 
            columns=exposure[1].columns, 
            index=exposure[1].index,
        )


        # compute multi-model standard deviation (mms) per extreme impact category
        if len(indices_extremes)> 1:  
            d_exposure_mms[extreme]  = pd.DataFrame(
                np.nanmean(stacked_perextreme, axis=0), 
                columns=exposure[1].columns, 
                index=exposure[1].index,
            )
        else:
            d_exposure_mms[extreme] = pd.DataFrame(
                np.zeros_like(exposure[1].values), 
                columns=exposure[1].columns, 
                index=exposure[1].index,
            )


        # compute multi-model IQR (q25 and q75) per extreme impact category
        d_exposure_q25[extreme] = pd.DataFrame(
            np.quantile(stacked_perextreme, 0.25, axis=0), 
            columns=exposure[1].columns, 
            index=exposure[1].index,
        )
        d_exposure_q75[extreme] = pd.DataFrame(
            np.quantile(stacked_perextreme, 0.75,  axis=0), 
            columns=exposure[1].columns, 
            index=exposure[1].index,
        )


        # geometric means - not calculated here

    return d_exposure_mmm, d_exposure_mms, d_exposure_q25, d_exposure_q75    

#%% ----------------------------------------------------------------
# function computing the Exposure Multiplication Factor (EMF)
def calc_exposure_EMF(
    d_exposure_mmm, 
    d_exposure_q25, 
    d_exposure_q75, 
    exposure_ref,
):

    # intialise dictionaries of EMF per extreme
    d_EMF_mmm = {}
    d_EMF_q25 = {}
    d_EMF_q75 = {}

    # loop over all extremes
    for extreme in d_exposure_mmm:

        # Check whether EMFs need to be computed
        if len(d_exposure_mmm[extreme]) == len(birth_years): 

            # define the year of the reference age
            ref_age_year = birth_years[np.where(ages==age_ref)][0]

            # repeat ref for ref age to divide 
            df_exposure_ref_repeated = pd.DataFrame(
                np.tile(exposure_ref[extreme].loc[ref_age_year].values, (len(birth_years), 1)), 
                columns=exposure_ref[extreme].columns, 
                index=exposure_ref[extreme].index
            )
                
            # compute Exposure Multiplication Factor (EMF) maps
            d_EMF_mmm[extreme] = d_exposure_mmm[extreme] / df_exposure_ref_repeated
        

            # get IQR EMFs - note that we use the q25/75 of the young age but the mmm of the reference age
            d_EMF_q25[extreme] = d_exposure_q25[extreme] / df_exposure_ref_repeated
            d_EMF_q75[extreme] = d_exposure_q75[extreme] / df_exposure_ref_repeated


            # set EMFs for inf to 100 
            d_EMF_mmm[extreme].replace(np.inf,100,inplace=True)
        
            # for all extremes, get the geometric mean of the EMFs - for the main paper figures
            # not translated
            # EMF_mmm(nextremes+1,:,:) = geomean(EMF_mmm(1:nextremes,:,:), 1, 'omitnan');

        
            # for all extremes, get the harmonic mean of the EMFs - for SI sensitivity plot
            # EMF_mmm_harmmean(nextremes+1,:,:) = harmmean(EMF_mmm(1:nextremes,:,:), 1, 'omitnan');


        # for all extremes, get the EMF of the sum of exposure across categories - for SI sensitivity plot
        # not translated
        #EMF_mmm_geomexp = exposure_mmm ./ repmat(exposure_ref(:,:, ages == age_ref), 1, 1, nbirthyears);   

            
        # for all extremes, get the geometric mean of the IQR of the EMFs - for the uncertainty bands in fig. 3
        # not translated
        # EMF_q25(nextremes+1,:,:) = geomean(EMF_q25(1:nextremes,:,:), 1, 'omitnan');
        # EMF_q75(nextremes+1,:,:) = geomean(EMF_q75(1:nextremes,:,:), 1, 'omitnan');

        
        else: 
        
            # do not compute EMFS
            d_EMF_mmm[extreme] = []
            d_EMF_q25[extreme] = []
            d_EMF_q75[extreme] = []
        
    return d_EMF_mmm, d_EMF_q25, d_EMF_q75

#%% ----------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span
def calc_exposure(
    grid_area,
    d_regions,
    d_isimip_meta, 
    df_birthyears_regions, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    da_population, 
    df_life_expectancy_5, 
    df_birthyears, 
    df_GMT_15, 
    df_GMT_20, 
    df_GMT_NDC,
):

        # initialise dicts
        d_RCP2GMT_maxdiff_15      = {}
        d_RCP2GMT_maxdiff_20      = {}
        d_RCP2GMT_maxdiff_NDC     = {}
        d_RCP2GMT_maxdiff_R26eval = {}

        d_exposure_perrun_RCP     = {}
        d_exposure_perrun_15      = {}
        d_exposure_perrun_20      = {}
        d_exposure_perrun_NDC     = {}
        d_exposure_perrun_R26eval = {} 
        
        d_landfrac_peryear_perregion = {}
        
        # unpack region information
        df_birthyears_regions = d_regions['birth_years']
        d_cohort_weights_regions = d_regions['cohort_size']        

        # loop over simulations
        for i in list(d_isimip_meta.keys()): 

            print('simulation '+str(i)+ ' of '+str(len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/pickles/isimip_AFA_{}_{}.pkl'.format(d_isimip_meta[i]['extreme'],str(i)), 'rb') as f:
                da_AFA = pk.load(f)

            # Get ISIMIP GMT indices closest to GMT trajectories        
            RCP2GMT_diff_15 = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=1)
            RCP2GMT_diff_20 = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=1)
            RCP2GMT_diff_NDC = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=1)
            RCP2GMT_diff_R26eval = np.min(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=1)

            ind_RCP2GMT_15 = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=1)
            ind_RCP2GMT_20 = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=1)
            ind_RCP2GMT_NDC = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=1)
            ind_RCP2GMT_R26eval = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=1)

            # Get maximum T difference between RCP and GMT trajectories (to remove rows later)
            d_RCP2GMT_maxdiff_15[i] = np.nanmax(RCP2GMT_diff_15)
            d_RCP2GMT_maxdiff_20[i] = np.nanmax(RCP2GMT_diff_20)
            d_RCP2GMT_maxdiff_NDC[i] = np.nanmax(RCP2GMT_diff_NDC)
            d_RCP2GMT_maxdiff_R26eval[i] = np.nanmax(RCP2GMT_diff_R26eval)

            # --------------------------------------------------------------------
            # per country 

            # initialise dicts
            d_exposure_peryear_percountry = {}

            # get spatial average
            for j, country in enumerate(df_countries['name']):

                print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')
                
                # calculate mean per country weighted by population
                ind_country = countries_regions.map_keys(country)

                # historical + RCP simulations
                d_exposure_peryear_percountry[country] = calc_weighted_fldmean( 
                    da_AFA,
                    da_population, 
                    countries_mask, 
                    ind_country, 
                    flag_region= False,
                )

            # call function to compute extreme event exposure per country and per lifetime
            d_exposure_perrun_RCP[i] = calc_life_exposure(
                df_life_expectancy_5, 
                df_countries, 
                df_birthyears, 
                d_exposure_peryear_percountry,
            )

            # calculate exposure for GMTs, replacing d_exposure_perrun_RCP by indexed dictionary according to corresponding GMTs with ISIMIP.
            d_exposure_perrun_15[i] = calc_life_exposure(
                df_life_expectancy_5,
                df_countries,
                df_birthyears,
                {country: da[ind_RCP2GMT_15].assign_coords(time=np.arange(year_start,year_end+1)) for country, da in d_exposure_peryear_percountry.items()},
            )
            d_exposure_perrun_20[i] = calc_life_exposure(
                df_life_expectancy_5,
                df_countries,
                df_birthyears,
                {country: da[ind_RCP2GMT_20].assign_coords(time=np.arange(year_start,year_end+1)) for country, da in d_exposure_peryear_percountry.items()},
            )
            d_exposure_perrun_NDC[i] = calc_life_exposure(
                df_life_expectancy_5,
                df_countries,
                df_birthyears,
                {country: da[ind_RCP2GMT_NDC].assign_coords(time=np.arange(year_start,year_end+1)) for country, da in d_exposure_peryear_percountry.items()},
            )
            d_exposure_perrun_R26eval[i] = calc_life_exposure(
                df_life_expectancy_5,
                df_countries,
                df_birthyears,
                {country: da[ind_RCP2GMT_R26eval].assign_coords(time=np.arange(year_start,year_end+1)) for country, da in d_exposure_peryear_percountry.items()},
            )
            
            # --------------------------------------------------------------------
            # per region
            #  

            print('')

            # initialise dictionaries
            d_landfrac_peryear_perregion[i] = {}
            d_exposure_perregion_perrun_RCP = {}

            # loop over regions
            for k, region in enumerate(df_birthyears_regions.columns): 
                
                print('processing region '+str(k+1)+' of '+str(len(df_birthyears_regions.columns)), end='\r')

                # initialise dict
                d_exposure_perregion_RCP = {}

                # Get list of member countries from region - with seperate treatment for world (luke: now inside get_countries_of_regions func)
                member_countries = get_countries_of_region(region, df_countries)
        
                # get spatial average of landfraction: historical + RCP simulations
                ind_countries = countries_regions.map_keys(member_countries)

                print('calculating landfrac')
                d_landfrac_peryear_perregion[i][region] = calc_weighted_fldmean(
                    da_AFA, 
                    grid_area, 
                    countries_mask, 
                    ind_countries, 
                    flag_region=True,
                )

                print('calculating cohort weights')
                # filter cohort weights to only keep countries within mask 
                d_cohort_weights_regions[region] = d_cohort_weights_regions[region].loc[:,d_cohort_weights_regions[region].columns.isin(df_countries.index)]
                
                # get weighted spatial average for all member countries per region
                d_exposure_perregion_RCP[region]   = (d_exposure_perrun_RCP[i].loc[:,member_countries] * d_cohort_weights_regions[region].values).sum(axis=1) /\
                    np.nansum(d_cohort_weights_regions[region].values, axis=1)

            # save exposures for every run
            d_exposure_perregion_perrun_RCP[i]  = d_exposure_perregion_RCP

        # --------------------------------------------------------------------
        # save workspave in pickles
        #  

        # save pickles
        print()
        print('Saving processed exposures')

        # pack exposure information
        d_exposure = {
            'exposure_perrun_RCP' : d_exposure_perrun_RCP, 
            'exposure_perrun_15' : d_exposure_perrun_15,
            'exposure_perrun_20' : d_exposure_perrun_20,
            'exposure_perrun_NDC' : d_exposure_perrun_NDC,
            'exposure_perregion_perrun_RCP' : d_exposure_perregion_perrun_RCP, 
            'landfrac_peryear_perregion' : d_landfrac_peryear_perregion 
        }

        with open('./data/pickles/exposure_{}.pkl'.format(d_isimip_meta[i]['extreme']), 'wb') as f:
            pk.dump(d_exposure,f)

        return d_exposure_perrun_RCP, d_exposure_perregion_perrun_RCP,
    
#%% ----------------------------------------------------------------
# convert PIC Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span
def calc_exposure_pic(
    grid_area,
    d_regions,
    d_pic_meta, 
    df_birthyears_regions, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    da_population, 
    df_life_expectancy_5, 
    df_birthyears, 
):

        d_exposure_perrun_pic     = {}       
        d_landfrac_peryear_perregion_pic = {}
        
        # unpack region information
        df_birthyears_regions = d_regions['birth_years']
        d_cohort_weights_regions = d_regions['cohort_size']                
        
        # loop over simulations
        for i in list(d_pic_meta.keys()): 

            print('simulation '+str(i)+ ' of '+str(len(d_pic_meta)))

            # load AFA data of that run
            with open('./data/pickles/isimip_AFA_pic_{}_{}.pkl'.format(d_pic_meta[i]['extreme'],str(i)), 'rb') as f:
                da_AFA_pic = pk.load(f)

            # --------------------------------------------------------------------
            # per country 
            
            d_exposure_peryear_percountry_pic = {}
            # get spatial average
            for j, country in enumerate(df_countries['name']):

                print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')
                # calculate mean per country weighted by population
                ind_country = countries_regions.map_keys(country)

                # corresponding picontrol - assume constant 1960 population density (this line takes about 16h by itself)
                d_exposure_peryear_percountry_pic[country] = calc_weighted_fldmean(
                    da_AFA_pic, 
                    da_population[0,:,:], # earliest year used for weights
                    countries_mask, 
                    ind_country, 
                    flag_region= False,
                )

            # call function to compute extreme event exposure per country and per lifetime
            d_exposure_perrun_pic[i] = calc_life_exposure(
                df_life_expectancy_5, 
                df_countries, 
                df_birthyears, 
                d_exposure_peryear_percountry_pic,
            )

        
            # --------------------------------------------------------------------
            # per region
            #  

            print('')

            # initialise dictionaries
            d_landfrac_peryear_perregion_pic[i] = {}
            d_exposure_perregion_perrun_pic = {}

            # loop over regions
            for k, region in enumerate(df_birthyears_regions.columns): 
                
                print('processing region '+str(k+1)+' of '+str(len(df_birthyears_regions.columns)), end='\r')

                # initialise dict
                d_exposure_perregion_pic = {}

                # Get list of member countries from region - with seperate treatment for world (luke: now inside get_countries_of_regions func)
                member_countries = get_countries_of_region(region, df_countries)
        
                # get spatial average of landfraction: historical + RCP simulations
                ind_countries = countries_regions.map_keys(member_countries)

                print('calculating landfrac')
                d_landfrac_peryear_perregion_pic[i][region] = calc_weighted_fldmean(
                    da_AFA_pic, 
                    grid_area, 
                    countries_mask, 
                    ind_countries, 
                    flag_region=True,
                )

                print('calculating cohort weights')
                # filter cohort weights to only keep countries within mask 
                d_cohort_weights_regions[region] = d_cohort_weights_regions[region].loc[:,d_cohort_weights_regions[region].columns.isin(df_countries.index)]
                
                # get weighted spatial average for all member countries per region
                d_exposure_perregion_pic[region]   = (d_exposure_perrun_pic[i].loc[:,member_countries] * d_cohort_weights_regions[region].values).sum(axis=1) /\
                    np.nansum(d_cohort_weights_regions[region].values, axis=1)

            # save exposures for every run
            d_exposure_perregion_perrun_pic[i]  = d_exposure_perregion_pic

        # --------------------------------------------------------------------
        # save workspave in pickles
        #  


        # save pickles
        print()
        print('Saving processed exposures')

        # pack region information
        d_exposure = {
            'exposure_perrun_RCP' : d_exposure_perrun_pic, 
            'exposure_perregion_perrun_RCP' : d_exposure_perregion_perrun_pic, 
            'landfrac_peryear_perregion' : d_landfrac_peryear_perregion_pic 
        }

        with open('./data/pickles/exposure_pic_{}.pkl'.format(d_pic_meta[i]['extreme']), 'wb') as f:
            pk.dump(d_exposure,f)

        return d_exposure_perrun_pic, d_exposure_perregion_perrun_pic

