# ---------------------------------------------------------------
# Functions to load and manipulate data
# ----------------------------------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle as pk
from scipy import interpolate
import regionmask
import glob

from settings import *
init()

# ---------------------------------------------------------------
# 1. Functions to load (see ms_load.m)
# ----------------------------------------------------------------

#%% ----------------------------------------------------------------
# Load observational data
def load_worldbank_unwpp_data():

    # load World Bank life expectancy at birth data (source: https://data.worldbank.org/indicator/SP.DYN.LE00.IN) - not used in final analysis
    worldbank_years        = np.arange(1960,2018) 
    
    df_worldbank = pd.read_excel('./data/world_bank/world_bank_life_expectancy_by_country.xls', header=None)
    worldbank_country_data = df_worldbank.iloc[:,4:].values
    worldbank_country_meta = df_worldbank.iloc[:,:4].values
    
    df_worldbank_country = pd.DataFrame(
        data=worldbank_country_data.transpose(), 
        index=worldbank_years, 
        columns=worldbank_country_meta[:,0]
    )

    df_worldbank_regions   = pd.read_excel(
        './data/world_bank/world_bank_life_expectancy_by_country.xls', 
        'world regions', 
        header=None
    )
    
    worldbank_region_data  = df_worldbank_regions.iloc[:,2:].values
    worldbank_region_meta  = df_worldbank_regions.iloc[:,:2].values
    
    df_worldbank_region    = pd.DataFrame(
        data=worldbank_region_data.transpose(), 
        index=worldbank_years, 
        columns=worldbank_region_meta[:,0]
    )

    # convert metadata in usable dataframe (the original code for this is in ms_manip.m) 
    df_countries = pd.DataFrame(worldbank_country_meta,columns=['name','abbreviation','region','incomegroup']).set_index('name')
    df_regions = pd.DataFrame(worldbank_region_meta,columns=['name','abbreviation']).set_index('name')

    # load United Nations life expectancy at age 5 data, defined as years left to live (source: https://population.un.org/wpp/Download/Standard/Mortality/)
    unwpp_years = np.arange(1952,2017+5,5)  # assume block is 5 instead of reported 6 years to avoid overlap and take middle of that 5-year block (so 1952 for period 1950-1955). Substract 5 to get birth year of 5-year old (a 5-year old in 1952 was born in 1947 and we need the latter). hard coded from 'WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES_orig.xls'

    df_unwpp = pd.read_excel('./data/UN_WPP/WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx',header=None)
    unwpp_country_data = df_unwpp.values[:,4:]
    
    df_unwpp_country = pd.DataFrame(
        data=unwpp_country_data.transpose(), 
        index=unwpp_years, 
        columns=worldbank_country_meta[:,0]
    )

    df_unwpp_region_raw =  pd.read_excel(
        './data/UN_WPP/WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx', 
        'world regions', 
        header=None
    )
    
    unwpp_region_data = df_unwpp_region_raw.values[:,2:]
    
    df_unwpp_region = pd.DataFrame(
        data=unwpp_region_data.transpose(), 
        index=unwpp_years, 
        columns=worldbank_region_meta[:,0]
    )
    
    # manually adjust country names with accent problems
    correct_names = {
        'CÃ´te d\'Ivoire' : 'Côte d\Ivoire', 
        'SÃ£o TomÃ© and Principe' : 'São Tomé and Principe'
    }

    df_worldbank_country.rename(columns=correct_names, inplace=True)
    df_unwpp_country.rename(columns=correct_names, inplace=True)
    df_countries.rename(index=correct_names, inplace=True)


    # bundle for communicaton
    meta = (df_countries, df_regions)
    worldbank = (df_worldbank_country, df_worldbank_region)
    unwpp = (df_unwpp_country, df_unwpp_region)
    
    return meta, worldbank, unwpp

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
# Load global mean temperature projections
def load_GMT(
    year_start,
    year_end,
):

    # Load global mean temperature projections from SR15
    df_GMT_SR15 = pd.read_excel('./data/temperature_trajectories_SR15/GMT_50pc_manualoutput_4pathways.xlsx', header=1);
    df_GMT_SR15 = df_GMT_SR15.iloc[:4,1:].transpose().rename(columns={
        0 : 'IPCCSR15_IMAGE 3.0.1_SSP1-26_GAS', 
        1 : 'IPCCSR15_MESSAGE-GLOBIOM 1.0_ADVANCE_INDC_GAS',
        2 : 'IPCCSR15_MESSAGE-GLOBIOM 1.0_SSP2-19_GAS',
        3 : 'IPCCSR15_MESSAGEix-GLOBIOM 1.0_LowEnergyDemand_GAS'
    })

    if np.nanmax(df_GMT_SR15.index) < year_end: 
        # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099)
        GMT_last_10ymean = df_GMT_SR15.iloc[-10:,:].mean()
        for year in range(np.nanmax(df_GMT_SR15.index),year_end+1): 
            df_GMT_SR15 = pd.concat([df_GMT_SR15, pd.DataFrame(GMT_last_10ymean).transpose().rename(index={0:year})])

    # cut to analysis years
    df_GMT_15     = df_GMT_SR15.loc[year_start:year_end,'IPCCSR15_MESSAGEix-GLOBIOM 1.0_LowEnergyDemand_GAS']
    df_GMT_20     = df_GMT_SR15.loc[year_start:year_end,'IPCCSR15_IMAGE 3.0.1_SSP1-26_GAS']
    df_GMT_NDC    = df_GMT_SR15.loc[year_start:year_end,'IPCCSR15_MESSAGE-GLOBIOM 1.0_ADVANCE_INDC_GAS']

    # check and drop duplicate years
    df_GMT_15   = df_GMT_15[~df_GMT_15.index.duplicated(keep='first')]
    df_GMT_20   = df_GMT_20[~df_GMT_20.index.duplicated(keep='first')]
    df_GMT_NDC   = df_GMT_NDC[~df_GMT_NDC.index.duplicated(keep='first')]
    df_GMT_SR15 = df_GMT_SR15[~df_GMT_SR15.index.duplicated(keep='first')]

    return df_GMT_15, df_GMT_20, df_GMT_NDC

#%% ----------------------------------------------------------------
# Load SSP population totals
def load_population(
    year_start,
    year_end,
):

    # load 2D model constants
    da_population_histsoc = xr.open_dataset('./data/isimip/population/population_histsoc_0p5deg_annual_1861-2005.nc4', decode_times=False)['number_of_people'] 
    da_population_ssp2soc = xr.open_dataset('./data/isimip/population/population_ssp2soc_0p5deg_annual_2006-2100.nc4', decode_times=False)['number_of_people'] 

    # manually adjust time dimension in both data arrays (because original times could not be decoded)
    da_population_histsoc['time'] = np.arange(1861,2006)
    da_population_ssp2soc['time'] = np.arange(2006,2101)
    # concatenate historical and future data
    da_population = xr.concat([da_population_histsoc, da_population_ssp2soc], dim='time') 


    # if needed, repeat last year until entire period of interest is covered
    if np.nanmax(da_population.time) < year_end:
        population_10y_mean = da_population.loc[-10:,:,:].mean(dim='time').expand_dims(dim='time',axis=0) # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099)
        for year in range(np.nanmax(da_population.time)+1,year_end+1): 
            da_population = xr.concat([da_population,population_10y_mean.assign_coords(time = [year])], dim='time')

    # retain only period of interest
    da_population = da_population.sel(time=slice(year_start,year_end))

    return da_population

#%% ----------------------------------------------------------------
# Load ISIMIP model data
def load_isimip(
    flags_run, 
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
): 
    
    if flags_run: 

        print('Processing isimip')

        # initialise counter, metadata dictionary, pic list, pic meta, and 
        i = 1
        d_isimip_meta = {}
        pic_list = []
        d_pic_meta = {}

        # loop over extremes
        for extreme in extremes:

            # define all models
            models = model_names[extreme]

            # loop over models
            for model in models: 

                # store all files starting with model name
                file_names = glob.glob('./data/isimip/'+extreme+'/'+model.lower()+'/'+model.lower()+'*rcp*landarea*2099*')

                for file_name in file_names: 

                    print('Loading '+file_name.split('\\')[-1]+' ('+str(i)+')')

                    # load rcp data (AFA: Area Fraction Affected) - and manually add correct years
                    da_AFA_rcp = open_dataarray_isimip(file_name)

                    # save metadata
                    d_isimip_meta[i] = {
                        'model': file_name.split('_')[0].split('\\')[-1], 
                        'gcm': file_name.split('_')[1], 
                        'rcp': file_name.split('_')[2],                         
                        'extreme': file_name.split('_')[3], 
                    }

                    #load associated historical variable
                    file_name_his = glob.glob('./data/isimip/'+extreme+'/'+model.lower()+'/'+model.lower()+'*'+d_isimip_meta[i]['gcm']+'*_historical_*landarea*')[0]
                    da_AFA_his = open_dataarray_isimip(file_name_his)

                    # load GMT for rcp and historical period - note that these data are in different files
                    file_names_gmt = glob.glob('./data/isimip/DerivedInputData/globalmeans/tas/'+d_isimip_meta[i]['gcm'].upper()+'/*.fldmean.yearmean.txt') # ignore running mean files
                    file_name_gmt_fut = [s for s in file_names_gmt if d_isimip_meta[i]['rcp'] in s]
                    file_name_gmt_his = [s for s in file_names_gmt if '_historical_' in s]
                    file_name_gmt_pic = [s for s in file_names_gmt if '_piControl_' in s]

                    GMT_fut = pd.read_csv(
                        file_name_gmt_fut[0],
                        delim_whitespace=True,
                        skiprows=1,
                        header=None).rename(columns={0:'year',1:'tas'}).set_index('year')
                    GMT_his = pd.read_csv(
                        file_name_gmt_his[0],
                        delim_whitespace=True, 
                        skiprows=1, 
                        header=None).rename(columns={0:'year',1:'tas'}).set_index('year')
                    GMT_pic = pd.read_csv(
                        file_name_gmt_pic[0],
                        delim_whitespace=True, 
                        skiprows=1, 
                        header=None).rename(columns={0:'year',1:'tas'}).set_index('year')

                    # concatenate historical and future data
                    da_AFA = xr.concat([da_AFA_his,da_AFA_rcp], dim='time')
                    df_GMT = pd.concat([GMT_his,GMT_fut])

                    # convert GMT from absolute values to anomalies - use data from pic until 1861 and from his from then onwards
                    df_GMT = df_GMT - pd.concat([GMT_pic.loc[year_start_GMT_ref:np.min(GMT_his.index)-1,:], GMT_his.loc[:year_end_GMT_ref,:]]).mean()

                    # if needed, repeat last year until entire period of interest is covered
                    if da_AFA.time.max() < year_end: 
                        # line below was fixed; supposed to be average of last 10 years, but we only selected last year
                        # da_AFA_lastyear = da_AFA.sel(time=da_AFA.time.max()).expand_dims(dim='time',axis=0) # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099)
                        da_AFA_lastyear = da_AFA.sel(time=slice(da_AFA.time.max()-9,da_AFA.time.max())).mean(dim='time').expand_dims(dim='time',axis=0)
                        # also adapted line below for GMTs
                        # GMT_lastyear = df_GMT.iloc[-1:,:]
                        GMT_lastyear = df_GMT.iloc[-10:,:].mean()

                        for year in range(da_AFA.time.max().values+1,year_end+1): 
                            da_AFA = xr.concat([da_AFA,da_AFA_lastyear.assign_coords(time = [year])], dim='time')
                            if len(df_GMT) < 439: # necessary to avoid this filling from 2100-2113 if GMTs already go to 2299
                                df_GMT = pd.concat([df_GMT,pd.DataFrame(data={'tas':GMT_lastyear['tas']},index=[year])])
                            # changed below to above line so that new mean of last 10 years, like with afa, is properly appended to df_GMT
                            # df_GMT = pd.concat([df_GMT,pd.DataFrame(GMT_lastyear).rename(index={0:year})])

                    # retain only period of interest
                    da_AFA = da_AFA.sel(time=slice(year_start,year_end))
                    df_GMT = df_GMT.loc[year_start:year_end,:]

                    # save GMT in metadatadict
                    d_isimip_meta[i]['GMT'] = df_GMT 
                    
                    # get ISIMIP GMT indices closest to GMT trajectories        
                    RCP2GMT_diff_15 = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=0)
                    RCP2GMT_diff_20 = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=0)
                    RCP2GMT_diff_NDC = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=0)
                    RCP2GMT_diff_R26eval = np.min(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=0)

                    ind_RCP2GMT_15 = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=0)
                    ind_RCP2GMT_20 = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=0)
                    ind_RCP2GMT_NDC = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=0)
                    ind_RCP2GMT_R26eval = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=0)
                    
                    # store GMT maxdiffs and indices in metadatadict
                    d_isimip_meta[i]['GMT_15_maxdiff'] = np.nanmax(RCP2GMT_diff_15)
                    d_isimip_meta[i]['GMT_20_maxdiff'] = np.nanmax(RCP2GMT_diff_20)
                    d_isimip_meta[i]['GMT_NDC_maxdiff'] = np.nanmax(RCP2GMT_diff_NDC)
                    d_isimip_meta[i]['GMT_R26eval_maxdiff'] = np.nanmax(RCP2GMT_diff_R26eval)       
                    d_isimip_meta[i]['GMT_15_valid'] = np.nanmax(RCP2GMT_diff_15) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_20_valid'] = np.nanmax(RCP2GMT_diff_20) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_NDC_valid'] = np.nanmax(RCP2GMT_diff_NDC) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_R26eval_valid'] = np.nanmax(RCP2GMT_diff_R26eval) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['ind_RCP2GMT_15'] = ind_RCP2GMT_15
                    d_isimip_meta[i]['ind_RCP2GMT_20'] = ind_RCP2GMT_20
                    d_isimip_meta[i]['ind_RCP2GMT_NDC'] = ind_RCP2GMT_NDC
                    d_isimip_meta[i]['ind_RCP2GMT_R26eval'] = ind_RCP2GMT_R26eval
                    
                    # adding this to avoid duplicates of da_AFA_pic in pickles
                    if '{}_{}'.format(d_isimip_meta[i]['model'],d_isimip_meta[i]['gcm']) not in pic_list:

                        # load associated picontrol variables (can be from up to 4 files)
                        file_names_pic  = glob.glob('./data/isimip/'+extreme+'/'+model.lower()+'/'+model.lower()+'*'+d_isimip_meta[i]['gcm']+'*_picontrol_*landarea*')

                        if  isinstance(file_names_pic, str): # single pic file 
                            da_AFA_pic  = open_dataarray_isimip(file_names_pic)
                        else:                               # concat pic files
                            das_AFA_pic = [open_dataarray_isimip(file_name_pic) for file_name_pic in file_names_pic]
                            da_AFA_pic  = xr.concat(das_AFA_pic, dim='time')
                            
                        # save AFA field as pickle
                        with open('./data/pickles/isimip_AFA_pic_{}_{}.pkl'.format(extreme,str(i)), 'wb') as f: # added extreme to string of pickle
                            pk.dump(da_AFA_pic,f)
                            
                        pic_list.append('{}_{}'.format(d_isimip_meta[i]['model'],d_isimip_meta[i]['gcm']))
                        
                        # save metadata
                        d_pic_meta[i] = {
                            'model': d_isimip_meta[i]['model'], 
                            'gcm': d_isimip_meta[i]['gcm'],              
                            'extreme': file_name.split('_')[3], 
                            'years': str(len(da_AFA_pic.time)),
                        }
                            
                    # save AFA field as pickle
                    with open('./data/pickles/isimip_AFA_{}_{}.pkl'.format(extreme,str(i)), 'wb') as f: # added extreme to string of pickle
                        pk.dump(da_AFA,f)

                    # update counter
                    i += 1
        
            # save metadata dictionary as a pickle
            print('Saving metadata')
            with open('./data/pickles/isimip_metadata_{}.pkl'.format(extreme), 'wb') as f:
                pk.dump(d_isimip_meta,f)
            with open('./data/pickles/isimip_pic_metadata_{}.pkl'.format(extreme), 'wb') as f:
                pk.dump(d_pic_meta,f)            

    else: 
        
        # loop over extremes
        print('Loading processed isimip data')
        # loac pickled metadata for isimip and isimip-pic simulations
        extreme = extremes[0]

        with open('./data/pickles/isimip_metadata_{}.pkl'.format(extreme), 'rb') as f:
            d_isimip_meta = pk.load(f)
        with open('./data/pickles/isimip_pic_metadata_{}.pkl'.format(extreme), 'rb') as f:
            d_pic_meta = pk.load(f)                

    return d_isimip_meta,d_pic_meta

#%% ----------------------------------------------------------------   
# Function to open isimip data array and read years from filename
# (the isimip calendar "days since 1661-1-1 00:00:00" cannot be read by xarray datetime )
# this implies that years in file need to correspond to years in filename
def open_dataarray_isimip(file_name): 
    
    begin_year = int(file_name.split('_')[-2])
    end_year = int(file_name.split('_')[-1].split('.')[0])
    
    # some files contain extra var 'time_bnds', first try reading for single var
    try:
        
        da = xr.open_dataarray(file_name, decode_times=False)
        
    except:
        
        da = xr.open_dataset(file_name, decode_times=False).exposure
    
    da['time'] = np.arange(begin_year,end_year+1)
    
    return da

# ---------------------------------------------------------------
# 2. Functions to manipulate (see ms_manip.m)
# ----------------------------------------------------------------


#%% ----------------------------------------------------------------
# interpolate life expectancies
def get_life_expectancies(
    df_worldbank_country, 
    df_unwpp_country,
):

    # original data runs from 1960 to 2017 but we want estimates from 1960 to 2020
    # add three rows of 0s
    df_extrayears = pd.DataFrame(
        np.empty([year_ref- df_worldbank_country.index.max(),len(df_worldbank_country.columns)]),
        columns=df_worldbank_country.columns,
        index=np.arange(df_worldbank_country.index.max()+1,year_ref+1,1),
    )
    df_worldbank_country = pd.concat([df_worldbank_country, df_extrayears]) # Luke: why does worldbank data go unused?

    # store birth_year data
    # dataframe filled with birthyears for every country
    df_birthyears = pd.DataFrame(np.transpose(np.tile(birth_years, (len(df_unwpp_country.keys()),1))) , columns=df_unwpp_country.keys(), index=birth_years)

    # extract life expectancy at age 5 data from UN WPP file and
    # linearly interpolate from 5-year WPP blocks to pre-defined birth
    # year (extrapolate from 2013 to 2020, note that UN WPP has no NaNs)
    df_birthyears_empty = pd.DataFrame(columns=df_unwpp_country.keys(), index=birth_years)
    
    df_unwpp_country_startyear = df_unwpp_country.set_index(df_unwpp_country.index.values-5)
    df_concat = pd.concat([df_unwpp_country_startyear,df_birthyears_empty]).sort_index()
    df_concat = df_concat[~df_concat.index.duplicated(keep='last')]
    df_unwpp_country_interp = df_concat.astype('float').interpolate(
        method='slinear', # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d
        limit_direction='both',
        fill_value='extrapolate',
    )
    df_unwpp_country_interp = df_unwpp_country_interp[df_unwpp_country_interp.index.isin(df_birthyears_empty.index)]
    df_life_expectancy_5 = df_unwpp_country_interp + 5 + 6

    return df_birthyears, df_life_expectancy_5

#%% ----------------------------------------------------------------
# interpolate cohortsize per country
def get_cohortsize_countries(
    wcde, 
    df_countries, 
    df_GMT_15,
): 

    # unpack loaded wcde values
    wcde_years, wcde_ages, wcde_country_data, unused = wcde 
    # 31 year ranges, 21 age categories

    # initialise dictionary to store cohort sizes dataframes per country with years as rows and ages as columns
    d_cohort_size = {}

    for i,name in enumerate(df_countries.index):
        # extract population size per age cohort data from WCDE file and
        # linearly interpolate from 5-year WCDE blocks to pre-defined birth year
        # ! this gives slightly different values than MATLAB at some interpolation points inherent to the interpolation
        wcde_country_data_reshape = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))).transpose()
        wcde_per_country = np.hstack((np.expand_dims(wcde_country_data_reshape[:,0],axis=1),wcde_country_data_reshape)) 
        wcde_per_country = np.array(np.vstack([wcde_per_country,wcde_per_country[-1,:]]), dtype='float64')
        [Xorig, Yorig] = np.meshgrid(np.concatenate(([np.min(ages)], wcde_ages)),np.concatenate((wcde_years, [np.max(df_GMT_15.index)]))) 
        [Xnew, Ynew] = np.meshgrid(ages, np.array(df_GMT_15.index))                                             # prepare for 2D interpolation
        wcde_country_data_raw = interpolate.griddata(
            (Xorig.ravel(),Yorig.ravel()),
            wcde_per_country.ravel(),
            (Xnew.ravel(),Ynew.ravel()),
        )
        wcde_country_data_interp = wcde_country_data_raw.reshape( len(df_GMT_15.index),len(ages))
        d_cohort_size[name] = pd.DataFrame(
            (wcde_country_data_interp /5), 
            columns=ages, 
            index=df_GMT_15.index
        )
    
    return d_cohort_size

#%% ----------------------------------------------------------------
# mask population per country based on gridded population and countrymask
# also communicate country masks as regionmask object 
def get_mask_population(
    da_population, 
    gdf_country_borders, 
    df_countries,
):

    # load country borders; join layer with country names (to have corresponding names for later purposes) and add country index as additional column
    df_countries['name'] = df_countries.index.values
    gdf_country_borders = gdf_country_borders.merge(
        df_countries, 
        left_on='ADM0_A3', 
        right_on='abbreviation'
    )

    # create regionmask regions object 
    countries_regions = regionmask.from_geopandas(
        gdf_country_borders, 
        names='name', 
        abbrevs="abbreviation", 
        name="country"
    )
    countries_mask = countries_regions.mask(da_population.lon, da_population.lat)

    # loop over countries as read in by worldbank data - Palestine and South Sudan are not in shapefile
    for name in df_countries.index.values: 

        if name in gdf_country_borders['name'].values:
            # only keep countries that are resolved with mask (get rid of small countries)
            if da_population.where(countries_mask==countries_regions.map_keys(name), drop=True).size != 0:
                # get mask index and sum up masked population
                df_countries.loc[name,'population'] = da_population.where(countries_mask==countries_regions.map_keys(name), drop=True).sum().values
        
    # remove countries which are not found in country borders file
    df_countries = df_countries[~df_countries.loc[:, 'population'].isnull()]
    
    # fix country borders dataframe for return
    gdf_country_borders = gdf_country_borders.set_index(gdf_country_borders.name).loc[:,['geometry','region']].reindex(df_countries.index)

    return  df_countries, countries_regions, countries_mask, gdf_country_borders

#%% ----------------------------------------------------------------
# get countries per region, returns dictionary with regions as keys and countries as values
def get_countries_per_region(
    df_countries, 
    df_regions,
):
    
    d_region_countries = {}
    for region in df_regions.index:
        if df_countries.loc[df_countries['region']==region].index.values.size > 0: # if not empty
            d_region_countries[region] = df_countries.loc[df_countries['region']==region].index.values
        elif df_countries.loc[df_countries['incomegroup']==region].index.values.size > 0: # if not empty
            d_region_countries[region] = df_countries.loc[df_countries['incomegroup']==region].index.values
        elif region == 'World': # take all countries
            d_region_countries[region] = df_countries.index.values
            
    return d_region_countries

#%% ----------------------------------------------------------------
# Get life expectancy, birth years and cohort weights per region, as well as countries per region
def get_regions_data(
    df_countries, 
    df_regions, 
    df_worldbank_region, 
    df_unwpp_region, 
    d_cohort_size,
):
    
    # get countries per region
    d_region_countries = get_countries_per_region(df_countries, df_regions)

    # filter for regions used
    df_regions = df_regions[df_regions.index.isin(d_region_countries.keys())]
    df_worldbank_region = df_worldbank_region.filter(items=d_region_countries.keys())
    df_unwpp_region = df_unwpp_region.filter(items=d_region_countries.keys())

    # get birthyears and life expectancy for regions
    df_birthyears_regions, df_life_expectancy_5_regions = get_life_expectancies(df_worldbank_region, df_unwpp_region)

    # get total population in the region per cohort in 2020
    cohort_size_year_ref = np.asarray([d_cohort_size[country].loc[year_ref] for country in d_cohort_size.keys()])
    df_cohort_size_year_ref = pd.DataFrame(
        cohort_size_year_ref,
        index=df_countries.index, 
        columns=ages
    )

    d_cohort_weights_regions = {}
    for region in d_region_countries.keys():
        d_cohort_weights_regions[region] = df_cohort_size_year_ref[df_cohort_size_year_ref.index.isin(d_region_countries[region])].transpose()
    
    return d_region_countries, df_birthyears_regions, df_life_expectancy_5_regions, d_cohort_weights_regions


# %%
