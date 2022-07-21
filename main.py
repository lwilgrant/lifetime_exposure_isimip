# ---------------------------------------------------------------
# Main script to postprocess and visualise lifetime exposure data
#
# Python translation of the MATLAB scripts of Thiery et al. (2021)
# https://github.com/VUB-HYDR/2021_Thiery_etal_Science
# ----------------------------------------------------------------

#%% ----------------------------------------------------------------
# Summary and notes

# to save the enironment used (with your path to the env directory): 
# conda env export -p C:\Users\ivand\anaconda3\envs\exposure_env > exposure_env.yml


# Data types are defined in the variable names starting with:  
#     df_     : DataFrame    (pandas)
#     gdf_    : GeoDataFrame (geopandas)
#     da_     : DataArray    (xarray)
#     d_      : dictionary  
#     sf_     : shapefile
#     ...dir  : directory

# Luke's review:
# writing "# DONE" to mark as read


# TODO
# - not yet masked for small countries
# - how to handle South-Sudan and Palestina? Now manually filtered out in load
# - calc_weighted_fldmean: adding different masks is very time-inefficient. Find more efficient way of doing this


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

scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# FLAGS
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'cropfailedarea'   # 0: all
                                    # 1: burntarea
                                    # 2: cropfailedarea
                                    # 3: driedarea
                                    # 4: floodedarea
                                    # 5: heatwavedarea
                                    # 6: tropicalcyclonedarea
                                    # 7: waterscarcity
flags['runs'] = 1          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 1         # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['exposure'] = 1       # 0: do not process ISIMIP runs to compute exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute exposure (i.e. produce and save exposure as pickle)
flags['exposure_pic'] = 1   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                            # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)


# TODO: add rest of flags


#%% ----------------------------------------------------------------
# INITIALISE
# ----------------------------------------------------------------
from settings import *

# set global variables
init()


# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)

#%% ----------------------------------------------------------------
# LOAD AND MANIPULATE DATA
# ----------------------------------------------------------------

# TODO: when regions added, make this one function returning dict! 
from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC

df_GMT_15, df_GMT_20, df_GMT_NDC = load_GMT(
    year_start,
    year_end,
) 

# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    # load worldbank and unwpp data
    meta, worldbank, unwpp = load_worldbank_unwpp_data()

    # unpack values
    df_countries, df_regions = meta
    df_worldbank_country, df_worldbank_region = worldbank
    df_unwpp_country, df_unwpp_region = unwpp

    # manipulate worldbank and unwpp data to get birth year and life expectancy values
    df_birthyears, df_life_expectancy_5 = get_life_expectancies(
        df_worldbank_country, 
        df_unwpp_country,
    )

    # load population size per age cohort data
    wcde = load_wcde_data() 

    # interpolate population size per age cohort data to our ages
    d_cohort_size = get_cohortsize_countries(
        wcde, 
        df_countries, 
        df_GMT_15,
    )

    # do the same for the regions; get life expectancy, birth years and cohort weights per region, as well as countries per region
    d_region_countries, df_birthyears_regions, df_life_expectancy_5_regions, d_cohort_weights_regions = get_regions_data(
        df_countries, 
        df_regions, 
        df_worldbank_region, 
        df_unwpp_region, 
        d_cohort_size,
    )

    # --------------------------------------------------------------------
    # Load population and country masks, and mask population per country
    
    # Load SSP population totals 
    da_population = load_population(
        year_start,
        year_end,
    )
    
    gdf_country_borders = gpd.read_file('./data/natural_earth/Cultural_10m/Countries/ne_10m_admin_0_countries.shp'); 

    # mask population totals per country  and save country regions object and countries mask
    df_countries, countries_regions, countries_mask, gdf_country_borders = get_mask_population(
        da_population, 
        gdf_country_borders, 
        df_countries,
    ) 

    # pack country information
    d_countries = {
        'info_pop' : df_countries, 
        'borders' : gdf_country_borders,
        'population_map' : da_population,
        'birth_years' : df_birthyears,
        'life_expectancy_5': df_life_expectancy_5, 
        'cohort_size' : d_cohort_size, 
        'mask' : (countries_regions,countries_mask),
    }

    # pack region information
    d_regions = {
        'birth_years' : df_birthyears_regions,
        'life_expectancy_5': df_life_expectancy_5_regions, 
        'cohort_size' : d_cohort_weights_regions,
    }

    # save metadata dictionary as a pickle
    print('Saving country and region data')
    
    if not os.path.isdir('./data/pickles'):
        os.mkdir('./data/pickles')
    with open('./data/pickles/country_info.pkl', 'wb') as f: # note; 'with' handles file stream closing
        pk.dump(d_countries,f)
    with open('./data/pickles/region_info.pkl', 'wb') as f:
        pk.dump(d_regions,f)

else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/pickles/country_info.pkl', 'rb'))

    # unpack country information
    df_countries = d_countries['info_pop']
    gdf_country_borders = d_countries['borders']
    da_population = d_countries['population_map']
    df_birthyears = d_countries['birth_years']
    df_life_expectancy_5 = d_countries['life_expectancy_5']
    d_cohort_size = d_countries['cohort_size']
    countries_regions, countries_mask = d_countries['mask']

    # load regions pickle
    d_regions = pk.load(open('./data/pickles/region_info.pkl', 'rb'))

    # unpack region information
    df_birthyears_regions = d_regions['birth_years']
    df_life_expectancy_5_regions = d_regions['life_expectancy_5']
    d_cohort_weights_regions = d_regions['cohort_size']
 
# --------------------------------------------------------------------
# Load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta,d_pic_meta = load_isimip(
    flags['runs'], 
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,    
)


#%% ----------------------------------------------------------------
# COMPUTE EXPOSURE PER LIFETIME
# ------------------------------------------------------------------

from exposure import * 

# --------------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span


if flags['exposure']: 
    
    start_time = time.time()
    
    # calculate exposure per country and per region and save data (takes 23 mins)
    d_exposure_perrun_RCP, d_exposure_perregion_perrun_RCP, = calc_exposure(
        grid_area,
        d_regions,
        d_isimip_meta, 
        df_birthyears_regions, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5,
    )
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )

else: # load processed country data

    print('Loading processed exposures')

    # load country pickle
    with open('./data/pickles/exposure_{}.pkl'.format(d_isimip_meta[list(d_isimip_meta.keys())[0]]['extreme']), 'rb') as f:
        d_exposure = pk.load(f)

    # unpack country information
    d_exposure_perrun_RCP = d_exposure['exposure_perrun_RCP']
    d_exposure_perrun_15 = d_exposure['exposure_perrun_15']
    d_exposure_perrun_20 = d_exposure['exposure_perrun_20']
    d_exposure_perrun_NDC = d_exposure['exposure_perrun_NDC']

    # unpack region information
    d_exposure_perregion_perrun_RCP = d_exposure['exposure_perregion_perrun_RCP']
    d_landfrac_peryear_perregion = d_exposure['landfrac_peryear_perregion']

# --------------------------------------------------------------------
# process picontrol data

if flags['exposure_pic']:
    
    start_time = time.time()
    
    # takes 38 mins crop failure
    d_exposure_perrun_pic, d_exposure_perregion_perrun_pic, = calc_exposure_pic(
        grid_area,
        d_regions,
        d_pic_meta, 
        df_birthyears_regions, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5,
    )
    
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )    
    
else: # load processed country data
    
    print('Loading processed exposures')

    with open('./data/pickles/exposure_pic_{}.pkl'.format(d_pic_meta[list(d_pic_meta.keys())[0]]['extreme']), 'rb') as f:
        d_exposure_pic = pk.load(f)
    
    # unpack pic country information
    d_exposure_perrun_pic = d_exposure_pic['exposure_perrun']
    
    # unpack pic regional information
    d_exposure_perregion_perrun_pic = d_exposure_pic['exposure_perregion_perrun']
    d_landfrac_peryear_perregion_pic = d_exposure_pic['landfrac_peryear_perregion']
    
#%% --------------------------------------------------------------------
# compile hist+RCP and pic for EMF and emergence analysis

# call function to compute mmm, std, qntl for exposure (also 99.99 % of pic as "ext")
ds_exposure_RCP = calc_exposure_mmm_xr(
    d_exposure_perrun_RCP,
    'country',
    'RCP',
)
ds_exposure_15 = calc_exposure_mmm_xr(
    d_exposure_perrun_15,
    'country',
    '15',
)
ds_exposure_20 = calc_exposure_mmm_xr(
    d_exposure_perrun_20,
    'country',
    '20',
)
ds_exposure_NDC = calc_exposure_mmm_xr(
    d_exposure_perrun_NDC,
    'country',
    'NDC',
)
ds_exposure_perregion = calc_exposure_mmm_xr(
    d_exposure_perregion_perrun_RCP,
    'region',
    'RCP',
)
ds_exposure_pic = calc_exposure_mmm_pic_xr(
    d_exposure_perrun_pic,
    'country',
    'pic',
)
ds_exposure_pic_perregion = calc_exposure_mmm_pic_xr(
    d_exposure_perregion_perrun_pic,
    'region',
    'pic',
)

# pool all datasets for different trajectories
ds_exposure = xr.merge([
    ds_exposure_RCP,
    ds_exposure_15,
    ds_exposure_20,
    ds_exposure_NDC,
])

# emergence calculations
gdf_exposure_emergence_birth_year = calc_exposure_emergence(
    ds_exposure,
    ds_exposure_pic,
    gdf_country_borders,
)


#%% --------------------------------------------------------------------
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

def c(x):
   col = plt.cm.OrRd(x)
   fig, ax = plt.subplots(figsize=(1,1))
   fig.set_facecolor(col)
   ax.axis("off")
   plt.show()

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
            
#%% --------------------------------------------------------------------
# code for comparing wtih matlab
# compare individual runs for certain gcm and mod combos between here and matlab
runs = [k for k,v in d_isimip_meta.items() if v['gcm'] == 'gfdl-esm2m' and v['model'] == 'gepic']
run26 = d_exposure_perrun_20[runs[0]]
run60 = d_exposure_perrun_20[runs[1]]

# compare indices for GMT mapping
os.chdir(r'C:\Users\adm_lgrant\Documents\repos\lifetime_exposure_isimip\data\matlab_comparison')
test15 = pd.read_excel('indexes_RCP2GMT_15.xlsx',header=None) # indices from matlab
test20 = pd.read_excel('indexes_RCP2GMT_20.xlsx',header=None)
testNDC = pd.read_excel('indexes_RCP2GMT_NDC.xlsx',header=None)

df_test15 = pd.DataFrame(np.empty_like(test15.values)) # indices from python
df_test20 = pd.DataFrame(np.empty_like(test20.values))
df_testNDC = pd.DataFrame(np.empty_like(testNDC.values))
for i in d_isimip_meta.keys():
    df_test15.at[i-1,:] = d_isimip_meta[i]['ind_RCP2GMT_15']
    df_test20.at[i-1,:] = d_isimip_meta[i]['ind_RCP2GMT_20']
    df_testNDC.at[i-1,:] = d_isimip_meta[i]['ind_RCP2GMT_NDC']
for df,mdf in zip([df_test15,df_test20,df_testNDC],[test15,test20,testNDC]):
    df['match_bool'] = np.zeros(len(df.index.values))
    for i in list(d_isimip_meta.keys()):
        df.loc[i-1,'match_bool'] = np.all(df.iloc[i-1,:-1].values.astype('int64') == mdf.loc[i-1,:].values - 1)
        
for i in range(len(testNDC.loc[8,:].values)):
    print(int(df_testNDC.loc[8,i])==int(testNDC.loc[8,i]+1))        

# compare maxdiffs here with printout on matlab
maxdiffs_NDC = [v['GMT_NDC_maxdiff'] for k,v in list(d_isimip_meta.items())]

# compare mmm to excel sheets of mmm from matlab (round to 2 digits)
os.chdir(r'C:\Users\adm_lgrant\Documents\repos\lifetime_exposure_isimip\data\matlab_comparison')

countries_matlab = pd.read_excel('matlab_countries.xlsx',header=None)
countries_matlab = countries_matlab.drop(index=[131,145]).reset_index().drop(columns='index')
countries_python = ds_exposure_15.country.values
# not_in_python_countries = ['Palestine','South Sudan']

# matlab sheets
matlab_15_mmm = pd.read_excel('exposure_mean_15.xlsx',header=None).round(decimals=2)
matlab_15_mmm = matlab_15_mmm.drop(index=[131,145]).reset_index().drop(columns='index')
matlab_20_mmm = pd.read_excel('exposure_mean_20.xlsx',header=None).round(decimals=2)
matlab_20_mmm = matlab_20_mmm.drop(index=[131,145]).reset_index().drop(columns='index')
matlab_NDC_mmm = pd.read_excel('exposure_mean_NDC.xlsx',header=None).round(decimals=2)
matlab_NDC_mmm = matlab_NDC_mmm.drop(index=[131,145]).reset_index().drop(columns='index')

# python sheets
python_15_mmm = pd.DataFrame(ds_exposure['mmm_15'].values.transpose()).round(decimals=2)
python_20_mmm = pd.DataFrame(ds_exposure['mmm_20'].values.transpose()).round(decimals=2)
python_NDC_mmm = pd.DataFrame(ds_exposure['mmm_NDC'].values.transpose()).round(decimals=2)

countries_matlab.index[countries_matlab[0]=='France']
matlab_20_mmm.loc[countries_matlab.index[countries_matlab[0]=='China']]
python_20_mmm.loc[countries_matlab.index[countries_matlab[0]=='China']]

# plot
for i in python_15_mmm.index.values:
    f,ax = plt.subplots()
    ax.plot(np.squeeze(python_20_mmm.loc[i].values),color='k',label='python') # python
    ax.plot(np.squeeze(matlab_20_mmm.loc[i].values),color='blue',label='matlab') # python
    ax.set_title(countries_python[i])
    
# compare canada 2.0 GMT mapping from matlab to python
matlab_20_canada = pd.read_excel('canada_perrun_20.xlsx',header=None).round(decimals=2)
python_20_canada = pd.DataFrame().reindex_like(matlab_20_canada)
for i in list(d_exposure_perrun_20.keys()):
    python_20_canada.at[i-1,:] = d_exposure_perrun_20[i].loc[:,'Canada'].values
python_20_canada = python_20_canada.round(decimals=2)

# compare canada 1.5 GMT mapping from matlab to python
matlab_15_canada = pd.read_excel('canada_perrun_15.xlsx',header=None).round(decimals=2)
python_15_canada = pd.DataFrame().reindex_like(matlab_15_canada)
for i in list(d_exposure_perrun_20.keys()):
    python_15_canada.at[i-1,:] = d_exposure_perrun_15[i].loc[:,'Canada'].values
python_15_canada = python_15_canada.round(decimals=2)

# compare canada 1.5 GMT mapping from matlab to python
matlab_RCP_canada = pd.read_excel('canada_perrun_RCP.xlsx',header=None).round(decimals=2)
python_RCP_canada = pd.DataFrame().reindex_like(matlab_RCP_canada)
for i in list(d_exposure_perrun_RCP.keys()):
    python_RCP_canada.at[i-1,:] = d_exposure_perrun_RCP[i].loc[:,'Canada'].values
python_RCP_canada = python_RCP_canada.round(decimals=2)

# plot
for i in list(d_exposure_perrun_RCP.keys()):
    f,ax = plt.subplots()
    ax.plot(python_RCP_canada.loc[i-1,:].values,color='k',label='python') # python
    ax.plot(matlab_RCP_canada.loc[i-1,:].values,color='blue',label='matlab') # python
    ax.set_title(str(i))
    # clear that runs 7, 15 and 20 are all considerably higher in Canada

# palestine is 131
# south sudan is 145
# countries_matlab.drop(index=[131,145]).reset_index()
# # call function computing the Exposure Multiplication Factor (EMF)
# # here I use multi-model mean as a reference
# d_EMF_mmm, d_EMF_q25, d_EMF_q75 = calc_exposure_EMF(
#     d_exposure_mmm, 
#     d_exposure_q25, 
#     d_exposure_q75, 
#     d_exposure_mmm,
# )
