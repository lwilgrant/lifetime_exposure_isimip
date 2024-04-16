#!/usr/bin/env python3
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

# To do
# Put all GDP/deprivation loading (GDP and GDP per capita for max time length of product in function under Emergence section)
    # run read in of isimip hist + rcp26 
    # pip install py-cdo; write CDO python lines to run remapcon2 if isimip grid Wang and Sun files not there
    # run read in of wang and sung;
    # pickle datasets
# Make lifetime gdp function (birth cohort choice in settings)
    # currently have sum option, need to add mean (don't worry about fractional last year of life span)
    # pickle datasets
# Cross lifetime gdp against emergence masks 
    # get pop estimates for sample of percentiles for each GDP product (2)
# Put all deprivation conversion/loading in function and run it in emergence section
    # pickle data
# Cross deprivation against emergence masks
    # get pop estimates for sample percentiles for deprivation
    
    
    


#               
#%%  ----------------------------------------------------------------
# import and path
# ----------------------------------------------------------------

import xarray as xr
import pickle as pk
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cr
import geopandas as gpd
# import seaborn as sns # must comment this out for things to work on the server
scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# flags
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'heatwavedarea' # 0: all
                                # 1: burntarea
                                # 2: cropfailedarea
                                # 3: driedarea
                                # 4: floodedarea
                                # 5: heatwavedarea
                                # 6: tropicalcyclonedarea
                                # 7: waterscarcity
flags['gmt'] = 'ar6_new'    # original: use Wim's stylized trajectory approach with max trajectory a linear increase to 3.5 deg                               
                            # ar6: substitute the linear max wth the highest IASA c7 scenario (increasing to ~4.0), new lower bound, and new 1.5, 2.0, NDC (2.8), 3.0
                            # ar6_new: works off ar6, but ensures only 1.5-3.5 with perfect intervals of 0.1 degrees (less proc time and data volume)
flags['rm'] = 'rm'       # no_rm: no smoothing of RCP GMTs before mapping
                         # rm: 21-year rolling mean on RCP GMTs 
flags['version'] = 'pickles_v2'     # pickles: original version, submitted to Nature
                                        # inconsistent GMT steps (not perfect 0.1 degree intervals)
                                        # GMT steps ranging 1-4 (although study only shows ~1.5-3.5, so runs are inefficient)
                                        # only 99.99% percentile for PIC threshold
                                    # pickles_v2: version generated after submission to Nature in preparation for criticism/review
                                        # steps fixed in load_manip to be only 1.5-3.5, with clean 0.1 degree intervals
                                        # 5 percentiles for PIC threshold and emergence for each
                         # rm: 21-year rolling mean on RCP GMTs                          
flags['run'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['lifetime_exposure_cohort'] = 0       # 0: do not process ISIMIP runs to compute exposure across cohorts (i.e. load exposure pickle)
                                            # 1: process ISIMIP runs to compute exposure across cohorts (i.e. produce and save exposure as pickle)                            
flags['lifetime_exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                                     # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)
flags['emergence'] = 0      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)
flags['birthyear_emergence'] = 0    # 0: only run calc_birthyear_align with birth years from 1960-2020
                                    # 1: run calc_birthyear_align with birth years from 1960-2100                             
flags['gridscale'] = 0      # 0: do not process grid scale analysis, load pickles
                            # 1: process grid scale analysis
flags['gridscale_country_subset'] = 0      # 0: run gridscale analysis on all countries
                                           # 1: run gridscale analysis on subset of countries determined in "get_gridscale_regions" 
flags['gridscale_union'] = 0        # 0: do not process/load pickles for mean emergence and union of emergence across hazards
                                    # 1: process/load those^ pickles    
flags['global_emergence_recollect'] = 1       # 0: do not load pickles of global emergence masks
                                    # 1: load pickles                                                                               
flags['gdp_deprivation'] = 1        # 0: do not process/load lifetime GDP/GRDI average
                                    # 1: load lifetime GDP average analysis        
flags['vulnerability'] = 1          # 0: do not process subsets of d_collect_emergence vs gdp & deprivation quantiles
                                    # 1: process/load d_collect_emergence vs gdp & deprivation quantiles for vulnerability analysis
flags['plot_ms'] = 0 # 1 yes plot, 0 no plot
flags['plot_si'] = 0
flags['reporting'] = 0  
flags['testing'] = 0   



#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)

#%% ----------------------------------------------------------------
# load and manipulate demographic, GMT and ISIMIP data
# ----------------------------------------------------------------

from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj

df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj = load_GMT(
    year_start,
    year_end,
    year_range,
    flags,
)

# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    d_countries = all_country_data(flags)

else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/{}/country_info.pkl'.format(flags['version']), 'rb'))
    
# unpack country information
df_countries = d_countries['info_pop']
gdf_country_borders = d_countries['borders']
da_population = d_countries['population_map']
df_birthyears = d_countries['birth_years']
df_life_expectancy_5 = d_countries['life_expectancy_5']
da_cohort_size = d_countries['cohort_size']
countries_regions, countries_mask = d_countries['mask']    
 
# --------------------------------------------------------------------
# load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

d_isimip_meta,d_pic_meta = load_isimip(
    extremes,
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
    flags,
)

sims_per_step = {}
for step in GMT_labels:
    sims_per_step[step] = []
    for i in list(d_isimip_meta.keys()):
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            sims_per_step[step].append(i)

#%% ----------------------------------------------------------------
# compute exposure per lifetime at country-scale
# ------------------------------------------------------------------

from exposure import *

# --------------------------------------------------------------------
# process lifetime exposure across cohorts

if flags['lifetime_exposure_cohort']:
    
    start_time = time.time()
    
    # calculate exposure per country and per cohort
    calc_cohort_lifetime_exposure(
        d_isimip_meta,
        df_countries,
        countries_regions,
        countries_mask,
        da_population,
        da_cohort_size,
        flags,
    )
    
    print("--- {} minutes to compute cohort exposure ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )
    
else:  # load processed cohort exposure data
    
    print('Processed exposures will be loaded in emergence calculation')

# --------------------------------------------------------------------
# process picontrol lifetime exposure

if flags['lifetime_exposure_pic']:
    
    start_time = time.time()
    
    # takes 38 mins crop failure
    d_exposure_perrun_pic = calc_lifetime_exposure_pic(
        d_pic_meta, 
        df_countries, 
        countries_regions, 
        countries_mask, 
        da_population, 
        df_life_expectancy_5, 
        flags,
    )
    
    print("--- {} minutes for PIC exposure ---".format(
        np.floor((time.time() - start_time) / 60),
        )
          )    
    
else: # load processed pic data
    
    print('Loading processed pic exposures')

#     with open('./data/{}/{}/exposure_pic_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
#         d_exposure_perrun_pic = pk.load(f)
    
# ds_exposure_pic = calc_exposure_mmm_pic_xr(
#     d_exposure_perrun_pic,
#     'country',
#     'pic',
# )

#%% ----------------------------------------------------------------
# compute lifetime emergence
# ------------------------------------------------------------------

from emergence import *

# --------------------------------------------------------------------
# process emergence of cumulative exposures, mask cohort exposures for time steps of emergence

if flags['emergence']:
    
    if flags['birthyear_emergence']:
        
        by_emergence = np.arange(1960,2101)
        
    else:
        
        by_emergence = birth_years        
    
    if not os.path.isfile('./data/{}/cohort_sizes.pkl'.format(flags['version'])):
        
        # need new cohort dataset that has total population per birth year (using life expectancy info; each country has a different end point)
        da_cohort_aligned = calc_birthyear_align(
            da_cohort_size,
            df_life_expectancy_5,
            by_emergence,
        )
        
        # convert to dataset and add weights
        ds_cohorts = ds_cohort_align(
            da_cohort_size,
            da_cohort_aligned,
        )
        
        # pickle birth year aligned cohort sizes and global mean life expectancy
        with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_cohorts,f)  

    else:
        
        # load pickled birth year aligned cohort sizes and global mean life expectancy
        with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
            ds_cohorts = pk.load(f)                             
    
    ds_ae_strj, ds_pf_strj = strj_emergence(
        d_isimip_meta,
        df_life_expectancy_5,
        ds_exposure_pic,
        ds_cohorts,
        by_emergence,
        flags,
    )
        
else: # load pickles
    
    pass
    
    # # birth year aligned population
    # with open('./data/{}/cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
    #     ds_cohorts = pk.load(f)
    
    # # pop frac
    # with open('./data/{}/{}/pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
    #     ds_pf_strj = pk.load(f)                
    
                 
#%% ----------------------------------------------------------------
# grid scale emergence
# ------------------------------------------------------------------

from gridscale import *

# list of countries to run gridscale analysis on (sometimes doing subsets across basiss/regions in floods/droughts)
gridscale_countries = get_gridscale_regions(
    grid_area,
    flags,
    gdf_country_borders,
)

# birth year aligned cohort sizes for gridscale analysis (summed over lat/lon per country)
if not os.path.isfile('./data/{}/gs_cohort_sizes.pkl'.format(flags['version'])):

    da_gs_popdenom = get_gridscale_popdenom(
        gridscale_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        da_population,
        df_life_expectancy_5,
    )

    # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
        pk.dump(da_gs_popdenom,f)  
        
else:
    
    # load pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
        da_gs_popdenom = pk.load(f)               

# run gridscale emergence analysis
if flags['gridscale']:
    
    ds_le_gs, ds_pf_gs = gridscale_emergence(
        d_isimip_meta,
        d_pic_meta,
        flags,
        gridscale_countries,
        da_cohort_size,
        countries_regions,
        countries_mask,
        df_life_expectancy_5,
        da_population,
    )
    
else:
    
    # load pickled aggregated lifetime exposure, age emergence and pop frac datasets
    with open('./data/{}/{}/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
        ds_le_gs = pk.load(f)
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
        ds_pf_gs = pk.load(f)
            
# # estimate union of all hazard emergences
# if flags['gridscale_union']:
    
#     da_emergence_mean, da_emergence_union = get_gridscale_union(
#         da_population,
#         flags,
#         gridscale_countries,
#         countries_mask,
#         countries_regions,
#     )

# read in global emergence masks
if flags['global_emergence_recollect']:
    
    d_global_emergence = collect_global_emergence(
        grid_area,
        flags,
        countries_mask,
        countries_regions,
        gridscale_countries,
        df_GMT_strj,
    )
    
# load/proc GDP and deprivation data
if flags['gdp_deprivation']:
    
    ds_gdp, ds_grdi = load_gdp_deprivation(
        flags,
        grid_area,
        gridscale_countries,
        df_life_expectancy_5,
    )
    
# vulnerability subsetting
if flags['vulnerability']:  

    da_cohort_size_1960_2020 = get_spatially_explicit_cohorts_1960_2020(
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
        da_cohort_size,
    )
    
    # adds data arrays to ds_gdp and ds_grdi with ranked vulnerability binned by population (i.e. ranges of ranked vulnerability, physically distributed, grouped/binned by population size)            
    ds_gdp_qntls, ds_grdi_qntls = get_vulnerability_quantiles(
        flags,
        grid_area,
        da_cohort_size_1960_2020,
        ds_gdp,
        ds_grdi,
    )
        
    # just a dummy d_global_emergence to run emergence_by_vulnerability
    try:
        d_global_emergence
    except NameError:
        print('to save memory on my laptop, d_global_emergence is not unpickled. defining a dummy var for emergence_by_vulnerability')
        d_global_emergence={}
    else:
        pass

    ds_vulnerability = emergence_by_vulnerability(
        flags,
        df_GMT_strj,
        ds_gdp_qntls,
        ds_grdi_qntls,
        da_cohort_size_1960_2020,
        d_global_emergence,
    )
    
    if flags['testing']:
            
        # GDP vulnerability quantile pyramid plots ===========================================================
        extremes = [
            'burntarea', 
            'cropfailedarea', 
            'driedarea', 
            'floodedarea', 
            'heatwavedarea', 
            'tropicalcyclonedarea',
        ]
        
        GMT_integers = [0,10,12,17,20]
        
        # also plot 1960 vs 2020 for gdp (grdi only has 2020)
        population_quantiles_10poorest = []
        population_quantiles_10richest = []
        
        population_quantiles_20poorest = []
        population_quantiles_20richest = []
        
        for by in birth_years:
            
            gdp = ds_gdp['gdp_isimip_rcp26_mean'].sel(birth_year=by)
            pop = da_cohort_size_1960_2020.sel(birth_year=by)

            gdp = gdp.where(pop.notnull())
            pop = pop.where(gdp.notnull())

            #=======================
            # another attempt where I rename and assign coords to "dim_0" so that coord labels are preserved after dropping nans
            # will play around with order of sorting/dropping nans/binning here until I get population to bin with roughly equal group sizes, like above, 
            # NOTE this is the correct/working approach
            vulnerability = xr.DataArray(gdp.values.flatten())
            vulnerability = vulnerability.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(vulnerability))}) # have to do this so the coords are traceable back to the 2-D layout
            vulnerability_ranks = vulnerability.rank(dim='gridcell_number').round()
            vulnerability_indices = vulnerability_ranks.gridcell_number
            sorted_vi = vulnerability_indices.sortby(vulnerability_ranks) # (I don't end up using this)
            sorted_vr = vulnerability_ranks.sortby(vulnerability_ranks) # sort ranks of vulnerability
            sorted_vr_nonans = sorted_vr[sorted_vr.notnull()] # drop nans from sorted v-ranks
            sorted_v = vulnerability.sortby(vulnerability_ranks) # sort vulnerability
            sorted_v_nonans = sorted_v[sorted_v.notnull()] # drop nans from sorted vulnerability array

            pop_flat = xr.DataArray(pop.values.flatten())
            pop_flat = pop_flat.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(pop_flat))}) # have to do this so the coords are traceable back to the 2-D layout
            sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because gdp and pop need common mask
            sorted_pop_nonans = sorted_pop[sorted_pop.notnull()]
            sorted_pop_nonans_cumsum = sorted_pop_nonans.cumsum()
            sorted_pop_nonans_cumsum_pct = sorted_pop_nonans_cumsum / sorted_pop_nonans.sum()

            # gather pop totals for plotting for each birth year
            population_quantiles_10poorest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()/10**6) # groups all even population!!!    
            population_quantiles_10richest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item()/10**6) # groups all even population!!!  
            
            population_quantiles_20poorest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()+
                 sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[1].item())/10**6
            )
            population_quantiles_20richest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-2].item()+
                 sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item())/10**6
            )
            
        for e in extremes:
            
            for GMT in GMT_current_policies:#GMT_integers:
                
                # ensemble mean unprecedented population for poor and rich quantiles
                unprec_pop_quantiles_10poorest = []
                unprec_pop_quantiles_10richest = []
                
                unprec_pop_quantiles_20poorest = []
                unprec_pop_quantiles_20richest = []     
                
                # errorbar info (+/- std) for the above quantiles
                unprec_pop_std_10poorest = []
                unprec_pop_std_10richest = []
                
                unprec_pop_std_20poorest = []
                unprec_pop_std_20richest = []         
                
                # ttest results ("*_poor" means that we test for poor quantiles to be greater. "*_rich" for rich quantiles to be greater)
                ttest_10pc_pvals_poor = []       
                ttest_20pc_pvals_poor = []       
                ttest_10pc_pvals_rich = []       
                ttest_20pc_pvals_rich = []       
                
                indices_vulnerability = ds_vulnerability.vulnerability_index.data
                v='gdp_q_by_p'
                df_vulnerability = ds_vulnerability.to_dataframe().reset_index()      
                df_vulnerability_e = df_vulnerability.loc[:,['run','GMT','qntl','vulnerability_index','birth_year',e]]
                df_vulnerability_e.loc[:,e] = df_vulnerability_e.loc[:,e] / 10**6 # convert to millions of people    
                
                for by in birth_years:
                    
                    # gather unprec totals for plotting
                    # poorest 10 percent
                    poor_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10poorest.append(poor_unprec_10pc.mean())
                    unprec_pop_std_10poorest.append(poor_unprec_10pc.std())
                    
                    # richest 10 percent
                    rich_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10richest.append(rich_unprec_10pc.mean())
                    unprec_pop_std_10richest.append(rich_unprec_10pc.std())
                    
                    # t test for difference between rich and poor samples
                    ttest_10pc_poor = ttest_rel(
                        a=poor_unprec_10pc[poor_unprec_10pc.notnull()].values,
                        b=rich_unprec_10pc[rich_unprec_10pc.notnull()].values,
                        alternative='greater'
                    )
                    ttest_10pc_pvals_poor.append(ttest_10pc_poor.pvalue)
                    ttest_10pc_rich = ttest_rel(
                        a=rich_unprec_10pc[rich_unprec_10pc.notnull()].values,
                        b=poor_unprec_10pc[poor_unprec_10pc.notnull()].values,
                        alternative='greater',
                    )
                    ttest_10pc_pvals_poor.append(ttest_10pc_poor.pvalue) 
                    ttest_10pc_pvals_rich.append(ttest_10pc_rich.pvalue)                   
                    
                    # poorest 20 percent
                    poor_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    poor_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==1)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20poorest.append(poor_unprec_20pci.mean() +poor_unprec_20pcii.mean())       
                    unprec_pop_std_20poorest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']<=1)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )
                    
                    # richest 20 percent
                    rich_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==8)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    rich_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20richest.append(rich_unprec_20pci.mean()+rich_unprec_20pcii.mean())
                    unprec_pop_std_20richest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']>=8)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )  
                    
                    # t test for difference between rich and poor samples
                    ttest_20pc_poor = ttest_rel(
                        a=np.concatenate((poor_unprec_20pci[poor_unprec_20pci.notnull()].values,poor_unprec_20pcii[poor_unprec_20pcii.notnull()].values)),
                        b=np.concatenate((rich_unprec_20pci[rich_unprec_20pci.notnull()].values,rich_unprec_20pcii[rich_unprec_20pcii.notnull()].values)),
                        alternative='greater'
                    )  
                    ttest_20pc_pvals_poor.append(ttest_20pc_poor.pvalue)
                    ttest_20pc_rich = ttest_rel(
                        a=np.concatenate((rich_unprec_20pci[rich_unprec_20pci.notnull()].values,rich_unprec_20pcii[rich_unprec_20pcii.notnull()].values)),
                        b=np.concatenate((poor_unprec_20pci[poor_unprec_20pci.notnull()].values,poor_unprec_20pcii[poor_unprec_20pcii.notnull()].values)),
                        alternative='greater'
                    )                      
                    ttest_20pc_pvals_rich.append(ttest_20pc_rich.pvalue)
                
                # for qntl_range in ('10', '20'):
                qntl_range='10'
                    
                if qntl_range == '10':
                    
                    # data
                    poor_unprec = np.asarray(unprec_pop_quantiles_10poorest)
                    poor_std = np.asarray(unprec_pop_std_10poorest)
                    poor_pop = np.asarray(population_quantiles_10poorest)
                    rich_unprec = np.asarray(unprec_pop_quantiles_10richest)
                    rich_std = np.asarray(unprec_pop_std_10richest)
                    rich_pop = np.asarray(population_quantiles_10richest)
                    pvalues_poor = np.asarray(ttest_10pc_pvals_poor)
                    pvalues_rich = np.asarray(ttest_10pc_pvals_rich)
                    
                    # # labels
                    # ax1_xticks = [-4,-8,-12]
                    # ax1_xtick_labels = ["4","8","12"]
                    
                    # ax2_xticks = [4,8,12]
                    # ax2_xtick_labels = ax1_xtick_labels      
                    # labels
                    ax_xts = {}
                    ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
                    ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
                    ax_xts['xtick_labels_pple'] = ["4","8","12"]
                    ax_xts['xtick_labels_pct'] = ["25","50","75","100"]

                    ax_xts['ax2_xticks_pple'] = [4,8,12]
                    ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
                    
                elif qntl_range == '20':
                    
                    poor_unprec = np.asarray(unprec_pop_quantiles_20poorest)
                    poor_std = np.asarray(unprec_pop_std_20poorest)
                    poor_pop = np.asarray(population_quantiles_20poorest)
                    rich_unprec = np.asarray(unprec_pop_quantiles_20richest)
                    rich_std = np.asarray(unprec_pop_std_20richest)
                    rich_pop = np.asarray(population_quantiles_20richest)
                    pvalues_poor = np.asarray(ttest_20pc_pvals_poor)
                    pvalues_rich = np.asarray(ttest_20pc_pvals_rich)
                    
                    # labels
                    ax_xts = {}
                    ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
                    ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
                    ax_xts['xtick_labels_pple'] = ["5","10","15","20","25"]
                    ax_xts['xtick_labels_pct'] = ["25","50","75","100"]

                    ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
                    ax_xts['ax2_xticks_pct'] = [25,50,75,100]
                
                vln_type='gdp'
                sl=0.05 # significance level
                # for unit in ('pct','pple'): # 'pple' ("people") or 'pct' ("percentage"), for xaxis ticks
                unit='pple'
                pyramid_plot(
                    e,
                    GMT,
                    df_GMT_strj,
                    poor_pop,
                    poor_unprec,
                    poor_std,
                    rich_pop,
                    rich_unprec,
                    rich_std,
                    pvalues_poor,
                    pvalues_rich,
                    sl,
                    qntl_range,
                    unit,
                    ax_xts,
                    vln_type,
                )
        
        # grdi vulnerability quantil pyramid plots ===========================================================
        
        # also plot 1960 vs 2020 for gdp (grdi only has 2020)
        population_quantiles_10poorest = []
        population_quantiles_10richest = []
        
        population_quantiles_20poorest = []
        population_quantiles_20richest = []
        
        for by in birth_years:
            
            grdi = ds_grdi['grdi']
            pop = da_cohort_size_1960_2020.sel(birth_year=by)

            grdi = grdi.where(pop.notnull())
            pop = pop.where(grdi.notnull())

            #=======================
            # another attempt where I rename and assign coords to "dim_0" so that coord labels are preserved after dropping nans
            # will play around with order of sorting/dropping nans/binning here until I get population to bin with roughly equal group sizes, like above, 
            # NOTE this is the correct/working approach
            vulnerability = xr.DataArray(grdi.values.flatten())
            vulnerability = vulnerability.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(vulnerability))}) # have to do this so the coords are traceable back to the 2-D layout
            vulnerability_ranks = vulnerability.rank(dim='gridcell_number').round()
            vulnerability_indices = vulnerability_ranks.gridcell_number
            sorted_vi = vulnerability_indices.sortby(vulnerability_ranks) # (I don't end up using this)
            sorted_vr = vulnerability_ranks.sortby(vulnerability_ranks) # sort ranks of vulnerability
            sorted_vr_nonans = sorted_vr[sorted_vr.notnull()] # drop nans from sorted v-ranks
            sorted_v = vulnerability.sortby(vulnerability_ranks) # sort vulnerability
            sorted_v_nonans = sorted_v[sorted_v.notnull()] # drop nans from sorted vulnerability array

            pop_flat = xr.DataArray(pop.values.flatten())
            pop_flat = pop_flat.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(pop_flat))}) # have to do this so the coords are traceable back to the 2-D layout
            sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because gdp and pop need common mask
            sorted_pop_nonans = sorted_pop[sorted_pop.notnull()]
            sorted_pop_nonans_cumsum = sorted_pop_nonans.cumsum()
            sorted_pop_nonans_cumsum_pct = sorted_pop_nonans_cumsum / sorted_pop_nonans.sum()

            # gather pop totals for plotting for each birth year
            population_quantiles_10poorest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item()/10**6) # groups all even population!!!    
            population_quantiles_10richest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()/10**6) # groups all even population!!!  
            
            population_quantiles_20poorest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-2].item()+
                 sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item())/10**6
            )
            population_quantiles_20richest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()+
                 sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[1].item())/10**6
            )
            
        for e in extremes:
            
            for GMT in GMT_current_policies:
                
                # ensemble mean unprecedented population for poor and rich quantiles
                unprec_pop_quantiles_10poorest = []
                unprec_pop_quantiles_10richest = []
                
                unprec_pop_quantiles_20poorest = []
                unprec_pop_quantiles_20richest = []     
                
                # errorbar info (+/- std) for the above quantiles
                unprec_pop_std_10poorest = []
                unprec_pop_std_10richest = []
                
                unprec_pop_std_20poorest = []
                unprec_pop_std_20richest = []                
                
                indices_vulnerability = ds_vulnerability.vulnerability_index.data
                v='grdi_q_by_p'
                df_vulnerability = ds_vulnerability.to_dataframe().reset_index()      
                df_vulnerability_e = df_vulnerability.loc[:,['run','GMT','qntl','vulnerability_index','birth_year',e]]
                df_vulnerability_e.loc[:,e] = df_vulnerability_e.loc[:,e] / 10**6 # convert to millions of people    
                
                for by in birth_years:
                    
                    # gather unprec totals for plotting
                    # poorest 10 percent
                    poor_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10poorest.append(poor_unprec_10pc.mean())
                    unprec_pop_std_10poorest.append(poor_unprec_10pc.std())
                    
                    # richest 10 percent
                    rich_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10richest.append(rich_unprec_10pc.mean())
                    unprec_pop_std_10richest.append(rich_unprec_10pc.std())
                    
                    # t test for difference between rich and poor samples
                    ttest_rel(a=)                    
                    
                    # poorest 20 percent
                    poor_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==8)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    poor_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20poorest.append(poor_unprec_20pci.mean() +poor_unprec_20pcii.mean())       
                    unprec_pop_std_20poorest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']>=8)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )
                    
                    # richest 20 percent
                    rich_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    rich_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==1)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20richest.append(rich_unprec_20pci.mean()+rich_unprec_20pcii.mean())
                    unprec_pop_std_20richest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']<=1)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )                    
                
                for qntl_range in ('10', '20'):
                    
                    if qntl_range == '10':
                        
                        # data
                        poor_unprec = unprec_pop_quantiles_10poorest
                        poor_std = unprec_pop_std_10poorest
                        poor_pop = population_quantiles_10poorest
                        rich_unprec = unprec_pop_quantiles_10richest
                        rich_std = unprec_pop_std_10richest
                        rich_pop = population_quantiles_10richest
                        
                        # labels
                        ax1_xticks = [-4,-8,-12]
                        ax1_xtick_labels = ["4","8","12"]
                        
                        ax2_xticks = [4,8,12]
                        ax2_xtick_labels = ax1_xtick_labels             
                        
                    elif qntl_range == '20':
                        
                        poor_unprec = unprec_pop_quantiles_20poorest
                        poor_std = unprec_pop_std_20poorest
                        poor_pop = population_quantiles_20poorest
                        rich_unprec = unprec_pop_quantiles_20richest
                        rich_std = unprec_pop_std_20richest
                        rich_pop = population_quantiles_20richest    
                        
                        # labels
                        ax1_xticks = [-5,-10,-15,-20,-25]
                        ax1_xtick_labels = ["5","10","15","20","25"]
                        
                        ax2_xticks = [5,10,15,20,25]
                        ax2_xtick_labels = ax1_xtick_labels
                    
                        
                    vln_type='grdi'
                    pyramid_plot(
                        df_GMT_strj,
                        poor_pop,
                        poor_unprec,
                        poor_std,
                        rich_pop,
                        rich_unprec,
                        rich_std,
                        qntl_range,
                        ax1_xticks,
                        ax1_xtick_labels,
                        ax2_xticks,
                        ax2_xtick_labels,
                        vln_type,
                    )

#%% ----------------------------------------------------------------
# main text plots
# ------------------------------------------------------------------       

if flags['plot_ms']:

    from plot_ms import *

    # f1 of ms, conceptual figure of city grid cell
    plot_conceptual(
        da_cohort_size,
        countries_mask,
        countries_regions,
        d_isimip_meta,
        flags,
        df_life_expectancy_5,
    )

    # f2 of ms, combined heatwave plot
    plot_combined_piechart(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
        df_countries,
    )
    
    # # f2 alternative with both absolute pops below box plots and pie charts
    # plot_combined_population_piechart(
    #     df_GMT_strj,
    #     ds_pf_gs,
    #     da_gs_popdenom,
    #     gdf_country_borders,
    #     sims_per_step,
    #     flags,
    #     df_countries,
    # )    
    
    # # f2 alternative with absolute pops below box plots and no pie charts
    # plot_combined_population(
    #     df_GMT_strj,
    #     ds_pf_gs,
    #     da_gs_popdenom,
    #     gdf_country_borders,
    #     sims_per_step,
    #     flags,
    #     df_countries,
    # )        

    # f3 of heatmaps across all hazards
    plot_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )

    # f4 of emergence union plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_emergence_union(
        grid_area,
        da_emergence_mean,
    )

    # f4 alternative for hexagons and multiple thresholds
    plot_hexagon_multithreshold(
        d_global_emergence,
    )    

#%% ----------------------------------------------------------------
# supplementary text plots
# ------------------------------------------------------------------  

if flags['plot_si']:

    from plot_si import *

    # pf time series for 2020 birth year across GMTs
    plot_pf_gmt_tseries_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )
    
    # pf time series for 2.7 degree world across birth years
    plot_pf_by_tseries_allhazards(
        flags,
        df_GMT_strj,
        da_gs_popdenom,
    )   
    
    # pf box plots for 1.5, 2.5 and 3.5 degree world across birth years
    plot_boxplots_allhazards(
        da_gs_popdenom,
        df_GMT_strj,
        flags,
    )     
    
    # pf maps for 1..5, 2.5, 3.5 for all hazards
    plot_pf_maps_allhazards(
        da_gs_popdenom,
        gdf_country_borders,
        flags,
    )    
    
    # emergence fraction plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_emergence_fracs(
        grid_area,
        da_emergence_mean,
    )    
    
    # plot locations where exposure occurs at all in our dataset
    plot_exposure_locations(
        grid_area,
        countries_mask,
        flags,
    )    
    
    # plot tseries box plots for 1.5, 2.5 and 3.5 when denominator contrained by exposure extent
    plot_geoconstrained_boxplots(
        flags,
    )    

    # plot gmt pathways of rcps and ar6
    plot_gmt_pathways(
        df_GMT_strj,
        d_isimip_meta,
    )
    
    # plot heatmaps of pf for country level emergence
    plot_heatmaps_allhazards_countryemergence(
        df_GMT_strj,
        flags,
    )     
    
    # plot pie charts of all hazards
    plot_allhazards_piecharts(
        da_gs_popdenom,
        df_countries,
        flags,
    )
    
    # plot cohort sizes in stacked bar chart
    plot_cohort_sizes(
        df_countries,
        da_gs_popdenom,
    )    
    
    # plot hexagon landfracs (will change to only show landfracs for SI)
    plot_hexagon_landfrac(
        d_global_emergence,
    )    
    

#%% ----------------------------------------------------------------
# sample analytics for paper
# ------------------------------------------------------------------

if flags['reporting']:
    
    from reporting import *
    
    # estimates of land area and (potential) pf for 1960 and 2020 emergencve of multiple hazards
    multi_hazard_emergence(
        grid_area,
        da_emergence_mean,
        da_gs_popdenom,
    )
    
    # get birth year cohort sizes at grid scale
    gridscale_cohort_sizes(
        da_population,
        gridscale_countries,   
    )    
    
    # per hazard, locations where exposure occurs across whole ensemble
    exposure_locs(
        grid_area,
    )
    
    # per run for 1.5, 2.5, 2.7 and 3.5, collect maps of emergence locations to be used in geographically constrained pf estimates
    emergence_locs_perrun(
        flags,
        grid_area,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )    
    
    # compute geographically constrained pf
    pf_geoconstrained()
    
    # print geographically constrained pf vs regular pf
    print_pf_geoconstrained(
        flags,
        da_gs_popdenom,
    )    

    # checking for signifiance of change in means between 1960 and 2020 pf per event and for a GMT level
    paired_ttest(
        flags,
        da_gs_popdenom,
    )
    
    # print latex table on ensemble members per hazard
    print_latex_table_ensemble_sizes(
        flags,
        df_GMT_strj,
    )   
    
    # children living unprec exposure between 1.5 and 2.7 degrees warming
    print_millions_excess(
        flags,
        df_GMT_strj,
    )     

    # print pf info    
    print_pf_ratios(
        df_GMT_strj,
        da_gs_popdenom,
    )    
    
    # get number of million people unprecedented: (will change this stuff to run for all extremes and birth years for table in paper)
    print_absolute_unprecedented(
        ds_pf_gs,
    )
 
    # get cities that work for 
    find_valid_cities(
        df_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        d_isimip_meta,
        flags,
    )
    
    # latex tables of CF per extr and GMT pathway
    print_latex_table_unprecedented(
        flags,
        da_gs_popdenom,
    )    
    
    print_latex_table_unprecedented_sideways(
        flags,
        da_gs_popdenom,
    )    
