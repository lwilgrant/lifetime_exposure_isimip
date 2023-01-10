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

#%% ----------------------------------------------------------------
# bootstrapping function 
# ------------------------------------------------------------------

def resample(
    da, 
    resample_dim,
    life_extent,
):
    """Resample with replacement in dimension ``resample_dim``. https://climpred.readthedocs.io/en/stable/_modules/climpred/bootstrap.html

    Args:
        initialized (xr.Dataset): input xr.Dataset to be resampled.
        resample_dim (str): dimension to resample along.
        life_extent (int): number of years per lifetime
        
    Returns:
        xr.Dataset: resampled along ``resample_dim``.

    """
    to_be_resampled = da[resample_dim].values
    smp = np.random.choice(to_be_resampled, life_extent)
    smp_da = da.sel({resample_dim: smp})
    smp_da[resample_dim] = np.arange(1960,1960+life_extent)
    return smp_da

#%% ----------------------------------------------------------------
# grid scale
# ------------------------------------------------------------------

def grid_scale_emergence(
    d_isimip_meta,
    d_pic_meta,
    flag_extr,
    d_all_cohorts,
    df_countries,
    countries_regions,
    countries_mask,
    df_life_expectancy_5,
    GMT_indices,
    da_population,
):
    
    # cohort conversion to data array (again)
    da_cohort_size = xr.DataArray(
        np.asarray([v for k,v in d_all_cohorts.items() if k in list(df_countries['name'])]),
        coords={
            'country': ('country', list(df_countries['name'])),
            'time': ('time', year_range),
            'age': ('age', np.arange(104,-1,-1)),
        },
        dims=[
            'country',
            'time',
            'age',
        ]
    )

    # lifetime exposure dataset (pop weighted mean of pixel scale lifetime exposure per country, run, GMT and birthyear)
    ds_le = xr.Dataset(
        data_vars={
            'lifetime_exposure_popweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'lifetime_exposure_latweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            )            
        },
        coords={
            'country': ('country', sample_countries),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )

    # age emergence dataset (pop weighted mean of pixel scale age emergence per country, run, GMT and birthyear)
        # spatial samples can go to ds_spatial
    ds_ae = xr.Dataset(
        data_vars={
            'age_emergence_popweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'age_emergence_latweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            )            
        },
        coords={
            'country': ('country', sample_countries),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )

    # pop fraction dataset (sum of unprecedented exposure pixels' population per per country, run, GMT and birthyear)
    ds_pf = xr.Dataset(
        data_vars={
            'unprec': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_fraction': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(sample_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            )        
        },
        coords={
            'country': ('country', sample_countries),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )

    for cntry in sample_countries:

        print(cntry)
        da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
        da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='age') # cohort relative sizes in sample country
        da_cntry = xr.DataArray(
            np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
            dims=countries_mask.dims,
            coords=countries_mask.coords,
        )        
        
        # spatial lifetime exposure dataset (subsetting birth years and GMT steps to reduce data load) per country
            # can also add spatial age emergence to here
        ds_spatial = xr.Dataset(
            data_vars={
                'lifetime_exposure': (
                    ['run','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(list(d_isimip_meta.keys())),len(GMT_indices),len(sample_birth_years),len(da_cntry.lat.data),len(da_cntry.lon.data)),
                        fill_value=np.nan,
                    ),
                ),
                'age_emergence': (
                    ['run','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(list(d_isimip_meta.keys())),len(GMT_indices),len(sample_birth_years),len(da_cntry.lat.data),len(da_cntry.lon.data)),
                        fill_value=np.nan,
                    ),
                ),
                'population_emergence': (
                    ['run','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(list(d_isimip_meta.keys())),len(GMT_indices),len(sample_birth_years),len(da_cntry.lat.data),len(da_cntry.lon.data)),
                        fill_value=np.nan,
                    ),
                ),
                'emergence_mask': (
                    ['run','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(list(d_isimip_meta.keys())),len(GMT_indices),len(sample_birth_years),len(da_cntry.lat.data),len(da_cntry.lon.data)),
                        fill_value=np.nan,
                    ),
                )
            },
            coords={
                'lat': ('lat', da_cntry.lat.data),
                'lon': ('lon', da_cntry.lon.data),
                'birth_year': ('birth_year', sample_birth_years),
                'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
                'GMT': ('GMT', GMT_indices)
            }
        )
        
        # weights for latitude (probably won't use but will use population instead)
        lat_weights = np.cos(np.deg2rad(da_cntry.lat))
        lat_weights.name = "weights"
        da_smple_pop = da_population.where(da_cntry==1) * da_smple_cht_prp # use pop and relative cohort sizes to get people per cohort

        # demography dataset
        ds_dmg = xr.Dataset(
            data_vars={
                'life_expectancy': (
                    ['birth_year'],
                    df_life_expectancy_5[cntry].values
                ),
                'death_year': (
                    ['birth_year'],
                    np.floor(df_life_expectancy_5[cntry].values + df_life_expectancy_5[cntry].index).astype('int')
                ),
                'population': (
                    ['time','lat','lon','age'],
                    da_smple_pop.data
                ),
                'country_extent': (
                    ['lat','lon'],
                    da_cntry.data
                ),
            },
            coords={
                'birth_year': ('birth_year', birth_years),
                'time': ('time', da_population.time.data),
                'lat': ('lat', da_cntry.lat.data),
                'lon': ('lon', da_cntry.lon.data),
                'age': ('age', np.arange(104,-1,-1)),
            }
        )

        # get birthyear aligned population for unprecedented calculation (population_by), also use for weighted mean of lifetime exposure and age emergence
        bys = []
        for by in birth_years:
                
            time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
            ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
            data = ds_dmg['population'].sel(time=time,age=ages) # paired selections
            data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
            data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
            data = data.assign_coords({'birth_year':by}).drop_vars('age')
            bys.append(data)

        ds_dmg['population_by'] = xr.concat(bys,dim='birth_year').sum(dim='time').where(ds_dmg['country_extent']==1)

        # pic dataset
        ds_pic = xr.Dataset(
            data_vars={
                'lifetime_exposure': (
                    ['lifetimes','lat','lon'],
                    np.full(
                        (len(list(d_pic_meta.keys())*nboots),len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                        fill_value=np.nan,
                    ),
                )
            },
            coords={
                'lat': ('lat', da_cntry.lat.data),
                'lon': ('lon', da_cntry.lon.data),
                'lifetimes': ('lifetimes', np.arange(len(list(d_pic_meta.keys())*nboots))),
            }
        )
        
        # check for PIC pickle file (for ds_pic); process and dump pickle if not already existing
        if not os.path.isfile('./data/pickles/gridscale_le_pic_{}_{}.pkl'.format(flag_extr,cntry)):
            
            # loop over PIC simulations
            c = 0
            for i in list(d_pic_meta.keys()):
                
                print('simulation {} of {}'.format(i,len(d_pic_meta)))
                
                # load AFA data of that run
                with open('./data/pickles/isimip_AFA_pic_{}_{}.pkl'.format(flag_extr,str(i)), 'rb') as f:
                    da_AFA_pic = pk.load(f)
                    
                da_AFA_pic = da_AFA_pic.where(ds_dmg['country_extent']==1,drop=True)
                
                # resample 100 lifetimes and then sum 
                da_exposure_pic = xr.concat(
                    [resample(da_AFA_pic,resample_dim,pic_life_extent) for i in range(nboots)],
                    dim='lifetimes'    
                ).assign_coords({'lifetimes':np.arange(c*nboots,c*nboots+nboots)})
                
                # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                da_pic_le = da_exposure_pic.loc[
                    {'time':np.arange(pic_by,ds_dmg['death_year'].sel(birth_year=pic_by).item()+1)}
                ].sum(dim='time') +\
                    da_exposure_pic.loc[{'time':ds_dmg['death_year'].sel(birth_year=pic_by).item()+1}].drop('time') *\
                        (ds_dmg['life_expectancy'].sel(birth_year=pic_by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=pic_by).item()))
                        
                ds_pic['lifetime_exposure'].loc[
                    {
                        'lat': da_pic_le.lat.data,
                        'lon': da_pic_le.lon.data,
                        'lifetimes': np.arange(c*nboots,c*nboots+nboots),
                    }
                ] = da_pic_le
                c += 1
                
            # pic extreme lifetime exposure definition
            ds_pic['99.99'] = ds_pic['lifetime_exposure'].quantile(
                    q=pic_qntl,
                    dim='lifetimes',
                )
            
            # pickle PIC for var, country
            with open('./data/pickles/gridscale_le_pic_{}_{}.pkl'.format(flag_extr,cntry), 'wb') as f:
                pk.dump(ds_pic,f)
                
        else:
            
            # load PIC pickle
            with open('./data/pickles/gridscale_le_pic_{}_{}.pkl'.format(flag_extr,cntry), 'rb') as f:
                ds_pic = pk.load(f)        

        # loop over simulations
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/pickles/isimip_AFA_{}_{}.pkl'.format(flag_extr,str(i)), 'rb') as f:
                da_AFA = pk.load(f)
                
            # mask to sample country and reduce spatial extent
            da_AFA = da_AFA.where(ds_dmg['country_extent']==1,drop=True)
            
            for step in GMT_labels:
                
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    
                    # check for pickle of gridscale lifetime exposure (da_le); process if not existing
                    if not os.path.isfile('./data/pickles/gridscale_le_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step)):
                        
                        da_AFA = da_AFA.reindex(
                            {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                        ).assign_coords({'time':year_range}) 
                            
                        # simple lifetime exposure sum
                        da_le = xr.concat(
                            [(da_AFA.loc[{'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1)}].sum(dim='time') +\
                            da_AFA.sel(time=ds_dmg['death_year'].sel(birth_year=by).item()+1).drop('time') *\
                            (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item()))\
                            for by in birth_years],
                            dim='birth_year',
                        ).assign_coords({'birth_year':birth_years})
                        
                        # dump spatial lifetime exposure for this country/run/GMT
                        with open('./data/pickles/gridscale_le_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step), 'wb') as f:
                            pk.dump(da_le,f)
                    
                    # load existing pickle
                    else:
                        
                        with open('./data/pickles/gridscale_le_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step), 'rb') as f:
                            da_le = pk.load(f)
                
                    # assign lifetime exposure array to spatial dataset for run i and for subset of birth years if this corresponds to step in GMT_indices (lb, 1.5, 2.0, NDC, 3.0 or 4.0)
                    if step in GMT_indices:
                        
                        ds_spatial['lifetime_exposure'].loc[{
                            'run':i,
                            'GMT':step,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,
                        }] = da_le.loc[{
                            'birth_year':sample_birth_years,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,                            
                        }]
                    
                    # assign pop weighted mean exposure to dataset
                    ds_le['lifetime_exposure_popweight'].loc[
                        {
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }
                    ] = da_le.weighted(ds_dmg['population_by'].fillna(0)).mean(('lat','lon'))
                    
                    # assign lat weighted mean exposure to dataset
                    ds_le['lifetime_exposure_latweight'].loc[
                        {
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }
                    ] = da_le.weighted(lat_weights).mean(('lat','lon'))                    
                        
                    da_exp_py_pa = da_AFA * xr.full_like(ds_dmg['population'],1)
                    bys = []
            
                    # to be new func, per birth year, make (year,age) selections
                    for by in birth_years:
                            
                        time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                        ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                        data = da_exp_py_pa.sel(time=time,age=ages) # paired selections
                        data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                        data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                        data = data.assign_coords({'birth_year':by}).drop_vars('age')
                        data.loc[
                            {'time':ds_dmg['death_year'].sel(birth_year=by).item()+1}
                        ] = da_AFA.loc[{'time':ds_dmg['death_year'].sel(birth_year=by).item()+1}] *\
                            (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item())
                        bys.append(data)
            
                    da_exp_py_pa = xr.concat(bys,dim='birth_year')
                            
                    # cumulative sum per birthyear
                    da_exp_py_pa_cumsum = da_exp_py_pa.cumsum(dim='time')
                
                    # check for pickles of gridscale exposure emergence mask and age emergence
                    if not os.path.isfile('./data/pickles/gridscale_exposure_mask_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step)) or not os.path.isfile('./data/pickles/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step)):
                        
                        # generate exposure mask for timesteps after reaching pic extreme to find age of emergence
                        da_age_exposure_mask = xr.where(
                            da_exp_py_pa_cumsum >= ds_pic['99.99'],
                            1,
                            0,
                        )
                        da_age_emergence = da_age_exposure_mask * (da_age_exposure_mask.time - da_age_exposure_mask.birth_year)
                        da_age_emergence = da_age_emergence.where(da_age_emergence!=0).min(dim='time',skipna=True)
                            
                        # find birth years/pixels crossing threshold
                        da_birthyear_exposure_mask = xr.where(da_age_exposure_mask.sum(dim='time')>0,1,0) 
                        
                        with open('./data/pickles/gridscale_exposure_mask_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step), 'wb') as f:
                            pk.dump(da_birthyear_exposure_mask,f)
                            
                        with open('./data/pickles/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step), 'wb') as f:
                            pk.dump(da_age_emergence,f)
                            
                    # load existing pickles
                    else: 
                        
                        with open('./data/pickles/gridscale_exposure_mask_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step), 'rb') as f:
                            da_birthyear_exposure_mask = pk.load(f)
                        with open('./data/pickles/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flag_extr,cntry,i,step), 'rb') as f:
                            da_age_emergence = pk.load(f)
                            
                    # grid cells of population emerging
                    da_unprec_pop = ds_dmg['population_by'].where(da_birthyear_exposure_mask==1)
                            
                    if step in GMT_indices:
                        
                        ds_spatial['age_emergence'].loc[{
                            'run':i,
                            'GMT':step,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,
                        }] = da_age_emergence.loc[{
                            'birth_year':sample_birth_years,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,                            
                        }]
                        
                        ds_spatial['population_emergence'].loc[{
                            'run':i,
                            'GMT':step,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,                            
                        }] = da_unprec_pop.loc[{
                            'birth_year':sample_birth_years,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,                               
                        }]
                        ds_spatial['emergence_mask'].loc[{
                            'run':i,
                            'GMT':step,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,
                        }] = da_birthyear_exposure_mask.loc[{
                            'birth_year':sample_birth_years,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,
                        }]
                    
                    # assign mean/sum age emergence/pop unprec
                    ds_ae['age_emergence_popweight'].loc[{
                        'country':cntry,
                        'run':i,
                        'GMT':step,
                        'birth_year':birth_years,
                    }] = da_age_emergence.weighted(ds_dmg['population_by'].fillna(0)).mean(('lat','lon'))
                    
                    ds_ae['age_emergence_latweight'].loc[{
                        'country':cntry,
                        'run':i,
                        'GMT':step,
                        'birth_year':birth_years,
                    }] = da_age_emergence.weighted(lat_weights).mean(('lat','lon'))                    
                    
                    ds_pf['unprec'].loc[{
                        'country':cntry,
                        'run':i,
                        'GMT':step,
                        'birth_year':birth_years,
                    }] = da_unprec_pop.sum(('lat','lon'))

                    ds_pf['unprec_fraction'].loc[{
                        'country':cntry,
                        'run':i,
                        'GMT':step,
                        'birth_year':birth_years,
                    }] = da_unprec_pop.sum(('lat','lon')) / ds_dmg['population_by'].sum(('lat','lon'))
                    
        # pickle spatially explicit dataset with subset of birth years and GMTs
        with open('./data/pickles/gridscale_spatially_explicit_{}_{}.pkl'.format(flag_extr,cntry), 'wb') as f:
            pk.dump(ds_spatial,f)

    # pickle aggregated lifetime exposure, age emergence and pop frac datasets
    with open('./data/pickles/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flag_extr), 'wb') as f:
        pk.dump(ds_le,f)    
    with open('./data/pickles/gridscale_aggregated_age_emergence_{}.pkl'.format(flag_extr), 'wb') as f:
        pk.dump(ds_ae,f)
    with open('./data/pickles/gridscale_aggregated_pop_frac_{}.pkl'.format(flag_extr), 'wb') as f:
        pk.dump(ds_pf,f)
        
    return ds_le, ds_ae, ds_pf