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
import regionmask as rm
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

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
# function for returning countries for grid scale analysis
# ------------------------------------------------------------------

def get_gridscale_regions(
    grid_area,
    flags,
    gdf_country_borders,
):

    lat = grid_area.lat.values
    lon = grid_area.lon.values  
    
    if flags['gridscale_country_subset']:
    
        # take all countries for heatwavedarea
        if flags['extr'] == 'heatwavedarea':
            
            gdf_country = gdf_country_borders.loc[:,'geometry']
            list_countries = gdf_country.index.values

        # countries in Mediterranean for drought
        elif flags['extr'] == 'driedarea': 
            
            med = 'Mediterranean'
            gdf_ar6 = gpd.read_file('./data/shapefiles/IPCC-WGI-reference-regions-v4.shp')
            gdf_ar6 = gdf_ar6.loc[gdf_ar6['Name']==med]
            gdf_ar6 = gdf_ar6.loc[:,['Name','geometry']]
            gdf_ar6 = gdf_ar6.rename(columns={'Name':'name'})
            gdf_country = gdf_country_borders.loc[:,'geometry']
            gdf_country = gdf_country.reset_index()
            gdf_med_countries = gdf_country.loc[gdf_country.intersects(gdf_ar6['geometry'].iloc[0])]
            countries_med_3D = rm.mask_3D_geopandas(gdf_med_countries,lon,lat)
            ar6_regs_3D = rm.defined_regions.ar6.land.mask_3D(lon,lat)
            med_3D = ar6_regs_3D.isel(region=(ar6_regs_3D.names == 'Mediterranean')).squeeze()
            
            c_valid = []
            for c in gdf_med_countries.index:
                # next line gives nans outside country, 
                # 1 in parts of country in AR6 Medit
                # and 0 in parts outside Medit.
                c_in_med = med_3D.where(countries_med_3D.sel(region=c)==1) 
                c_area_in_med = c_in_med.weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area = countries_med_3D.sel(region=c).weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area_frac = c_area_in_med.item() / c_area.item()
                if c_area_frac > 0.5:
                    c_valid.append(c)
            countries_med_3D = countries_med_3D.loc[{'region':c_valid}]
            gdf_med_countries = gdf_med_countries.loc[c_valid]
            list_countries = gdf_med_countries['name'].values
            
        # countries in the Nile for floods
        elif flags['extr'] == 'floodedarea':
            
            basin = 'Nile'
            gdf_basins = gpd.read_file('./data/shapefiles/Major_Basins_of_the_World.shp')
            gdf_basins = gdf_basins.loc[:,['NAME','geometry']]
            gdf_basin = gdf_basins.loc[gdf_basins['NAME']==basin]
            gdf_basin = gdf_basin.rename(columns={'Name':'name'})
            if len(gdf_basin.index) > 1:
                gdf_basin = gdf_basin.dissolve()
            gdf_country = gdf_country_borders.loc[:,'geometry']
            gdf_country = gdf_country.reset_index()
            gdf_basin_countries = gdf_country.loc[gdf_country.intersects(gdf_basin['geometry'].iloc[0])]
            countries_basin_3D = rm.mask_3D_geopandas(gdf_basin_countries,lon,lat)
            basin_3D = rm.mask_3D_geopandas(gdf_basin,lon,lat)
            
            c_valid = []
            for c in gdf_basin_countries.index:
                # next line gives nans outside country, 
                # 1 in parts of country in basin
                # and 0 in parts outside basin.
                c_in_basin = basin_3D.where(countries_basin_3D.sel(region=c)==1) 
                c_area_in_basin = c_in_basin.weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area = countries_basin_3D.sel(region=c).weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area_frac = c_area_in_basin.item() / c_area.item()
                if c_area_frac > 0.3:
                    c_valid.append(c)
            countries_basin_3D = countries_basin_3D.loc[{'region':c_valid}]
            gdf_basin_countries = gdf_basin_countries.loc[c_valid]
            list_countries = gdf_basin_countries['name'].values
            list_countries = np.sort(np.insert(list_countries,-1,'Egypt'))
            
    else:
        
        gdf_country = gdf_country_borders.loc[:,'geometry']
        list_countries = gdf_country.index.values        
            
    return list_countries

#%% ----------------------------------------------------------------
# grid scale
# ------------------------------------------------------------------

def gridscale_emergence(
    d_isimip_meta,
    d_pic_meta,
    flags,
    list_countries,
    da_cohort_size,
    countries_regions,
    countries_mask,
    df_life_expectancy_5,
    GMT_indices,
    da_population,
):

    # lifetime exposure dataset (pop weighted mean of pixel scale lifetime exposure per country, run, GMT and birthyear)
    ds_le = xr.Dataset(
        data_vars={
            'lifetime_exposure_popweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'lifetime_exposure_latweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            )            
        },
        coords={
            'country': ('country', list_countries),
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
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'age_emergence_latweight': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            )            
        },
        coords={
            'country': ('country', list_countries),
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
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_fraction': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            )        
        },
        coords={
            'country': ('country', list_countries),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )

    for cntry in list_countries:

        print(cntry)
        da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
        da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
        da_cntry = xr.DataArray(
            np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
            dims=countries_mask.dims,
            coords=countries_mask.coords,
        )
        da_cntry = da_cntry.where(da_cntry,drop=True)
        
        # spatial lifetime exposure dataset (subsetting birth years and GMT steps to reduce data load) per country
            # can also add spatial age emergence to here
        if not os.path.isfile('./data/pickles/{}/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry)):
            
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
        
        # check for demography pickle file
        if not os.path.isfile('./data/pickles/gridscale_dmg_{}.pkl'.format(cntry)):
            
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
                    'age': ('age', np.arange(100,-1,-1)),
                }
            )

            # get birthyear aligned population for unprecedented calculation (by_population), also use for weighted mean of lifetime exposure and age emergence
            bys = []
            for by in birth_years:
                    
                time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = ds_dmg['population'].sel(time=time,age=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('age')
                bys.append(data)

            ds_dmg['by_population'] = xr.concat(bys,dim='birth_year').sum(dim='time').where(ds_dmg['country_extent']==1)
            # adding by_population_y0 to rep cohort 
            ds_dmg['by_population_y0'] = xr.concat(bys,dim='birth_year').where(ds_dmg['country_extent']==1)
            da_times=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')
            da_birth_years=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')        
            ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].sel(time=da_times,birth_year=da_birth_years)
            # set order of dims
            ds_dmg['by_population'] = ds_dmg['by_population'].transpose('birth_year','lat','lon')
            ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].transpose('birth_year','lat','lon')
            
            # pickle gridscale demography for country
            with open('./data/pickles/gridscale_dmg_{}.pkl'.format(cntry), 'wb') as f:
                pk.dump(ds_dmg,f)       
                
        else:
            
            # load demography pickle
            with open('./data/pickles/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
                ds_dmg = pk.load(f)                      
        
        # check for PIC pickle file (for ds_pic); process and dump pickle if not already existing
        if not os.path.isfile('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry)):
            
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
            
            # loop over PIC simulations
            c = 0
            for i in list(d_pic_meta.keys()):
                
                print('simulation {} of {}'.format(i,len(d_pic_meta)))
                
                # load AFA data of that run
                with open('./data/pickles/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
                    da_AFA_pic = pk.load(f)
                    
                da_AFA_pic = da_AFA_pic.where(ds_dmg['country_extent']==1,drop=True)
                
                # regular boot strapping for all reasonably sized countries (currently only exception is for Russia, might expand this for hazards with more sims)
                if not cntry in ['Russian Federation','Canada','United States','China']:

                    # resample 10000 lifetimes and then sum 
                    da_exposure_pic = xr.concat(
                        [resample(da_AFA_pic,resample_dim,pic_life_extent) for i in range(nboots)],
                        dim='lifetimes'    
                    ).assign_coords({'lifetimes':np.arange(c*nboots,c*nboots+nboots)})
                    
                    # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                    da_pic_le = da_exposure_pic.loc[
                        {'time':np.arange(pic_by,ds_dmg['death_year'].sel(birth_year=pic_by).item()+1)}
                    ].sum(dim='time') +\
                        da_exposure_pic.loc[{'time':ds_dmg['death_year'].sel(birth_year=pic_by).item()}].drop('time') *\
                            (ds_dmg['life_expectancy'].sel(birth_year=pic_by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=pic_by).item()))
                            
                    ds_pic['lifetime_exposure'].loc[
                        {
                            'lat': da_pic_le.lat.data,
                            'lon': da_pic_le.lon.data,
                            'lifetimes': np.arange(c*nboots,c*nboots+nboots),
                        }
                    ] = da_pic_le
                    c += 1
                   
                # for exceptionally sized countries, do piecewise boot strapping (nboots in 10 steps)
                else:
                    
                    for bstep in range(10):
                        
                        # resample 1000 lifetimes and then sum 
                        da_exposure_pic = xr.concat(
                            [resample(da_AFA_pic,resample_dim,pic_life_extent) for i in range(int(nboots/10))],
                            dim='lifetimes'    
                        ).assign_coords({'lifetimes':np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10)})       
                        
                        # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                        da_pic_le = da_exposure_pic.loc[
                            {'time':np.arange(pic_by,ds_dmg['death_year'].sel(birth_year=pic_by).item()+1)}
                        ].sum(dim='time') +\
                            da_exposure_pic.loc[{'time':ds_dmg['death_year'].sel(birth_year=pic_by).item()}].drop('time') *\
                                (ds_dmg['life_expectancy'].sel(birth_year=pic_by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=pic_by).item()))
                                
                        ds_pic['lifetime_exposure'].loc[
                            {
                                'lat': da_pic_le.lat.data,
                                'lon': da_pic_le.lon.data,
                                'lifetimes': np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10),
                            }
                        ] = da_pic_le
                        
                    c += 1
                                
            # pic extreme lifetime exposure definition
            ds_pic['99.99'] = ds_pic['lifetime_exposure'].quantile(
                    q=pic_qntl,
                    dim='lifetimes',
                )
            
            # pickle PIC for var, country
            with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic,f)
                
        else:
            
            # load PIC pickle
            with open('./data/pickles/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
                ds_pic = pk.load(f)        

        # loop over simulations
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/pickles/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
                da_AFA = pk.load(f)
                
            # mask to sample country and reduce spatial extent
            da_AFA = da_AFA.where(ds_dmg['country_extent']==1,drop=True)
            
            for step in GMT_labels:
                
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    
                    da_AFA_step = da_AFA.reindex(
                        {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                    ).assign_coords({'time':year_range})                     
                    
                    # check for pickle of gridscale lifetime exposure (da_le); process if not existing
                    if not os.path.isfile('./data/pickles/{}/gridscale_le_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step)):
                            
                        # simple lifetime exposure sum
                        da_le = xr.concat(
                            [(da_AFA_step.loc[{'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1)}].sum(dim='time') +\
                            da_AFA_step.sel(time=ds_dmg['death_year'].sel(birth_year=by).item()).drop('time') *\
                            (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item()))\
                            for by in birth_years],
                            dim='birth_year',
                        ).assign_coords({'birth_year':birth_years})
                        
                        # dump spatial lifetime exposure for this country/run/GMT
                        with open('./data/pickles/{}/gridscale_le_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step), 'wb') as f:
                            pk.dump(da_le,f)
                    
                    # load existing pickle
                    else:
                        
                        with open('./data/pickles/{}/gridscale_le_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step), 'rb') as f:
                            da_le = pk.load(f)
                
                    # assign lifetime exposure array to spatial dataset for run i and for subset of birth years if this corresponds to step in GMT_indices (lb, 1.5, 2.0, NDC, 3.0 or 4.0)
                    if step in GMT_indices:
                        
                        if not os.path.isfile('./data/pickles/{}/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry)):
                        
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
                    ] = da_le.weighted(ds_dmg['by_population_y0'].fillna(0)).mean(('lat','lon')) # why is there a fillna here?
                    
                    # assign lat weighted mean exposure to dataset
                    ds_le['lifetime_exposure_latweight'].loc[
                        {
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }
                    ] = da_le.weighted(lat_weights).mean(('lat','lon'))       
                    
                    # check for pickles of gridscale exposure emergence mask and age emergence
                    if not os.path.isfile('./data/pickles/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step)) or not os.path.isfile('./data/pickles/{}/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step)):                                 
                        
                        da_exp_py_pa = da_AFA_step * xr.full_like(ds_dmg['population'],1)
                        bys = []
                
                        # to be new func, per birth year, make (year,age) selections
                        for by in birth_years:
                                
                            time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                            ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                            data = da_exp_py_pa.sel(time=time,age=ages) # paired selections
                            data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                            data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                            # when I reindex time, the len=1 lat coord for cyprus, a small country, disappears
                            # therefore need to reintroduce len=1 coord at correct position
                            for scoord in ['lat','lon']:
                                if data[scoord].size == 1:
                                    if scoord == 'lat':
                                        a_pos = 1
                                    elif scoord == 'lon':
                                        a_pos = 2
                                    data = data.expand_dims(dim={scoord:1},axis=a_pos).copy()
                            data = data.assign_coords({'birth_year':by}).drop_vars('age')
                            data.loc[
                                {'time':ds_dmg['death_year'].sel(birth_year=by).item()}
                            ] = da_AFA_step.loc[{'time':ds_dmg['death_year'].sel(birth_year=by).item()}] *\
                                (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item())
                            bys.append(data)
                
                        da_exp_py_pa = xr.concat(bys,dim='birth_year')
                                
                        # cumulative sum per birthyear (in emergence.py, this cumsum then has .where(==0), should I add this here too?)
                        da_exp_py_pa_cumsum = da_exp_py_pa.cumsum(dim='time')
                
                    # # check for pickles of gridscale exposure emergence mask and age emergence
                    # if not os.path.isfile('./data/pickles/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step)) or not os.path.isfile('./data/pickles/{}/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step)):
                        
                        # generate exposure mask for timesteps after reaching pic extreme to find age of emergence
                        da_emergence_mask = xr.where(
                            da_exp_py_pa_cumsum > ds_pic['99.99'],
                            1,
                            0,
                        )
                        da_age_emergence = da_emergence_mask * (da_emergence_mask.time - da_emergence_mask.birth_year)
                        da_age_emergence = da_age_emergence.where(da_age_emergence!=0).min(dim='time',skipna=True)
                            
                        # find birth years/pixels crossing threshold
                        da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0) 
                        
                        with open('./data/pickles/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step), 'wb') as f:
                            pk.dump(da_birthyear_emergence_mask,f)
                            
                        with open('./data/pickles/{}/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step), 'wb') as f:
                            pk.dump(da_age_emergence,f)
                            
                    # load existing pickles
                    else: 
                        
                        with open('./data/pickles/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step), 'rb') as f:
                            da_birthyear_emergence_mask = pk.load(f)
                        with open('./data/pickles/{}/gridscale_age_emergence_{}_{}_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry,i,step), 'rb') as f:
                            da_age_emergence = pk.load(f)
                            
                    # grid cells of population emerging
                    da_unprec_pop = ds_dmg['by_population_y0'].where(da_birthyear_emergence_mask==1)
                            
                    if step in GMT_indices:
                        
                        if not os.path.isfile('./data/pickles/{}/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry)):
                        
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
                            }] = da_birthyear_emergence_mask.loc[{
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
                    }] = da_age_emergence.weighted(ds_dmg['by_population_y0'].fillna(0)).mean(('lat','lon'))
                    
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
                    }] = da_unprec_pop.sum(('lat','lon')) / ds_dmg['by_population_y0'].sum(('lat','lon'))
                    
        # pickle spatially explicit dataset with subset of birth years and GMTs
        if not os.path.isfile('./data/pickles/{}/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry)):
            
            with open('./data/pickles/{}/gridscale_spatially_explicit_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_spatial,f)

    # pickle aggregated lifetime exposure, age emergence and pop frac datasets
    with open('./data/pickles/{}/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_le,f)    
    with open('./data/pickles/{}/gridscale_aggregated_age_emergence_{}.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_ae,f)
    with open('./data/pickles/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_pf,f)
        
    return ds_le, ds_ae, ds_pf

#%% ----------------------------------------------------------------
# grid scale population per birthyear
# ------------------------------------------------------------------

def get_gridscale_popdenom(
    list_countries,
    da_cohort_size,
    countries_mask,
    countries_regions,
    da_population,
    df_life_expectancy_5,
):
    # need denominator for pf @ grid scale, which is not just wcde data in ds_cohorts, but rather isimip pop data x fractions in ds_cohorts
    cntry_pops = []
    for cntry in list_countries:
    
        print(cntry)
        da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
        da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
        da_cntry = xr.DataArray(
            np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
            dims=countries_mask.dims,
            coords=countries_mask.coords,
        )
        da_cntry = da_cntry.where(da_cntry,drop=True)
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
                'age': ('age', np.arange(100,-1,-1)),
            }
        )

        # get birthyear aligned population for unprecedented calculation (by_population), also use for weighted mean of lifetime exposure and age emergence
        bys = []
        for by in birth_years:
                
            time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
            ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
            data = ds_dmg['population'].sel(time=time,age=ages) # paired selections
            data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
            data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
            data = data.assign_coords({'birth_year':by}).drop_vars('age')
            bys.append(data)

        ds_dmg['by_population_y0'] = xr.concat(bys,dim='birth_year').where(ds_dmg['country_extent']==1)
        da_times=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')
        da_birth_years=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')        
        ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].sel(time=da_times,birth_year=da_birth_years)
        ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].transpose('birth_year','lat','lon')     
        cntry_pops.append(ds_dmg['by_population_y0'].sum(dim=('lat','lon')))   
    
    da_cntry_pops = xr.concat(cntry_pops,dim='country').assign_coords({'country':list_countries})     
    return da_cntry_pops

#%% ----------------------------------------------------------------
# grid scale emergence union
# ------------------------------------------------------------------

def get_gridscale_union(
    da_population,
    flags,
    gridscale_countries,
    countries_mask,
    countries_regions,
):
    
    if not os.path.isfile('./data/pickles/emergence_hazards_subset_new.pkl') or not os.path.isfile('./data/pickles/emergence_union_subset_new.pkl'):
        
        extremes = [
            'burntarea', 
            'cropfailedarea', 
            'driedarea', 
            'floodedarea', 
            'heatwavedarea', 
            'tropicalcyclonedarea',
        ]

        ds_emergence_union = xr.Dataset(
            data_vars={
                'emergence_mean': (
                    ['hazard','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(extremes),len(GMT_labels),len(birth_years),len(da_population.lat.data),len(da_population.lon.data)),
                        fill_value=np.nan,
                    ),
                ),                           
                'emergence_union': (
                    ['GMT','birth_year','lat','lon'],
                    np.full(
                        (len(GMT_labels),len(birth_years),len(da_population.lat.data),len(da_population.lon.data)),
                        fill_value=np.nan,
                    ),
                ),        
            },
            coords={
                'lat': ('lat', da_population.lat.data),
                'lon': ('lon', da_population.lon.data),
                'birth_year': ('birth_year', birth_years),
                'hazard': ('hazard', extremes),
                'GMT': ('GMT', GMT_labels),
            }
        )

        # loop through extremes
        for extr in extremes:
            
            # get metadata for extreme
            with open('./data/pickles/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
                d_isimip_meta = pk.load(f)
                
            sims_per_step = {}
            for step in GMT_labels:
                sims_per_step[step] = []
                print('step {}'.format(step))
                for i in list(d_isimip_meta.keys()):
                    if d_isimip_meta[i]['GMT_strj_valid'][step]:
                        sims_per_step[step].append(i)                
                
            # loop through countries
            for cntry in gridscale_countries:
                
                da_cntry = xr.DataArray(
                    np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                    dims=countries_mask.dims,
                    coords=countries_mask.coords,
                )
                da_cntry = da_cntry.where(da_cntry,drop=True)                  
                    
                # loop through GMT trajectories
                for step in GMT_labels:
                    
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
                            
                    # compute mean for extreme - country - GMT, assign into greater dataset for eventual union
                    da_loc_mean = ds_cntry_emergence['emergence'].loc[{
                        'run':sims_per_step[step],
                        'lat':da_cntry.lat.data,
                        'lon':da_cntry.lon.data,
                    }].where(da_cntry==1).mean(dim='run')
                    
                    ds_emergence_union['emergence_mean'].loc[{
                        'hazard':extr,
                        'GMT':step,
                        'birth_year':birth_years,
                        'lat':da_cntry.lat.data,
                        'lon':da_cntry.lon.data,
                    }] = xr.where(
                        da_loc_mean.notnull(),
                        da_loc_mean,
                        ds_emergence_union['emergence_mean'].loc[{'hazard':extr,'GMT':step,'birth_year':birth_years,'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                    )                    
                    
        # if mean greater than 0.5, consider emerged on average, sum across hazards for union        
        ds_emergence_union['emergence_union'].loc[{
            'GMT':GMT_labels,
            'birth_year':birth_years,
            'lat':da_population.lat.data,
            'lon':da_population.lon.data,    
        }] = xr.where(ds_emergence_union['emergence_mean']>0,1,0).sum(dim='hazard')
                        
        # pickle mean emergence
        with open('./data/pickles/emergence_hazards.pkl', 'wb') as f:
            pk.dump(ds_emergence_union['emergence_mean'],f)  
            
        # pickle emergence union
        with open('./data/pickles/emergence_union.pkl', 'wb') as f:
            pk.dump(ds_emergence_union['emergence_union'],f)     
            
        da_emergence_mean = ds_emergence_union['emergence_mean']
        da_emergence_union = ds_emergence_union['emergence_union']
        
        # get subsets of these large objects/pickles for plotting locally
        da_emergence_mean_subset = da_emergence_mean.loc[{
            'GMT':[6,15,17,24],
            'birth_year':[1960,1980,2000,2020],
        }]  
        da_emergence_union_subset = da_emergence_union.loc[{
            'GMT':[6,15,17,24],
            'birth_year':[1960,1980,2000,2020],
        }]  
        
        # pickle mean emergence
        with open('./data/pickles/emergence_hazards_mean_subset.pkl', 'wb') as f:
            pk.dump(da_emergence_mean_subset,f)         
            
        # pickle emergence union
        with open('./data/pickles/emergence_union_subset.pkl', 'wb') as f:
            pk.dump(da_emergence_union_subset,f)                   
            
    else:
        
        # # load mean emergence
        # with open('./data/pickles/emergence_hazards.pkl', 'rb') as f:
        #     da_emergence_mean = pk.load(f)  
            
        # # load emergence union
        # with open('./data/pickles/emergence_union.pkl', 'rb') as f:
        #     da_emergence_union = pk.load(f)            
            
        with open('./data/pickles/emergence_hazards_subset_new.pkl', 'rb') as f:
            da_emergence_mean_subset = pk.load(f)  
            
        # load emergence union
        with open('./data/pickles/emergence_union_subset_new.pkl', 'rb') as f:
            da_emergence_union_subset = pk.load(f)             
                     
        
        
    return da_emergence_mean_subset,da_emergence_union_subset
# %%
