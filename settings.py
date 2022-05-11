# ----------------------------------------------------------------
# Settings
# These are global variables to be used throughout the whole project
# ----------------------------------------------------------------

import numpy as np


def init(): 

    # initialise age and associated time period of interest
    global ages, age_young, age_ref, year_ref, year_start, birth_years, year_end
    ages        = np.arange(60,-1,-1)
    age_young   = 0
    age_ref     = np.nanmax(ages)
    year_ref    = 2020
    year_start  = year_ref - age_ref
    birth_years = np.arange(year_start,year_ref+1)     
    year_end    = 2113                            # based on maximum life expectancy reported in UN WPP


    # initialise age groups
    # (https://www.carbonbrief.org/analysis-why-children-must-emit-eight-times-less-co2-than-their-grandparents)
    # (https://www.pewresearch.org/fact-tank/2019/01/17/where-millennials-end-and-generation-z-begins/)
    global agegroups
    agegroups = {'Boomers'   : (1950, 1965),
                'Gen X'     : (1965, 1981),
                'Millenials': (1981, 1997),
                'Gen Z'     : (1997, 2020)}


    # initialise reference period for computing GMT anomalies
    global year_start_GMT_ref, year_end_GMT_ref
    year_start_GMT_ref  = 1850
    year_end_GMT_ref    = 1900


    global extremes_legend_dict
    extremes_legend_dict = {'burntarea'            : 'Wildfires'         ,
                            'cropfailedarea'       : 'Crop failures'     , 
                            'driedarea'            : 'Droughts'          , 
                            'floodedarea'          : 'River floods'      ,
                            'heatwavedarea'        : 'Heatwaves'         ,
                            'tropicalcyclonedarea' : 'Tropical cyclones' ,
                            'all'                  : 'All'               }
    
    # initialise model names
    global model_names
    model_names = { 'burntarea'            : ['CARAIB', 'LPJ-GUESS', 'LPJmL', 'ORCHIDEE', 'VISIT'], 
                    'cropfailedarea'       : ['GEPIC' , 'LPJmL'    , 'PEPIC'                                                             ],
                    'driedarea'            : ['CLM45' , 'H08'      , 'LPJmL', 'JULES-W1', 'MPI-HM', 'ORCHIDEE', 'PCR-GLOBWB', 'WaterGAP2'],
                    'floodedarea'          : ['CLM45'],#testing , 'H08'      , 'LPJmL', 'JULES-W1', 'MPI-HM', 'ORCHIDEE', 'PCR-GLOBWB', 'WaterGAP2'],
                    'heatwavedarea'        : ['HWMId-humidex', 'HWMId99-humidex40', 'HWMId97p5-humidex40', 'HWMId99-tasmax35', 'HWMId97p5-tasmax35', 'HWMId99', 'HWMId97p5', 'humidex40d3', 'humidex40d5', 'humidex45d3', 'humidex45d5', 'CWMId99'],
                    'tropicalcyclonedarea' : ['KE-TG-meanfield']}


    # Set threshold maximum T difference between RCP and GMT trajectories
    # i.e. any run with T difference exceeding this threshold is excluded
    # year-to-year jumps in GMT larger than 0.1, so using a 0.1 maxdiff threshold erronously removes runs
    # used to be 0.5, but then rcp2.6 is used for high-warming levels
    # Anything between 0.1 and 0.2 removes RCP2.6 in NDC scenarios (see histograms of maxdiff_NDC)
    # take 0.2 to have more data in BE scenarios and hence smooth EMF curves in BE plot
    global RCP2GMT_maxdiff_threshold
    RCP2GMT_maxdiff_threshold = 0.2 # [K]


    # set kernel x-values
    global kernel_x
    kernel_x = np.arange(1,50.5,0.5)


# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
def set_extremes(flags):

    global extremes
    
    if not flags['extr'] == 'all': # processing for single extreme
        extremes = [flags['extr']]
    else: 
        extremes = ['burntarea', 'cropfailedarea', 'driedarea', 'floodedarea', 'heatwavedarea' , 'tropicalcyclonedarea']
