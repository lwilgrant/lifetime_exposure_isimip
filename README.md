# Lifetime exposure isimip
Python code for lifetime exposure analysis with ISIMIP simulations, translated from the MATLAB code of [Thiery et al. (2021)](https://github.com/VUB-HYDR/2021_Thiery_etal_Science) and extended toward assessing gridscale, unprecedented exposure.


## Environment
The python modules used in this repository can be installed using [exposure_env.yml](./exposure_env.yml). This may take up to an hour to compile in the Anaconda prompt.

```
conda env create -f exposure_env.yml

```

## Sample data
Sample data for our analysis for heatwaves is available [here](https://vub-my.sharepoint.com/:f:/g/personal/luke_grant_vub_be/Evp91Cvs6tFPumIUySMnkPcBvTh1_T6lKn3jhPYzOvfmDA?e=DsZTdQ). You need to copy the "data" folder and its contents to the same directory as the code for the repo to work.

## Running
Once your python environment is set up, running this analysis for heatwaves should take 3-6 hours. Simply choose the "heatwavedarea" flag and set all run options to full compute. This will produce a number of python pickle files for intermediate computations and final results. Final results are mostly present as xarray datasets/dataarrays. Note that some plotting functions will not work, as they require outputs of analyses for other extreme event categories for which sample data is not provided.

## License
This project is licensed under the MIT License. See also the 
[LICENSE](LICENSE) 
file.



## References
Thiery, W., Lange, S., Rogelj, J., Schleussner, C. F., Gudmundsson, L., Seneviratne, S. I., Andrijevic, M., Frieler, K., Emanuel, K., Geiger, T., Bresch, D. N., Zhao, F., Willner, S. N., Büchner, M., Volkholz, J., Bauer, N., Chang, J., Ciais, P., Dury, M., … Wada, Y. (2021). Intergenerational inequities in exposure to climate extremes. Science, 374(6564), 158–160. https://doi.org/10.1126/science.abi7339
