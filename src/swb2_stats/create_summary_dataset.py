import xarray as xr
#import rioxarray as rio
#import xrspatial as xrs
import numpy as np
import pandas as pd
import datetime as dt
import gc

NODATA_VALUE = -3.4028234663852886e+38 
OPEN_WATER_LANDUSE_CODE = 111

def create_summary_dataset(netcdf_filename,
                           scenario_name,
                           swb_variable_name, 
                           weather_data_name, 
                           short_time_period,
                           summary_basetype='none',
                           variable_operation='none',
                           mask_ds=None):

    xarray_dataset = xr.open_dataset(netcdf_filename, 
                                     decode_coords=True, 
                                     decode_cf=True)

    if mask_ds is not None:
        ds_masked = xarray_dataset.where(mask_ds.maskval)
        ds_masked[swb_variable_name] = xr.where(ds_masked[swb_variable_name].isnull(), 
                                            NODATA_VALUE, ds_masked[swb_variable_name])
    else:
        ds_masked = xarray_dataset

    units = ds_masked[f"{swb_variable_name}"].units
# todo: resample is apparently just a fancy wrapper around 'groupby'; reformulating these to groupby 'water_year' 
# should be possible

    summary_type = f"{summary_basetype}_{variable_operation}"
    # Get the starting and ending year
    start_year = ds_masked['time'].min().dt.year.item()
    end_year = ds_masked['time'].max().dt.year.item()

    match summary_type:

        case 'mean_growing-season_sum':

            growing_season_sums = []
            # Loop through each year in the dataset
            for year in range(start_year, end_year+1):  # Adjust the range according to your dataset
                # Create a time slice for the growing season of the current year
                growing_season_slice = ds_masked.sel(time=slice(f'{year}-05-01', f'{year}-09-30'))
                # Sum the 'gross_precip' for the growing season
                seasonal_sum = growing_season_slice[f"{swb_variable_name}"].sum(dim='time')
                # Add a new dimension 'year' with the current year value
                seasonal_sum = seasonal_sum.expand_dims(year=[year])
                # Append the result to the list
                growing_season_sums.append(seasonal_sum)
            # Stack the list into a single DataArray
            growing_season_sums_da = xr.concat(growing_season_sums, dim='year').mean(dim='year')

#            mean_growing_season_sum = ds_masked.where(ds_masked['time.month'].isin(growing_season_months),drop=True)[f"{swb_variable_name}"].groupby('time.year').sum(dim='time').mean(dim='year')
            result_dataset = xr.Dataset(
                {
                    f"{swb_variable_name}": growing_season_sums_da,
                    'lat': ds_masked['lat'],
                    'lon': ds_masked['lon'],
                },
            )
            del growing_season_slice
            del growing_season_sums
            del growing_season_sums_da
            gc.collect()   


        case 'growing-season_sum':
            # Define the growing season months
            growing_season_months = [5, 6, 7, 8, 9]  # May to September
            # Filter the dataset for the growing season
            # Create a mask for the growing season
            growing_season_mask = ds_masked['time.month'].isin(growing_season_months)
            # Apply the mask to the dataset
            #growing_season_subset = ds_masked.where(growing_season_mask, drop=True)
            growing_season_sum = ds_masked.where(growing_season_mask, drop=True)[f"{swb_variable_name}"].groupby('time.year').sum(dim='time')
            # Create a new Dataset to hold the mean precipitation and the lat/lon coordinates
            result_dataset = xr.Dataset(
                {
                    f"{swb_variable_name}": growing_season_sum,
                    'lat': ds_masked['lat'],
                    'lon': ds_masked['lon'],
                },
            )
            del growing_season_mask
            #del growing_season_subset
            del growing_season_sum
            gc.collect()


        case 'mean_growing-season_mean':
            # Define the growing season months
            growing_season_months = [5, 6, 7, 8, 9]  # May to September
            # Filter the dataset for the growing season
            # Create a mask for the growing season
            growing_season_mask = ds_masked['time.month'].isin(growing_season_months)
            # Apply the mask to the dataset
            growing_season_subset = ds_masked.where(growing_season_mask, drop=True)
            mean_growing_season_mean = growing_season_subset[f"{swb_variable_name}"].groupby('time.year').mean(dim='time').mean(dim='year')
            # Now calculate the mean over all growing seasons
            #mean_growing_season_mean = growing_season_sum.mean(dim='year')
            # Create a new Dataset to hold the mean precipitation and the lat/lon coordinates
            result_dataset = xr.Dataset(
                {
                    f"{swb_variable_name}": mean_growing_season_mean,
                    'lat': ds_masked['lat'],
                    'lon': ds_masked['lon'],
                },
            )
            del growing_season_mask
            del growing_season_subset
            del mean_growing_season_mean
            gc.collect()            


        case 'mean_seasonal_sum':
            # return 4 grids of summed daily gridded values
            # returns stats for DJF, MAM, JJA, SON
            result_dataset = ds_masked.resample(time="QS-DEC").sum(dim="time", skipna=True).groupby("time.month").mean(dim="time", skipna=True)

        case 'seasonal_sum':
            # return 'n/4' grids of summed daily gridded values with 'n/4' equal to the number of quarters in the input dataset
            # returns stats for DJF, MAM, JJA, SON
            result_dataset = ds_masked.resample(time="QS-DEC").sum(dim="time", skipna=True)

        case 'seasonal_mean':
            # return 'n/4' grids of averaged daily gridded values with 'n/4' equal to the number of quarters in the input dataset
            result_dataset = ds_masked.resample(time="QS-DEC").mean(dim="time", skipna=True)

        case 'mean_seasonal_mean':
            # return 4 grids of averaged daily gridded values
            result_dataset = ds_masked.resample(time="QS-DEC").mean(dim="time", skipna=True).groupby("time.month").mean(dim="time", skipna=True)

        case 'monthly_sum':     
            # return 'n' grids of summed daily gridded values with 'n' equal to the number of months in the input dataset    
            result_dataset = ds_masked.resample(time="ME").sum(dim="time", skipna=True)

        case 'monthly_mean':
            # return 'n' grids of averaged daily gridded values with 'n' equal to the number of months in the input dataset    
            result_dataset = ds_masked.resample(time="ME").mean(dim="time", skipna=True)

        case 'mean_monthly_sum':
            # return 12 grids of summed daily gridded values; each grid represents the mean of the 
            #   sum of all January values, February values, etc.    
            result_dataset = ds_masked.resample(time="ME").sum(dim="time", skipna=True).groupby("time.month").mean(dim="time", skipna=True)

        case 'mean_monthly_mean':
            # return 12 grids of averaged daily gridded values; each grid represents the mean of the 
            #   mean of all January values, February values, etc.    
            result_dataset = ds_masked.resample(time="ME").mean(dim="time", skipna=True).groupby("time.month").mean(dim="time", skipna=True)

        case 'annual_sum':
            # return 'n' grids of summed daily gridded values, with 'n' equal to the number of years in the input dataset
            #result_dataset = ds_masked.resample(time="A").sum(dim="time")
            result_dataset = ds_masked.resample(time="YE").sum(dim='time', skipna=True)

        case 'annual_mean':
            # return 'n' grids of averaged daily gridded values, with 'n' equal to the number of years in the input dataset
            #result_dataset = ds_masked.resample(time="A").mean(dim="time")
            result_dataset = ds_masked.resample(time="YE").mean(dim="time", skipna=True)

        case 'mean_annual_sum':
            # return a single grid representing the mean of each annual summed variable amount over all years
            # ==> .sum(dim='time', skipna=True) *should* result in a resampled grid that respects NaNs; however, this oes not appear to work at the moment
            result_dataset = ds_masked.resample(time="YE").sum(dim="time", skipna=True).mean(dim="time", skipna=True)

        case 'mean_annual_mean':
            # return a single grid representing the mean of each annual mean variable amount over all years
            result_dataset = ds_masked.resample(time="YE").mean(dim="time", skipna=True).mean(dim="time", skipna=True)

        case _:
              print(f"unknown calculation_type '{summary_basetype}'")
              exit(1)

    # after result_dataset is produced via resample/groupby:
    if 'month' in result_dataset.dims:
        # Replace broadcasted lat/lon (month,y,x) with 2-D originals (y,x)
        result_dataset = result_dataset.drop_vars(['lat', 'lon'], errors='ignore')
        result_dataset = result_dataset.assign(
            lat=ds_masked['lat'],
            lon=ds_masked['lon'],
        )

    result_dataset = result_dataset.assign_attrs(swb_variable_name=swb_variable_name, 
                                                 summary_basetype=summary_basetype,
                                                 variable_operation=variable_operation,
                                                 weather_data_name=weather_data_name,
                                                 scenario_name=scenario_name,
                                                 time_period=short_time_period,
                                                 units=units,
                                                 original_source_filename=str(netcdf_filename))

    ds_masked.close()
    # if crs is not None:
    #     result_dataset.rio.write_crs(crs)
    #     result_dataset['crs'] = crs

    return result_dataset