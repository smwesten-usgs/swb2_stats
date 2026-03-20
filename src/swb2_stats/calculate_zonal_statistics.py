import xarray as xr
import rioxarray as rio
import xrspatial as xrs
import numpy as np
import pandas as pd
import datetime as dt

seasons = {12: 'winter',3: 'spring',6: 'summer',9: 'fall'}

def calculate_zonal_statistics(xarray_dataset,
                               mask_filename,
                               scenario_name,
                               time_period,
                               swb_variable_name, 
                               weather_data_name, 
                               zone_char_width,
                               summary_basetype,
                               variable_operation):
    """
    Iterate over the grids in a xarray dataarray. It is assumed that this dataarray has already been
    summarized by resampling to a monthly or annual timestep. Zonal statistics are calculated for each of the
    distinct zone numbers contained in the zone mask file.
    """

    # idea here is that if there is a time series of grids, the 'time' dimension should be present, and
    # the shape should be ('time', 'y', 'x'). if we are summarizing a 'mean annual sum' grid, no 'time'
    # dimension will be present.

    xarray_dataarray = xarray_dataset[swb_variable_name]
    mask_dataarray = xr.open_dataarray(mask_filename).astype('int')[0,:,:]

    dims = list(xarray_dataarray.dims)

    summary_type_plus_operation = f"{summary_basetype}_{variable_operation}"

    if ('time' in dims or 'month' in dims):
        for i in range(xarray_dataarray.shape[0]):
            df = xrs.zonal.stats(zones=mask_dataarray, values=xarray_dataarray[i,:,:])
            if 'time' in dims:
                t = xarray_dataarray['time'][i].values
                year=pd.to_datetime(t).year
                month=pd.to_datetime(t).month
            elif 'month' in dims:
                year = np.nan
                month = xarray_dataarray['month'][i].values

            df['season_name'] = None
            df['month'] = None
            
            match summary_type_plus_operation:
                case 'monthly_sum' | 'monthly_mean':
                    df['month'] = month
                    df['year'] = year
                    df['date'] = dt.datetime.strptime(f"{year}-{month}-15", "%Y-%m-%d")
                    df['water_year'] = df['year'].where(df['month'] < 10, df['year'] + 1)

                case 'mean_monthly_sum' | 'mean_monthly_mean':
                    # no 'year' or 'date' element, since this should summarize all values for a given month
                    df['month'] = month
                case 'seasonal_sum' | 'seasonal_mean':
                    df['month'] = month
                    df['year'] = year
                    df['date'] = dt.datetime.strptime(f"{year}-{month}-15", "%Y-%m-%d")
                    df['water_year'] = df['year'].where(df['month'] < 10, df['year'] + 1)
                    df['month'] = month
                    try:
                       df['season_name'] = seasons[month]
                    except:
                       pass   
                case 'mean_seasonal_sum' | 'mean_seasonal_mean':
                    df['month'] = month
                    try:
                       df['season_name'] = seasons[month]
                    except:
                       pass   
                case 'annual_sum' | 'annual_mean':
                    df['month'] = 6
                    df['year'] = year
                    df['date'] = dt.datetime.strptime(f"{year}-06-01", "%Y-%m-%d")
                    df['water_year'] = df['year'].where(df['month'] < 10, df['year'] + 1)
                case 'mean_annual_sum' | 'mean_annual_mean':
                    pass
                    # for mean annual calculations there is no meaningful timestamp, year, or month value
                case _:
                    print(f"calculate_zonal_statistics: unknown summary_type_plus_operation '{summary_type_plus_operation}'")
                    exit(1) 

            #df['zone'] = df['zone']

            if i == 0:
                zonal_stats = df.copy()
            else:
                zonal_stats = pd.concat([zonal_stats, df])

    else:
        zonal_stats = xrs.zonal.stats(zones=mask_dataarray, values=xarray_dataarray, nodata_values=0)
        zonal_stats['month'] = None
        zonal_stats['season_name'] = None

    zonal_stats = zonal_stats[zonal_stats['zone'] > 0 ]

    # convert zone labels to string, prepend '0' if desired
    #zonal_stats['zone'] = fix_zone_labels(zonal_stats['zone'], num_zone_chars)
    zonal_stats['zone'] = zonal_stats['zone'].apply(str)
    if zone_char_width is not None:
        zonal_stats['zone'] = zonal_stats['zone'].apply(lambda x: f"{x:0>{zone_char_width}}")

    zonal_stats['summary_basetype'] = summary_basetype
    zonal_stats['variable_operation'] = variable_operation
    zonal_stats['scenario_name'] = scenario_name
    zonal_stats['time_period'] = time_period
    zonal_stats['swb_variable_name'] = swb_variable_name
    zonal_stats['weather_data_name'] = weather_data_name

    return zonal_stats
