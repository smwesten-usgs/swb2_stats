import xarray as xr
import rioxarray as rio
#import xrspatial as xrs
import numpy as np
import pandas as pd
import datetime as dt
import sys
from pyproj import CRS 
from pathlib import Path
import traceback
from utility_functions import (underscore_to_kebab,
                              underscore_to_camel)
#
# Information from Ryan Noe:
#
# This folder contains CMIP6 derived data products in .tif format. Each subfolder at the root level is a separate model.
# Parsing the filename requires some care because there are optional values and sometimes multiple units. 
# Each raster is named according the following convention:

# {SCENARIO}_{MODEL}_{OUTPUTTYPE}_{TIMEFRAME}-{SUBTIMEFRAME}_{VARIABLE-UNIT-DESCRIPTION}_{THRESHOLD-UNIT}.tif

# In the simplest case:
# historical_1983-1996_ensemble_modelVal_yearly_precip-inches.tif

# In an advanced case:
# RCP4.5_2043-2056_ensemble_numDif_slice-0501-0930_precip-days-max-consecutive-below_0.01-inch.tif

# # Ryan uses specific encoding for 'nodata' pixels:
#    with rasterio.open(template_tif_file) as src:
#         meta = src.meta.copy()

#     nodata_value = -3.4028234663852886e+38 
#     data = np.where(np.isnan(data), nodata_value, data)
    
# meta.update(dtype=rasterio.float32, count=1, nodata=nodata_value)

#     output_file = os.path.join(output_dir, f"{scenario}_ensemble_{outputType}_{timeFrame}_{variable}.tif")
#     with rasterio.open(output_file, 'w', **meta) as dst:
#         dst.write(data.astype(rasterio.float32), 1)




num_seasons = {12: 'winter', 3: 'spring', 6: 'summer', 9: 'fall'}
seasons = {'12': 'winter','03': 'spring','06': 'summer','09': 'fall'}
mn_seasons = {'12': 'seasonal-DJF', '03': 'seasonal-MAM', '06': 'seasonal-JJA', '09': 'seasonal-SON'}
month_name = {'01': 'january', '02': 'february', '03': 'march', '04': 'april',
              '05': 'may', '06': 'june', '07': 'july', '08': 'august',
              '09': 'september', '10': 'october', '11': 'november', '12': 'december'}

NODATA_VALUE = -3.4028234663852886e+38 

def export_xarray_dataset_as_netcdf(ds,
                                    output_grid_name):
  # Make a shallow copy so we don't mutate upstream references
    ds = ds.copy()
    # Remove stale dataset-level unlimited dims (e.g., {'time'})
    # This is where the warning originates.
    ds.encoding.pop('unlimited_dims', None)
    ds.to_netcdf(output_grid_name)

def write_tif(da, output_image_dir, file_prefix, from_epsg, to_epsg=None):
  """Write the contents of an xarray dataarray to a TIF file.

  Args:
      da (xarray dataarray): gridded data to be written to TIF
      output_image_dir (pathlib Path): location on disk where TIF is to be written
      file_prefix (str): prefix to be used in naming the output TIF
      to_epsg (integer, optional): EPSG code to be used in reprojecting the output grid. Defaults to None.
  """

  # Tell rioxarray which axes are spatial (the projected ones)
  da = da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)

  # Write the true source CRS
  da.rio.write_crs(CRS.from_epsg(5070), inplace=True)
 
  # Ensure spatial dims are set correctly
#  da = da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)
  # Now write the CRS
#  da.rio.write_crs(CRS.from_epsg(from_epsg), inplace=True)

  if to_epsg:
    try:
      ( da.rio.set_nodata(NODATA_VALUE)
          .rio.reproject(f"EPSG:{to_epsg}")
          .rio.to_raster(Path(output_image_dir) / f"{file_prefix}.tif" ,
                        driver="GTiff",
                        compress="LZW")
      )
    except:
       traceback.print_exc()  
  else:
    try:
      ( da.rio.set_nodata(NODATA_VALUE)
          .rio.to_raster(Path(output_image_dir) / f"{file_prefix}_EPSG-5070.tif" ,
                        driver="GTiff",
                        compress="LZW")
      )
    except:
      traceback.print_exc()  


def export_xarray_dataset_as_series_of_tif_images(ds,
                                            summary_basetype,
                                            variable_operation,
                                            scenario_name,
                                            weather_data_name,
                                            swb_variable_name,
                                            time_period,
                                            output_image_dir,
                                            from_epsg=5070,
                                            to_epsg=4326,
                                            mask_ds=None):

    
  # if mask_ds is not None:
  #   ds = ds.where(mask_ds.maskval)

  # ds = xr.where(ds)    

  if mask_ds is not None:
    ds_masked = ds.where(mask_ds.maskval)
    ds_masked[swb_variable_name] = xr.where(ds_masked[swb_variable_name].isnull(), 
                                            NODATA_VALUE, ds_masked[swb_variable_name])
  else:
    ds_masked = ds
    
  # convert the DataArray into a DataSet  
  try:
    da = ds[f"{swb_variable_name}"]
    da_masked = ds_masked[f"{swb_variable_name}"]
  except:
    traceback.print_exc()
    sys.exit(f"There were problems creating an xarray DataArray from a DataSet.")

  SCENARIO = f"{scenario_name}_{time_period}"
  MODEL = f"{weather_data_name}"
  OUTPUTTYPE = "modelVal"
  UNITS = da.units

  modelname = underscore_to_kebab(MODEL).upper()
  # ugly hack, need to ensure that reference ET is named properly for Ryan  
  variable_name = underscore_to_camel(swb_variable_name).replace("reference_ET0", "referenceEt0")
  units_txt = underscore_to_kebab(UNITS).replace("degrees-fahrenheit","degF")

  match summary_basetype:

    case "seasonal":

        for i in range(len(da.time.values)):
            seasonal = da.isel(time=i)
            seasonal_masked = da_masked.isel(time=i)
            year = int(seasonal.time.dt.year.values)
            month = int(seasonal.time.dt.month.values)
            month_val = f"{month:02d}"
            if month < 10:
                water_year = year
            else:
                water_year = year + 1

            # try:
            #   da = seasonal[f"{swb_variable_name}"]
            #   da_masked = seasonal_masked[f"{swb_variable_name}"]
            # except:
            #   traceback.print_exc()
            #   sys.exit(f"'month_val' is {month_val}. There were problems creating data array.")

            try:
              TIMEFRAME = f"{mn_seasons[month_val]}-wy{water_year}"
              file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
              write_tif(da=seasonal, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg)
              write_tif(da=seasonal_masked, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg)
            except:
               traceback.print_exc()
               sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems")

        return

    case "mean_seasonal":

        for i in da.month.values:
            seasonal = da.sel(month=i)
            seasonal_masked = da_masked.sel(month=i)
            month_val = f"{seasonal.month:02d}"

            try:
              TIMEFRAME = mn_seasons[month_val]
              file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
              write_tif(da=seasonal, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg)
              write_tif(da=seasonal_masked, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg)
              #  seasonal.rio.to_raster(output_image_dir / f"{file_prefix}_EPSG-5070.tif" ,
              #                         driver="GTiff", compress="LZW")
              #  seasonal_masked.rio.reproject(f"EPSG:{to_epsg}").rio.to_raster(output_image_dir / f"{file_prefix}.tif" ,
              #                         driver="GTiff", compress="LZW")
            except:
               traceback.print_exc()
               sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems")
        return

    case "mean_growing-season":

        try:
          TIMEFRAME = f"slice-0501-0930"
          file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
          write_tif(da=da, 
                    output_image_dir=output_image_dir,
                    file_prefix=file_prefix,
                    from_epsg=from_epsg)
          write_tif(da=da_masked, 
                    output_image_dir=output_image_dir,
                    file_prefix=file_prefix,
                    from_epsg=from_epsg,
                    to_epsg=to_epsg)
        except:
           traceback.print_exc()
           sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
        return

    case "mean_monthly":

        for i in range(len(da.month.values)):
            monthly = da.isel(month=i)
            monthly_masked = da_masked.isel(month=i)
            month = int(monthly.month.values)
            month_val = f"{month:02d}"
            #TIMEFRAME = f"monthly-{month_name[month_val]}"
            try:
              TIMEFRAME = f"monthly-{month}"
              file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
              write_tif(da=monthly, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg)
              write_tif(da=monthly_masked, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg)
            except:
              traceback.print_exc()
              sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
        return

    case "monthly":
        for i in range(len(da.time.values)):
            monthly = da.isel(time=i)
            monthly_masked = da_masked.isel(time=i)
            month_val = str(monthly.time.values).split('-')[1]
            year_val = str(monthly.time.values).split('-')[0]
            try:
              TIMEFRAME = f"monthly-{month_name[month_val]}-{year_val}"
              file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"

              write_tif(da=monthly, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg)
              write_tif(da=monthly_masked, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg)
            except:
              traceback.print_exc()
              sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
        return

    case "mean_annual":
      try:
        TIMEFRAME = "yearly"  
        file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"        
        write_tif(da=da, 
                  output_image_dir=output_image_dir,
                  file_prefix=file_prefix,
                  from_epsg=from_epsg)
        write_tif(da=da_masked, 
                  output_image_dir=output_image_dir,
                  file_prefix=file_prefix,
                  from_epsg=from_epsg,
                  to_epsg=to_epsg)
      except:
        traceback.print_exc()
        sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
      return

    case "annual":

        for i in range(len(da.time.values)):
            yearly = da.isel(time=i)
            yearly_masked = da_masked.isel(time=i)
            year_val = str(yearly.time.values).split('-')[0]
            try:
              TIMEFRAME = f"yearly-{year_val}"
              file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
              write_tif(da=yearly, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg)
              write_tif(da=yearly_masked, 
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg)
            except:
              traceback.print_exc()
              sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
        return

    case _:
          print(f"export_xarray_dataset_as_series_of_tif_images: unknown summary_basetype '{summary_basetype}'")
          sys.exit(1)

def export_zonal_stats_dataframe_as_parquet(df,
                                            optional_output_suffix,
                                            summary_basetype,
                                            variable_operation,
                                            scenario_name,
                                            weather_data_name,
                                            swb_variable_name,
                                            time_period,
                                            data_summary_dir):

  match summary_basetype:

    case "seasonal":

      df['season_name'] = df.month
      df.replace({'season_name': num_seasons}, inplace=True)
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "mean_seasonal":
      df['season_name'] = df.month
      df.replace({'season_name': num_seasons}, inplace=True)
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "mean_growing-season":
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "growing-season":
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "mean_monthly":
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "monthly":
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "mean_annual":
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case "annual":
      df.to_parquet(path=data_summary_dir / 
        f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}__{weather_data_name}__{swb_variable_name}{optional_output_suffix}.parquet")
      return

    case _:
      print(f"export_zonal_stats_dataframe_as_parquet: unknown summary_basetype '{summary_basetype}'")
