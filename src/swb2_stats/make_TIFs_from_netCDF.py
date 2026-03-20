import xarray as xr
import rioxarray
import numpy as np
import traceback
import re
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from utility_functions import (camel_to_underscore,
                               underscore_to_camel,
                               underscore_to_kebab,
                               pause)
from create_summary_dataset import create_summary_dataset
from export_functions import (export_xarray_dataset_as_netcdf,
                              export_xarray_dataset_as_series_of_tif_images)

mn_seasons = {'12': 'seasonal-DJF', '03': 'seasonal-MAM', '06': 'seasonal-JJA', '09': 'seasonal-SON'}
month_name = {'01': 'january', '02': 'february', '03': 'march', '04': 'april',
              '05': 'may', '06': 'june', '07': 'july', '08': 'august',
              '09': 'september', '10': 'october', '11': 'november', '12': 'december'}

NODATA_VALUE = -3.4028234663852886e+38 
OPEN_WATER_LANDUSE_CODE = 111

def make_mask_ds(landuse_filename):
    """
    Read in a cropland data layer TIF and return a xarray dataset mask.

    Args:
        landuse_filename(str): name of a cropland data layer TIF file
    """
    mask_ds = xr.open_dataset(landuse_filename)
    mask_ds['band_data2'] = mask_ds['band_data'].sel(band=1).drop_vars('band')
    mask_ds.drop_dims('band')
    mask_ds['maskval'] = xr.where(np.logical_or(mask_ds.band_data2 == OPEN_WATER_LANDUSE_CODE,
                                                mask_ds.band_data2.isnull()),False, True)
 
def parse_args() -> argparse.Namespace:
    """
    Read command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Define the SWB grid from an AOI polygon or bbox (raw extents only)."
    )

    p.add_argument(
        "--landuse_tif_filename",
        help="Name of a cropland data layer (CDL) tif file."
    )
    p.add_argument(
        "--swb_output_filename",
        help="Name of a SWB output netCDF file."
    )
    p.add_argument(
        "--output_dir",
        help="Directory in which the TIF file or file should be written (default: current dir)."
    )
    p.add_argument(
        "--summary_type",
        help="Type of summary desired: one of ['seasonal','mean_seasonal','annual','mean_annual','mean_growing-season'] (default: 'mean_annual')."
    )
    p.add_argument(
        "--to_epsg",
        help="If desired, an EPSG code to reproject to before writing out the masked TIF (default: 5070)."
    )
    p.add_argument(
        '--make_netcdf',
        action='store_true',
        help="Write statistics to netCDF file as well as TIF."
    )

    if len(sys.argv)==1:
        p.print_help(sys.stderr)
        sys.exit(0)

    return p.parse_args()

def extract_run_information_from_filename(nc_filename):
    """
    This function will fail unless naming conventions are strictly followed. The swb output files are assumed
    to be named as follows:

    scenario_name__weather_data_name__short_time_period__swb_variable_name__time_period__spatial_coverage.nc [note double underscores]

    For example:
      ssp245__bcc_csm2-mr__2040-2059__runoff__2040-01-01_to_2059-12-31__688_by_620.nc

    The idea is to make the output of the run as self-describing as possible, so that we don't have to create lists of 
    files, lists of weather data drivers, etc. in order to have the script run on all output.
    """  
    (scenario_name, 
    weather_data_name, 
    short_time_period,
    swb_variable_name, 
    time_period, 
    spatial_coverage) = nc_filename.split('__')

    start_date = time_period.split('_')[0]
    end_date = time_period.split('_')[2]
    
    spatial_coverage = spatial_coverage.split('.')[0]
    return (scenario_name, weather_data_name, short_time_period, swb_variable_name, time_period,
            start_date, end_date, spatial_coverage)

def main() -> None:
    args = parse_args()
       # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to project output_dir
        output_dir = Path.cwd()

    if args.swb_output_filename:
       nc_filename = args.swb_output_filename
    else:
       print("You need to supply the name of a SWB output netCDF file.")
       sys.exit()

    if args.to_epsg:
        to_epsg = args.to_epsg
    else:
        to_epsg = 5070

    if args.make_netcdf:
        make_netcdf=True
    else:
        make_netcdf=False

    (scenario_name, 
     weather_data_name, 
     short_time_period, 
     swb_variable_name, 
     time_period,
     start_date,
     end_date,
     spatial_coverage) = extract_run_information_from_filename(nc_filename=nc_filename)

    if args.summary_type:
        summary_basetype = args.summary_type
    else:
        summary_basetype = "mean_annual"

    if args.landuse_tif_filename:
        # create a true/false grid depending on whether we have nan/open water (False) or other land use (True)
        mask_ds = xr.open_dataset(args.landuse_tif_filename)
        mask_ds['band_data2'] = mask_ds['band_data'].sel(band=1).drop_vars('band')
        mask_ds.drop_dims('band')
        mask_ds['maskval'] = xr.where(np.logical_or(mask_ds.band_data2 == OPEN_WATER_LANDUSE_CODE,
                                                    mask_ds.band_data2.isnull()),False, True)
    else:
        mask_ds = None

    # SCENARIO = f"{scenario_name}_{time_period}"
    # MODEL = f"{weather_data_name}"
    # OUTPUTTYPE = "modelVal"
    # UNITS = da.units

    # modelname = underscore_to_kebab(MODEL).upper()
    # # ugly hack, need to ensure that reference ET is named properly for Ryan  
    # variable_name = underscore_to_camel(swb_variable_name).replace("reference_ET0", "referenceEt0")
    # units_txt = underscore_to_kebab(UNITS).replace("degrees-fahrenheit","degF")

    variable_operation = 'sum'
    if (swb_variable_name=='tmin' or 
        swb_variable_name=='tmax' or
        swb_variable_name=='soil_storage' or
        swb_variable_name=='tmax_minus_tmin'):
          variable_operation = 'mean'

    summary_type = f"{summary_basetype}_{variable_operation}"
              
    ds = create_summary_dataset(netcdf_filename=nc_filename,
                                scenario_name=scenario_name,
                                swb_variable_name=swb_variable_name, 
                                weather_data_name=weather_data_name, 
                                short_time_period=short_time_period,
                                summary_basetype=summary_basetype,
                                variable_operation=variable_operation,
                                mask_ds=mask_ds)
    
    if make_netcdf:
        output_grid_name = ( output_dir / 
                            f"{summary_type}__{scenario_name}__{weather_data_name}__{swb_variable_name}__{time_period}__{spatial_coverage}.nc" )
        export_xarray_dataset_as_netcdf(ds,
                                        output_grid_name)

    export_xarray_dataset_as_series_of_tif_images(ds,
                                            summary_basetype=summary_basetype,
                                            variable_operation=variable_operation,
                                            scenario_name=scenario_name,
                                            weather_data_name=weather_data_name,
                                            swb_variable_name=swb_variable_name,
                                            time_period=time_period,
                                            output_image_dir=output_dir,
                                            from_epsg=5070,
                                            to_epsg=4326,
                                            mask_ds=mask_ds)
    return ds

if __name__ == "__main__":
    ds = main()
