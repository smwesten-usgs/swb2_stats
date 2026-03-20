"""Export utilities for summarized SWB2 outputs.

This module provides functions to:
- write summarized xarray Datasets to **netCDF**;
- write xarray DataArrays as **GeoTIFF** rasters (with optional reprojection);
- serialize **zonal statistics** results to **Parquet**.

Coordinate Reference Systems (CRS)
----------------------------------
The GeoTIFF writer (`write_tif`) uses `rioxarray` to:
1. mark the spatial dimensions (`x`, `y`);
2. write the **source CRS** (default EPSG:5070; NAD_1983_Contiguous_USA_Albers);
3. optionally **reproject** to a target CRS (e.g., EPSG:4326, geographic lon/lat).

NoData Handling
---------------
Many tools expect a numeric NoData sentinel rather than NaN. This module uses:

- ``NODATA_VALUE = -3.4028234663852886e+38``

Before raster export, NaNs are replaced with this sentinel (when masking is used
upstream), and the raster is written with the appropriate nodata metadata.

Notes
-----
These functions assume upstream summarized Datasets/DataArrays already contain
the intended variable and coordinate layout (2-D `lat`/`lon` where applicable).
"""

from __future__ import annotations

import xarray as xr
import rioxarray as rio  # noqa: F401 (imported for side-effects / typing context)
# import xrspatial as xrs
import numpy as np
import pandas as pd
import datetime as dt
import sys
from pyproj import CRS
from pathlib import Path
import traceback
from typing import Optional

from .utility_functions import (
    underscore_to_kebab,
    underscore_to_camel,
)

num_seasons = {12: "winter", 3: "spring", 6: "summer", 9: "fall"}
seasons = {"12": "winter", "03": "spring", "06": "summer", "09": "fall"}
mn_seasons = {"12": "seasonal-DJF", "03": "seasonal-MAM", "06": "seasonal-JJA", "09": "seasonal-SON"}
month_name = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december",
}

#: Sentinel value used for NoData pixels in rasters and masked outputs.
NODATA_VALUE = -3.4028234663852886e+38


def export_xarray_dataset_as_netcdf(
    ds: xr.Dataset,
    output_grid_name: str | Path,
) -> None:
    """Write an xarray Dataset to a netCDF file.

    This function performs a **shallow copy** to avoid mutating caller references,
    removes stale dataset-level `encoding['unlimited_dims']` (a common warning source),
    and writes the dataset to the provided netCDF path.

    Args:
        ds: The xarray Dataset to export.
        output_grid_name: Path to the output netCDF file.

    Returns:
        None. The netCDF file is written to ``output_grid_name``.
    """
    # Make a shallow copy so we don't mutate upstream references
    ds = ds.copy()
    # Remove stale dataset-level unlimited dims (e.g., {'time'})
    ds.encoding.pop("unlimited_dims", None)
    ds.to_netcdf(output_grid_name)


def write_tif(
    da: xr.DataArray,
    output_image_dir: str | Path,
    file_prefix: str,
    from_epsg: int,
    to_epsg: Optional[int] = None,
) -> None:
    """Write a DataArray as a GeoTIFF, with optional reprojection.

    The function uses `rioxarray` to set spatial dims, write the source CRS
    (EPSG:5070 by default), optionally reproject to `to_epsg`, and write a
    compressed LZW GeoTIFF. If `to_epsg` is not supplied, the raster is
    written in the source CRS with a suffix indicating EPSG:5070.

    Args:
        da: The xarray DataArray to export as a raster. Must have `x`/`y` dims.
        output_image_dir: Directory where the GeoTIFF will be written.
        file_prefix: Filename prefix (without extension). A `.tif` extension will be added.
        from_epsg: EPSG code representing the source CRS (e.g., 5070).
        to_epsg: Optional EPSG code for reprojection (e.g., 4326 for lon/lat).

    Returns:
        None. A GeoTIFF is written to ``output_image_dir / f"{file_prefix}.tif"`` (if reprojected)
        or ``output_image_dir / f"{file_prefix}_EPSG-5070.tif"`` (if not reprojected).

    Raises:
        Any exception raised by `rioxarray` or `rasterio` during IO/reprojection
        will be caught and printed via `traceback.print_exc()`. The function does not re-raise.
    """
    # Tell rioxarray which axes are spatial (the projected ones)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    # Write the true source CRS
    da.rio.write_crs(CRS.from_epsg(5070), inplace=True)

    if to_epsg:
        try:
            (
                da.rio.set_nodata(NODATA_VALUE)
                .rio.reproject(f"EPSG:{to_epsg}")
                .rio.to_raster(
                    Path(output_image_dir) / f"{file_prefix}.tif",
                    driver="GTiff",
                    compress="LZW",
                )
            )
        except Exception:
            traceback.print_exc()
    else:
        try:
            (
                da.rio.set_nodata(NODATA_VALUE)
                .rio.to_raster(
                    Path(output_image_dir) / f"{file_prefix}_EPSG-5070.tif",
                    driver="GTiff",
                    compress="LZW",
                )
            )
        except Exception:
            traceback.print_exc()


def export_xarray_dataset_as_series_of_tif_images(
    ds: xr.Dataset,
    summary_basetype: str,
    variable_operation: str,
    scenario_name: str,
    weather_data_name: str,
    swb_variable_name: str,
    time_period: str,
    output_image_dir: str | Path,
    from_epsg: int = 5070,
    to_epsg: int = 4326,
    mask_ds: Optional[xr.Dataset] = None,
) -> None:
    """Write a series of GeoTIFF images from a summarized Dataset.

    Given a summarized dataset (e.g., monthly/seasonal/annual), this function
    constructs filenames that encode scenario, model, timeframe, variable, and units
    (compatible with existing conventions), and writes a GeoTIFF for each slice.
    If a `mask_ds` is provided, NaNs are replaced with `NODATA_VALUE` before export.

    Args:
        ds: Summarized xarray Dataset containing the variable to export.
        summary_basetype: Base summary type (e.g., ``"seasonal"``, ``"mean_seasonal"``,
            ``"mean_growing-season"``, ``"mean_monthly"``, ``"monthly"``,
            ``"mean_annual"``, ``"annual"``).
        variable_operation: Aggregation operation applied (e.g., ``"sum"`` or ``"mean"``).
        scenario_name: Scenario identifier (e.g., ``"ssp245"``).
        weather_data_name: Weather driver/model name (e.g., ``"bcc_csm2-mr"``).
        swb_variable_name: Name of the summarized variable in `ds`.
        time_period: Time period string used in filenames (e.g., ``"2040-2059"`` or
            ``"2040-01-01_to_2059-12-31"``).
        output_image_dir: Directory to which all GeoTIFFs will be written.
        from_epsg: Source CRS EPSG code (defaults to 5070).
        to_epsg: Target CRS EPSG code for reprojection (defaults to 4326).
        mask_ds: Optional mask Dataset with a boolean variable (e.g., ``maskval``).
            If provided, NaNs in the variable are replaced by :data:`NODATA_VALUE`.

    Returns:
        None. GeoTIFF files are written to ``output_image_dir``.

    Notes:
        - The filename convention follows an established pattern combining scenario,
          model, output type, timeframe, variable name, and units. For example:
          ``{SCENARIO}_{MODEL}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units}``.
        - Seasonal outputs distinguish DJF/MAM/JJA/SON via month labels and water year.
        - Monthly outputs may include month names or numeric month indexes.
        - When `mask_ds` is provided, masked exports are written both in source CRS
          and optionally reprojected to the target CRS.
    """
    # if mask_ds is not None:
    #     ds = ds.where(mask_ds.maskval)
    if mask_ds is not None:
        ds_masked = ds.where(mask_ds.maskval)
        ds_masked[swb_variable_name] = xr.where(
            ds_masked[swb_variable_name].isnull(),
            NODATA_VALUE,
            ds_masked[swb_variable_name],
        )
    else:
        ds_masked = ds

    # convert the DataArray into a DataSet
    try:
        da = ds[f"{swb_variable_name}"]
        da_masked = ds_masked[f"{swb_variable_name}"]
    except Exception:
        traceback.print_exc()
        sys.exit("There were problems creating an xarray DataArray from a DataSet.")

    SCENARIO = f"{scenario_name}_{time_period}"
    MODEL = f"{weather_data_name}"
    OUTPUTTYPE = "modelVal"
    UNITS = da.units
    modelname = underscore_to_kebab(MODEL).upper()
    # ugly hack, need to ensure that reference ET is named properly for Ryan
    variable_name = underscore_to_camel(swb_variable_name).replace("reference_ET0", "referenceEt0")
    units_txt = underscore_to_kebab(UNITS).replace("degrees-fahrenheit", "degF")

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

                try:
                    TIMEFRAME = f"{mn_seasons[month_val]}-wy{water_year}"
                    file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
                    write_tif(
                        da=seasonal,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                    )
                    write_tif(
                        da=seasonal_masked,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg,
                    )
                except Exception:
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
                    write_tif(
                        da=seasonal,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                    )
                    write_tif(
                        da=seasonal_masked,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg,
                    )
                except Exception:
                    traceback.print_exc()
                    sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems")
            return

        case "mean_growing-season":
            try:
                TIMEFRAME = "slice-0501-0930"
                file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
                write_tif(
                    da=da,
                    output_image_dir=output_image_dir,
                    file_prefix=file_prefix,
                    from_epsg=from_epsg,
                )
                write_tif(
                    da=da_masked,
                    output_image_dir=output_image_dir,
                    file_prefix=file_prefix,
                    from_epsg=from_epsg,
                    to_epsg=to_epsg,
                )
            except Exception:
                traceback.print_exc()
                sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
            return

        case "mean_monthly":
            for i in range(len(da.month.values)):
                monthly = da.isel(month=i)
                monthly_masked = da_masked.isel(month=i)
                month = int(monthly.month.values)
                # NOTE: using numeric month index rather than name here
                try:
                    TIMEFRAME = f"monthly-{month}"
                    file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
                    write_tif(
                        da=monthly,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                    )
                    write_tif(
                        da=monthly_masked,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg,
                    )
                except Exception:
                    traceback.print_exc()
                    sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
            return

        case "monthly":
            for i in range(len(da.time.values)):
                monthly = da.isel(time=i)
                monthly_masked = da_masked.isel(time=i)
                month_val = str(monthly.time.values).split("-")[1]
                year_val = str(monthly.time.values).split("-")[0]
                try:
                    TIMEFRAME = f"monthly-{month_name[month_val]}-{year_val}"
                    file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
                    write_tif(
                        da=monthly,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                    )
                    write_tif(
                        da=monthly_masked,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg,
                    )
                except Exception:
                    traceback.print_exc()
                    sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
            return

        case "mean_annual":
            try:
                TIMEFRAME = "yearly"
                file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
                write_tif(
                    da=da,
                    output_image_dir=output_image_dir,
                    file_prefix=file_prefix,
                    from_epsg=from_epsg,
                )
                write_tif(
                    da=da_masked,
                    output_image_dir=output_image_dir,
                    file_prefix=file_prefix,
                    from_epsg=from_epsg,
                    to_epsg=to_epsg,
                )
            except Exception:
                traceback.print_exc()
                sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
            return

        case "annual":
            for i in range(len(da.time.values)):
                yearly = da.isel(time=i)
                yearly_masked = da_masked.isel(time=i)
                year_val = str(yearly.time.values).split("-")[0]
                try:
                    TIMEFRAME = f"yearly-{year_val}"
                    file_prefix = f"{SCENARIO}_{modelname}_{OUTPUTTYPE}_{TIMEFRAME}_{variable_name}-{units_txt}"
                    write_tif(
                        da=yearly,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                    )
                    write_tif(
                        da=yearly_masked,
                        output_image_dir=output_image_dir,
                        file_prefix=file_prefix,
                        from_epsg=from_epsg,
                        to_epsg=to_epsg,
                    )
                except Exception:
                    traceback.print_exc()
                    sys.exit(f"'TIMEFRAME' is {TIMEFRAME}. File prefix is {file_prefix} There were problems writing the TIF files.")
            return

        case _:
            print(
                f"export_xarray_dataset_as_series_of_tif_images: unknown summary_basetype '{summary_basetype}'"
            )
            sys.exit(1)


def export_zonal_stats_dataframe_as_parquet(
    df: pd.DataFrame,
    optional_output_suffix: str,
    summary_basetype: str,
    variable_operation: str,
    scenario_name: str,
    weather_data_name: str,
    swb_variable_name: str,
    time_period: str,
    data_summary_dir: str | Path,
) -> None:
    """Write a zonal statistics DataFrame to a Parquet file.

    The output filename encodes the time period, summary type, operation,
    scenario, model, and variable. Seasonal cases also add a `season_name`
    column based on the month index.

    Args:
        df: Zonal statistics DataFrame produced upstream.
        optional_output_suffix: Optional string appended to the filename (e.g., filters).
        summary_basetype: Base summary type (e.g., ``"seasonal"``, ``"mean_seasonal"``,
            ``"mean_growing-season"``, ``"growing-season"``, ``"mean_monthly"``,
            ``"monthly"``, ``"mean_annual"``, ``"annual"``).
        variable_operation: Aggregation operation applied (e.g., ``"sum"`` or ``"mean"``).
        scenario_name: Scenario identifier (e.g., ``"ssp245"``).
        weather_data_name: Weather driver/model name (e.g., ``"bcc_csm2-mr"``).
        swb_variable_name: Name of the variable summarized in `df`.
        time_period: Time period string used in filenames (e.g., ``"2040-2059"``).
        data_summary_dir: Directory where the Parquet file will be written.

    Returns:
        None. A Parquet file is written to ``data_summary_dir``.
    """
    match summary_basetype:
        case "seasonal":
            df["season_name"] = df.month
            df.replace({"season_name": num_seasons}, inplace=True)
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "mean_seasonal":
            df["season_name"] = df.month
            df.replace({"season_name": num_seasons}, inplace=True)
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "mean_growing-season":
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "growing-season":
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "mean_monthly":
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "monthly":
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "mean_annual":
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case "annual":
            df.to_parquet(
                path=Path(data_summary_dir)
                / f"{time_period}__{summary_basetype}_{variable_operation}__{scenario_name}_{weather_data_name}_{swb_variable_name}{optional_output_suffix}.parquet"
            )
            return

        case _:
            print(
                f"export_zonal_stats_dataframe_as_parquet: unknown summary_basetype '{summary_basetype}'"
            )