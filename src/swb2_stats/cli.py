from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import xarray as xr
import numpy as np

from .create_summary_dataset import create_summary_dataset
from .export_functions import (
    export_xarray_dataset_as_netcdf,
    export_xarray_dataset_as_series_of_tif_images,
)

OPEN_WATER_LANDUSE_CODE = 111


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for swb2_stats CLI."""
    p = argparse.ArgumentParser(
        prog="swb2_stats",
        description="Summarize SWB2 outputs and export results (netCDF / GeoTIFF) or compute zonal statistics.",
    )
    p.add_argument(
        "--landuse_tif_filename",
        help="Path to a Cropland Data Layer (CDL) GeoTIFF used to mask open water / NaN.",
    )
    p.add_argument(
        "--swb_output_filename",
        required=True,
        help="Path to a SWB2 output netCDF file of daily values.",
    )
    p.add_argument(
        "--output_dir",
        default=".",
        help="Directory in which export files (TIF/netCDF) should be written (default: current dir).",
    )
    p.add_argument(
        "--summary_type",
        default="mean_annual",
        help="Summary type: one of ['seasonal','mean_seasonal','annual','mean_annual','mean_growing-season','monthly','mean_monthly'].",
    )
    p.add_argument(
        "--to_epsg",
        type=int,
        default=4326,
        help="Optional EPSG code to reproject GeoTIFF outputs (default: 4326). Source EPSG is assumed 5070.",
    )
    p.add_argument(
        "--make_netcdf",
        action="store_true",
        help="Also write the summarized dataset to netCDF.",
    )
    return p.parse_args()


def extract_run_information_from_filename(nc_filename: str | Path) -> Tuple[str, str, str, str, str, str, str, str]:
    """Extract scenario, model, period, variable, and spatial coverage from filename.

    Expected pattern (double underscores between parts):
    scenario__weather_model__short_period__variable__time_period__spatial_coverage.nc
    e.g.,
    ssp245__bcc_csm2-mr__2040-2059__runoff__2040-01-01_to_2059-12-31__688_by_620.nc
    """
    nc_filename = Path(nc_filename).name
    (
        scenario_name,
        weather_data_name,
        short_time_period,
        swb_variable_name,
        time_period,
        spatial_coverage,
    ) = nc_filename.split("__")

    start_date = time_period.split("_")[0]
    end_date = time_period.split("_")[2]
    spatial_coverage = spatial_coverage.split(".")[0]
    return (
        scenario_name,
        weather_data_name,
        short_time_period,
        swb_variable_name,
        time_period,
        start_date,
        end_date,
        spatial_coverage,
    )


def make_mask_ds(landuse_filename: str | Path) -> xr.Dataset:
    """Read a CDL GeoTIFF and produce a boolean mask Dataset (True=land, False=open water/Nan)."""
    mask_ds = xr.open_dataset(landuse_filename)
    mask_ds["band_data2"] = mask_ds["band_data"].sel(band=1).drop_vars("band")
    mask_ds.drop_dims("band")
    mask_ds["maskval"] = xr.where(
        np.logical_or(mask_ds.band_data2 == OPEN_WATER_LANDUSE_CODE, mask_ds.band_data2.isnull()), False, True
    )
    return mask_ds


def _choose_variable_operation(swb_variable_name: str) -> str:
    """Return 'mean' for variables that are not naturally summed; 'sum' otherwise."""
    if swb_variable_name in {"tmin", "tmax", "soil_storage", "tmax_minus_tmin"}:
        return "mean"
    return "sum"


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    nc_filename = args.swb_output_filename
    to_epsg: Optional[int] = args.to_epsg
    make_netcdf = bool(args.make_netcdf)

    (
        scenario_name,
        weather_data_name,
        short_time_period,
        swb_variable_name,
        time_period,
        start_date,
        end_date,
        spatial_coverage,
    ) = extract_run_information_from_filename(nc_filename=nc_filename)

    summary_basetype = args.summary_type
    mask_ds = make_mask_ds(args.landuse_tif_filename) if args.landuse_tif_filename else None

    variable_operation = _choose_variable_operation(swb_variable_name)
    summary_type = f"{summary_basetype}_{variable_operation}"

    ds = create_summary_dataset(
        netcdf_filename=nc_filename,
        scenario_name=scenario_name,
        swb_variable_name=swb_variable_name,
        weather_data_name=weather_data_name,
        short_time_period=short_time_period,
        summary_basetype=summary_basetype,
        variable_operation=variable_operation,
        mask_ds=mask_ds,
    )

    if make_netcdf:
        output_grid_name = (
            output_dir
            / f"{summary_type}__{scenario_name}__{weather_data_name}__{swb_variable_name}__{time_period}__{spatial_coverage}.nc"
        )
        export_xarray_dataset_as_netcdf(ds, output_grid_name)

    # Always write TIFs (preserving previous behavior); we can parameterize later.
    export_xarray_dataset_as_series_of_tif_images(
        ds,
        summary_basetype=summary_basetype,
        variable_operation=variable_operation,
        scenario_name=scenario_name,
        weather_data_name=weather_data_name,
        swb_variable_name=swb_variable_name,
        time_period=time_period,
        output_image_dir=output_dir,
        from_epsg=5070,
        to_epsg=to_epsg,
        mask_ds=mask_ds,
    )