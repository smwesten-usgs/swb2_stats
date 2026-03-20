# src/swb2_stats/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import xarray as xr

from .create_summary_dataset import create_summary_dataset
from .export_functions import (
    export_xarray_dataset_as_netcdf,
    export_xarray_dataset_as_series_of_tif_images,
)
from .utility_functions import (
    extract_run_information_from_filename,
    make_mask_ds,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="swb2_stats",
        description="Summarize SWB2 outputs and export results (netCDF / GeoTIFF) or compute zonal statistics.",
    )
    p.add_argument(
        "--landuse_tif_filename",
        help="Path to a land-use GeoTIFF used to build a mask (e.g., CDL). If omitted, no mask is applied.",
    )
    p.add_argument(
        "--open-water-code",
        type=int,
        default=None,
        help=(
            "Integer land-use class code representing open water in the raster. "
            "If omitted (default), only nodata/NaN pixels are considered non-land. "
            "If provided, both that code and NaNs are treated as non-land."
        ),
    )
    p.add_argument(
        "--swb_output_filename",
        required=True,
        help="Path to a SWB2 output netCDF file of daily values.",
    )
    p.add_argument(
        "--output_dir",
        default=".",
        help="Directory where exports (GeoTIFF/netCDF) are written (default: current directory).",
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

    # Build mask if a landuse raster is provided
    mask_ds = (
        make_mask_ds(args.landuse_tif_filename, open_water_code=args.open_water_code)
        if args.landuse_tif_filename
        else None
    )

    variable_operation = "mean" if swb_variable_name in {"tmin", "tmax", "soil_storage", "tmax_minus_tmin"} else "sum"
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
