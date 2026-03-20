# src/swb2_stats/__init__.py
"""
swb2_stats: Zonal statistics and summary exports for SWB2 netCDF outputs.

This package provides tools to:
- summarize SWB2 outputs (monthly/seasonal/annual/growing-season),
- export summaries to netCDF and GeoTIFF,
- compute zonal statistics.

"""

from .create_summary_dataset import create_summary_dataset
from .calculate_zonal_statistics import calculate_zonal_statistics
from .export_functions import (
    export_xarray_dataset_as_netcdf,
    export_xarray_dataset_as_series_of_tif_images,
    export_zonal_stats_dataframe_as_parquet,
)
from .utility_functions import (
    camel_to_underscore,
    underscore_to_camel,
    underscore_to_kebab,
    pause,
)

__all__ = [
    "create_summary_dataset",
    "calculate_zonal_statistics",
    "export_xarray_dataset_as_netcdf",
    "export_xarray_dataset_as_series_of_tif_images",
    "export_zonal_stats_dataframe_as_parquet",
    "camel_to_underscore",
    "underscore_to_camel",
    "underscore_to_kebab",
    "pause",
]