from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import re

import numpy as np
import xarray as xr

#: CDL land-use code representing open water (used when constructing masks).
OPEN_WATER_LANDUSE_CODE = 111

camel_pat = re.compile(r'([A-Z])')
under_pat = re.compile(r'_([a-z])')

def pause() -> None:
    programPause = input("Press the <ENTER> key to continue...")
     
def camel_to_underscore(name: str) -> str:
    return camel_pat.sub(lambda x: '_' + x.group(1).lower(), name)

def underscore_to_camel(name: str) -> str:
    return under_pat.sub(lambda x: x.group(1).upper(), name)

def underscore_to_kebab(name: str) -> str:
    return under_pat.sub(lambda x: '-' + x.group(1), name)

def extract_run_information_from_filename(nc_filename: str | Path) -> Tuple[str, str, str, str, str, str, str, str]:
    """Parse an SWB2 output filename that encodes run metadata.

    Expected pattern (double underscores between parts):
    ``scenario__weather_model__short_period__variable__time_period__spatial_coverage.nc``

    Example:
        ``ssp245__bcc_csm2-mr__2040-2059__runoff__2040-01-01_to_2059-12-31__688_by_620.nc``

    Args:
        nc_filename: Path or filename of the SWB2 netCDF output.

    Returns:
        Tuple of:
        ``(scenario_name, weather_data_name, short_time_period, swb_variable_name,
        time_period, start_date, end_date, spatial_coverage)``.
    """
    name = Path(nc_filename).name
    (
        scenario_name,
        weather_data_name,
        short_time_period,
        swb_variable_name,
        time_period,
        spatial_coverage,
    ) = name.split("__")

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

def make_mask_ds(landuse_filename: str | Path, open_water_code: int | None = None) -> xr.Dataset:
    """Create a boolean mask dataset from a land-use GeoTIFF.

    The mask marks **land** as True and **open water / nodata** as False.

    Args:
        landuse_filename: Path to a land-use GeoTIFF (e.g., CDL).
        open_water_code: Integer code representing open water in the raster.
            If ``None`` (default), only NaN pixels are treated as False (nodata-only masking).

    Returns:
        An xarray Dataset with a boolean variable ``maskval`` (shape matching the raster).
    """
    mask_ds = xr.open_dataset(landuse_filename)
    # Use the first band; drop band dimension to keep 2-D
    mask_ds["band_data2"] = mask_ds["band_data"].sel(band=1).drop_vars("band")
    mask_ds.drop_dims("band")

    if open_water_code is None:
        # Only NaNs are considered non-land (False)
        mask_ds["maskval"] = xr.where(mask_ds.band_data2.isnull(), False, True)
    else:
        # Treat NaNs and the specific open-water class as non-land
        mask_ds["maskval"] = xr.where(
            np.logical_or(mask_ds.band_data2 == open_water_code, mask_ds.band_data2.isnull()),
            False,
            True,
        )

    return mask_ds
