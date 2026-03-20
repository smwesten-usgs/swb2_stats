
"""Zonal statistics for summarized SWB2 outputs.

This module provides a single high-level function,
`calculate_zonal_statistics`, which computes per-zone statistics
(e.g., sum/mean/min/max) by intersecting a categorical **zone mask**
with a summarized SWB2 data array.

Typical workflow:
1. Produce a summarized Dataset (monthly/seasonal/annual/growing‑season)
   using `create_summary_dataset`.
2. Call `calculate_zonal_statistics` with the summarized variable and a
   zone mask file (integer zones).
3. Receive a tidy `pandas.DataFrame` of statistics per zone (and per time
   slice when applicable).

Notes
-----
- The zone mask file is read as an `xarray.DataArray` and cast to integer.
  Zones with value `<= 0` are removed from the output.
- If the summarized variable has a temporal dimension (e.g., `time` or `month`),
  statistics are computed for each slice and concatenated into a single
  DataFrame with additional columns describing the date context.
- Zone labels may be optionally zero-padded to a fixed width (e.g., `"0012"`).

"""
from __future__ import annotations

from typing import Optional
import xarray as xr
import rioxarray as rio
import xrspatial as xrs
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path 

seasons = {12: 'winter',3: 'spring',6: 'summer',9: 'fall'}


def calculate_zonal_statistics(
    xarray_dataset: xr.Dataset,
    mask_filename: str | Path,
    scenario_name: str,
    time_period: str,
    swb_variable_name: str,
    weather_data_name: str,
    zone_char_width: Optional[int],
    summary_basetype: str,
    variable_operation: str,
) -> pd.DataFrame:

    """Compute zonal statistics for a summarized SWB2 variable.

    Intersects a **zone mask** (categorical integers) with a summarized
    SWB2 variable from `xarray_dataset[swb_variable_name]` and returns
    a `pandas.DataFrame` containing per-zone statistics. When the input
    data have a temporal dimension (e.g., `time` or `month`), the function
    iterates each slice and appends metadata (year/month/season/water_year)
    as appropriate.

    Args:
        xarray_dataset: Dataset that contains the summarized variable
            (e.g., monthly/seasonal/annual output produced by
            `create_summary_dataset`). The variable is accessed via
            `xarray_dataset[swb_variable_name]`.
        mask_filename: Path to a raster mask file readable by `xarray`
            (e.g., GeoTIFF) whose values are **integer zone IDs**.
            The first band is read and cast to `int`.
        scenario_name: Scenario identifier (e.g., ``"ssp245"``) propagated
            to the output metadata columns.
        time_period: Human-readable time period string (e.g.,
            ``"2040-2059"`` or ``"2040-01-01_to_2059-12-31"``) propagated
            to the output metadata columns.
        swb_variable_name: Name of the summarized variable in `xarray_dataset`
            for which zonal statistics are computed (e.g., ``"runoff"``).
        weather_data_name: Weather driver/model name (e.g., ``"bcc_csm2-mr"``),
            added to output metadata columns.
        zone_char_width: Optional width used to **zero-pad** zone labels
            in the output (e.g., `4` → `"0005"`). Use `None` to leave
            zone labels unchanged.
        summary_basetype: Base summary classification of the input data,
            such as ``"monthly"``, ``"mean_monthly"``, ``"seasonal"``,
            ``"mean_seasonal"``, ``"annual"``, ``"mean_annual"``,
            ``"growing-season"``, or ``"mean_growing-season"``.
            This is recorded in the output.
        variable_operation: Aggregation operation applied to the summary
            (e.g., ``"sum"`` or ``"mean"``). This is recorded in the output.

    Returns:
        A `pandas.DataFrame` with one row per zone (and per time slice when
        applicable). Typical columns from `xrspatial.zonal.stats` include:
        ``zone``, ``count``, ``min``, ``max``, ``mean``, ``std``, ``sum``.
        Additional metadata columns are appended:
        - ``month`` (int or None),
        - ``year`` (int or NaN when not applicable),
        - ``date`` (mid-month timestamp for monthly/seasonal cases),
        - ``water_year`` (for monthly/seasonal conventions),
        - ``season_name`` (for seasonal conventions),
        - ``summary_basetype``,
        - ``variable_operation``,
        - ``scenario_name``,
        - ``time_period``,
        - ``swb_variable_name``,
        - ``weather_data_name``.

    Notes:
        - **Temporal handling:** If the summarized DataArray contains a
          temporal dimension (``"time"`` or ``"month"``), the function
          loops over each slice, computes zonal statistics, and concatenates
          results into a single DataFrame.
        - **Zone filtering:** Zones with ID ``<= 0`` are dropped.
        - **Zero-padding:** If `zone_char_width` is provided, zone labels
          are left-padded with zeros to the given width.
        - **Nodata:** If the summarized variable uses nodata (e.g., masked
          upstream), ensure values intended to be ignored are set to a nodata
          sentinel or NaN before calling this function. `xrspatial.zonal.stats`
          respects `nodata_values` when provided for non-temporal cases.

    Examples:
        Compute seasonal zonal statistics on a dataset summarized to quarters:

        >>> df = calculate_zonal_statistics(
        ...     xarray_dataset=seasonal_ds,
        ...     mask_filename="zones.tif",
        ...     scenario_name="ssp245",
        ...     time_period="2040-2059",
        ...     swb_variable_name="runoff",
        ...     weather_data_name="bcc_csm2-mr",
        ...     zone_char_width=4,
        ...     summary_basetype="seasonal",
        ...     variable_operation="sum",
        ... )
        >>> set(df.columns)  # doctest: +ELLIPSIS
        {..., 'zone', 'sum', 'mean', 'month', 'season_name', 'water_year', ...}
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
