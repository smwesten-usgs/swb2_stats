# tests/test_make_mask_ds.py
import numpy as np
from swb2_stats.utility_functions import make_mask_ds
import xarray as xr
from rasterio.transform import from_origin

def write_single_band_tif(array, path, epsg=5070, pixel_size=1000.0, origin_x=0.0, origin_y=0.0):
    """
    Write a 2-D numpy array to a single-band GeoTIFF with dims (y, x),
    variable name 'band_data', a valid CRS, and a geotransform.
    """
    h, w = array.shape

    # Build x/y coords that match the transform (center-of-pixel coordinates)
    x = origin_x + (np.arange(w) + 0.5) * pixel_size
    y = origin_y - (np.arange(h) + 0.5) * pixel_size  # y decreases down rows

    da = xr.DataArray(array, dims=("y", "x"), coords={"x": x, "y": y}, name="band_data")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    da = da.rio.write_crs(f"EPSG:{epsg}", inplace=False)
    da = da.rio.write_transform(from_origin(origin_x, origin_y, pixel_size, pixel_size), inplace=False)
    da = da.expand_dims(band=[1])  # single band

    da.rio.to_raster(path)

def test_make_mask_ds_nodata_only(tmp_path):
    data = np.array([[1, 1, 1],
                     [1, np.nan, 1],
                     [2, 2, 2]], dtype=float)
    tiff_path = tmp_path / "nodata_only.tif"
    write_single_band_tif(data, tiff_path, epsg=5070, pixel_size=1000.0, origin_x=500000.0, origin_y=4500000.0)

    mask_ds = make_mask_ds(tiff_path, open_water_code=None)
    mask = mask_ds["maskval"].values
    assert not bool(mask[1, 1])
    assert bool(mask[0, 0])
    assert bool(mask[2, 2])

def test_make_mask_ds_with_open_water_code(tmp_path):
    data = np.array([[111, 1, 1],
                     [1,  np.nan, 1],
                     [2,  2,    2]], dtype=float)
    tiff_path = tmp_path / "with_open_water.tif"
    write_single_band_tif(data, tiff_path, epsg=5070, pixel_size=1000.0, origin_x=500000.0, origin_y=4500000.0)

    mask_ds = make_mask_ds(tiff_path, open_water_code=111)
    mask = mask_ds["maskval"].values
    assert not bool(mask[0, 0])
    assert not bool(mask[1, 1])
    assert bool(mask[2, 2])