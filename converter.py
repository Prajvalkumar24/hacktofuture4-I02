from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    """Return first matching variable name (case-insensitive)."""
    by_lower = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match:
            return match
    return None


def _flatten(da: xr.DataArray) -> np.ndarray:
    """Flatten any N-D variable into 1-D for CSV export."""
    return np.asarray(da.values).reshape(-1)


def convert_netcdf_to_csv(nc_path: Path, csv_path: Path) -> None:
    with xr.open_dataset(nc_path) as ds:
        lat_var = _find_var(ds, ["LATITUDE", "LAT"])
        lon_var = _find_var(ds, ["LONGITUDE", "LON"])
        temp_var = _find_var(ds, ["TEMP", "TEMP_ADJUSTED", "TEMPERATURE"])

        if not lat_var or not lon_var or not temp_var:
            raise ValueError(
                "Could not find required variables in NetCDF file. "
                "Need latitude, longitude, and temperature."
            )

        lat = _flatten(ds[lat_var])
        lon = _flatten(ds[lon_var])
        temp = _flatten(ds[temp_var])

        max_len = max(len(lat), len(lon), len(temp))

        def pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) == max_len:
                return arr
            padding = np.full(max_len - len(arr), np.nan)
            return np.concatenate([arr.astype(float, copy=False), padding])

        df = pd.DataFrame(
            {
                "Latitude": pad(lat),
                "Longitude": pad(lon),
                "Temperature": pad(temp),
            }
        )

        # Keep rows that have temperature values for analysis.
        df = df[df["Temperature"].notna()].reset_index(drop=True)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    nc_files = sorted(Path(".").glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError("No .nc file found in current folder.")

    input_file = nc_files[0]
    output_file = Path("argo_data.csv")
    convert_netcdf_to_csv(input_file, output_file)
    print(f"Converted: {input_file.name} -> {output_file.name}")
