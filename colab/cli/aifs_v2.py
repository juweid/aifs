import platform
import torch
import flash_attn

import datetime
from collections import defaultdict
from pathlib import Path
import os
import copy
import shutil

import numpy as np

import earthkit.data as ekd
import earthkit.regrid as ekr
from earthkit.geo.grids.array import regrid

import xarray as xr

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

from ecmwf.opendata import Client as OpendataClient

COPY_TO_DRIVE = True # Set to True to copy the forecast output to Google Drive if available

# Helper functions

def get_open_data(param, source, date, levelist=[], **kwargs):
    fields = defaultdict(list)
    # Get the data for the current date and the previous date as the model is initialised with t-6h data
    for date in [date - datetime.timedelta(hours=6), date]:
        data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist, source = source, **kwargs)
        
        for f in data: # type: ignore
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721,1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)
            
    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields

def save_model_run(states, start_date, params, output_dir):
    resolution = 0.25
    data_var_arrays = {}
    forecast_times = []

    for state in states:
        fields = state["fields"]
        for param in params:
            values_native = fields[param]

            grid_values, _ = regrid(
                data=values_native,
                in_grid={"grid": "N320"},
                out_grid={"grid": [resolution, resolution]},
                backend="precomputed",
            )

            grid_values = np.flipud(grid_values)
            grid_values = np.roll(
                grid_values,
                -(grid_values.shape[1] // 2),
                axis=1,
            )

            data_var_arrays.setdefault(param, []).append(grid_values)

        forecast_times.append(start_date + state["step"])

    grid_lat = np.linspace(
        -90,
        90,
        int(round(180 / resolution)) + 1,
    )
    grid_lon = np.linspace(
        -180,
        180 - resolution,
        int(round(360 / resolution)),
    )

    data_vars = {}
    for param, data_arrays in data_var_arrays.items():
        grid_values_all = np.stack(data_arrays, axis=0)

        assert grid_values_all.shape == (
            len(forecast_times),
            len(grid_lat),
            len(grid_lon),
        )
        
        data_vars[param] = (["time", "lat", "lon"], grid_values_all)

    ds_clean = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(forecast_times, dtype="datetime64[ns]"),
            "lat": grid_lat,
            "lon": grid_lon,
        },
        attrs={
            "initialization_date": str(start_date),
            "grid_resolution": f"{resolution} degree",
        },
    )

    ds_clean = ds_clean.chunk(
        {
            "time": -1,
            "lat": 180,
            "lon": 180,
        }
    )

    
    out_path = Path(output_dir) / "aifs_ecmwf_025.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ds_clean.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
    )

    print(
        "Zarr save completed:",
        ds_clean.sizes,
    )
 

def main():
    
    # Check Runtime Environment

    print("Platform:", platform.platform())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This notebook requires a GPU.")

    # See https://github.com/huggingface/transformers/issues/28188
    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)

    print("GPU:", gpu_name)
    print(f"Compute capability: {major}.{minor}")

    if major < 8:
        raise RuntimeError(
            "FlashAttention requires Ampere GPUs or newer"
        )

    print("FlashAttention:", flash_attn.__version__)
    
    # Define the parameters and levels to retrieve from ECMWF Open Data
    PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor", "sd"]
    PARAM_SOIL =["vsw","sot"]
    PARAM_WAVE =["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
    PARAM_PL  = ["gh", "t", "u", "v", "q"]
    LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
    SOIL_LEVELS = [1,2]

    SOURCE = "ecmwf" # Other options are: "azure", "aws", "ecmwf" or "google" 

    # Retrieve the initial conditions 

    DATE = OpendataClient(SOURCE).latest()
    print("Initial date is", DATE)
    fields = {}

    ekd.settings.set("cache-policy", "user")

    fields.update(get_open_data(param=PARAM_SFC, source=SOURCE, date=DATE, levtype="sfc"))
    assert all(p in fields for p in PARAM_SFC), "Missing parameters: %s" % (set(PARAM_SFC) - set(fields.keys()))

    fields.update(get_open_data(param=PARAM_WAVE, source=SOURCE, date=DATE, stream="wave"))
    assert all(p in fields for p in PARAM_WAVE), "Missing parameters: %s" % (set(PARAM_WAVE) - set(fields.keys()))

    soil=get_open_data(param=PARAM_SOIL,source=SOURCE, date=DATE,levelist=SOIL_LEVELS)

    soil_names = [f"{p}_{lev}" for p in PARAM_SOIL for lev in SOIL_LEVELS]
    assert all(p in soil for p in soil_names), "Missing parameters: %s" % (set(soil_names) - set(soil.keys()))

    fields.update(get_open_data(param=PARAM_PL,source=SOURCE, date=DATE, levelist=LEVELS))

    PRESSURE_NAMES = [f"{p}_{lev}" for p in PARAM_PL for lev in LEVELS]
    assert all(p in fields for p in PRESSURE_NAMES), "Missing parameters: %s" % (set(PRESSURE_NAMES) - set(fields.keys()))

    if not os.path.exists("lsm.grib"):
        lsm_data = ekd.from_source("ecmwf-open-data", date=DATE, param="lsm", source=SOURCE)
        lsm_data.save("lsm.grib")

    # Transform data 

    mwd = fields.pop("mwd")
    mwd_rad = np.deg2rad(mwd)

    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2',
            'vsw_1': 'swvl1','vsw_2': 'swvl2'}
    for k,v in soil.items():
        fields[mapping[k]]=v
        
    fields.pop("q_10", None)  # Remove the 10hPa level for specific humidity, as it is not used in the model
    fields.pop("q_50", None);  # Remove the 50hPa level for specific humidity, as it is not used as a prognostic in the model

    lsm_field = ekd.from_source("file", 'lsm.grib')[0]
    lsm_values = np.roll(lsm_field.to_numpy(), -lsm_field.shape[1] // 2, axis=1)
    lsm_interpolated = ekr.interpolate(lsm_values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
    mask = np.equal(lsm_interpolated, 0)

    fields["sd"][:, mask] = np.nan
    fields["swvl1"][:, mask] = np.nan
    fields["swvl2"][:,mask] = np.nan

    # Transform GH to Z
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665
        
    # Run the model
    
    input_state = dict(date=DATE, fields=fields)
    checkpoint = {"huggingface":"ecmwf/aifs-single-2.0"}
    runner = SimpleRunner(checkpoint)
    
    LEAD_TIME = 360
    states = []

    for state in runner.run(input_state=input_state, lead_time=LEAD_TIME):
        state_snapshot = copy.deepcopy(state)
        states.append(state_snapshot)
        print_state(state_snapshot)
    
    # Save the model run to disk
    OUTPUT_DIR = Path("aifs_forecast")
    save_model_run(states, DATE, ["2t"], OUTPUT_DIR)

    # Save model run to Google Drive if available

    if os.path.exists("/content/drive") and COPY_TO_DRIVE:
        destination_dir = "/content/drive/MyDrive/aifs_forecast"

        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir)

        shutil.copytree(str(OUTPUT_DIR), destination_dir)
        
        
if __name__ == "__main__":
    main()