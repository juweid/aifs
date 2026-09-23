# AIFSv2 Single Notebook

This notebook runs the AIFSv2 Single forecast model in Google Colab, it is based on the original Notebook: [AIFS Single 2.0](https://huggingface.co/ecmwf/aifs-single-2.0/blob/main/run_AIFS_v2.0.ipynb).

## What it does

- checks the runtime and GPU
- installs the required Python packages
- loads initial conditions from ECMWF Open Data
- prepares and transforms the forecast inputs
- runs the AIFS model for a short forecast lead time
- saves the output as a Zarr archive
- optionally copies the results to Google Drive
- plots temperature for a city or region

## Requirements

- Google Colab
- CUDA-capable GPU
- Python 3.12.x (recommended for the bundled FlashAttention binaries)

## Supported runtime

| Runtime Version | Python | Status | GPUs Tested |
| --- | --- | --- | --- |
| 2026.07 | 3.12.13 | ✅ | L4 |

## Run order

Open `aifs_v2.ipynb` and run the cells in order from top to bottom.

The notebook is organized into sections:

1. install dependencies
2. import libraries
3. validate runtime and GPU
4. download initial conditions
5. transform fields for the model
6. create the runner and run the forecast
7. save the output to Zarr
8. copy results to Drive if desired
9. plot forecasts

## Output

The model output is saved under:

- `aifs_forecast/aifs_ecmwf_025.zarr`

and can be copied to Google Drive from the notebook.

## Notes

- The notebook expects an Ampere-class or newer GPU for FlashAttention.
- For a clean run, keep the notebook cells in their original order.
