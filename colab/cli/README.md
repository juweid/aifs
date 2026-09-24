

# AIFS on Google Colab

Run the AIFS forecast on a Colab GPU:

Run the commands below from the directory containing `requirements.txt` and `aifs_v2.py`.

1. Create a notebook with an L4 GPU:
    ```sh
    colab new -s aifs --gpu L4
    ```

    An L4 GPU is recommended. A100 and H100 GPUs are also supported.


2. Optionally mount Google Drive:
    ```sh
    colab drivemount -s aifs
    ```
3. Install dependencies using either option:

    **Option 1 — install interactively**
    ```sh
    colab upload -s aifs requirements.txt /content/requirements.txt
    colab console -s aifs
    uv pip install -r requirements.txt
    ```

    **Option 2 — install remotely**
    ```sh
    colab install -s aifs -r requirements.txt
    ```

    This command may time out even though the installation completes successfully. Before running the script, check that all requirements are installed.

4. Run the model:
    ```sh
    colab exec -s aifs -f aifs_v2.py
    ```
5. Delete the Colab runtime when finished to free up resources:
    ```sh
    colab stop -s aifs
    ```

The forecast is saved in the Colab runtime as `aifs_forecast/aifs_ecmwf_025.zarr`.
If Google Drive is mounted and `COPY_TO_DRIVE` is set to `True` in
`aifs_v2.py`, the forecast is copied automatically to
`/content/drive/MyDrive/aifs_forecast`. Otherwise, download it or move it to
persistent storage while the runtime is active.