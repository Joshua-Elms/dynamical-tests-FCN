from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
from earth2mip.initial_conditions import hdf5, cds
from earth2mip import registry, inference_ensemble
import datetime
import os
from time import perf_counter


def _setup(data_source_path, device, ngpus, vocal, mode: str):
    """
    Set environment variables, load model, and load data source.

    Parameters:
        data_source_path (str): Path to the directory containing a data.json file and subdirectories with HDF5 files. Not used if mode is "simple".
        device (str): Device to run the model on.
        ngpus (int): Number of GPUs to use.
        vocal (bool): If True, print out progress and results.
        mode (str): "custom" or "simple".

    Returns:
        package (earth2mip.registry.ModelPackage): The model package.
        sfno_inference_model (earth2mip.networks.fcnv2_sm.FCNv2SmallModel): The model.
        data_source (earth2mip.initial_conditions.base.DataSource): The data source.
    """

    # Set number of GPUs to use to 1 (using zero currently, check fcnv2_sm_load parameter "device")
    if ngpus > 0:
        os.environ["WORLD_SIZE"] = str(ngpus)

    # Load model(s) from registry
    package = registry.get_model("fcnv2_sm")

    if vocal:
        print("Loading FCNv2 Small Model...")
        start = perf_counter()

    sfno_inference_model = fcnv2_sm_load(package, device=device)

    if vocal:
        stop = perf_counter()
        print(f"Model loaded. Duration: {stop - start:.2f} seconds")

    # Load data source
    if mode == "custom":
        data_source = hdf5.DataSource.from_path(
            root=data_source_path)

    elif mode == "simple":
        data_source = cds.DataSource(sfno_inference_model.in_channel_names)

    else:
        raise ValueError("Invalid mode. Must be 'custom' or 'simple'")

    return package, sfno_inference_model, data_source


def _run_inference(
    sfno_inference_model,
    n_iters,
    data_source,
    start_time,
    vocal
):
    """
    Run inference for n iterations.

    Parameters:
        sfno_inference_model (earth2mip.networks.fcnv2_sm.FCNv2SmallModel): The model.
        n_iters (int): Number of timesteps to run the model for.
        data_source (earth2mip.initial_conditions.base.DataSource): The data source.
        start_time (datetime.datetime): The starting time for the model run.
        vocal (bool): If True, print out progress and results.

    Returns:
        xarray.Dataset: The output dataset.
    """
    if vocal:
        start = perf_counter()
        print(f"Starting {n_iters} {'iterations' if n_iters > 1 else 'iteration'} of inference...")

    # Run inference for n iterations
    ds = inference_ensemble.run_basic_inference(
        sfno_inference_model,
        n=n_iters,
        data_source=data_source,
        time=start_time,
    )

    if vocal:
        stop = perf_counter()
        print(f"Inference finished. Duration: {stop - start:.2f} seconds")

    return ds


def _save_output(ds, output_path, vocal):
    """
    Save the output dataset to a netCDF file.

    Parameters:
        ds (xarray.Dataset): The output dataset.
        output_path (str): Path to save the output netCDF file.
        vocal (bool): If True, print out progress and results.

    Returns: None
    """
    # convert time dimension into unix time so that it can be written to netcdf
    ds["time"] = ds["time"].astype("datetime64[s]").astype("int")

    if vocal:
        print("Output: \n\n", ds)

    # save output
    ds.to_netcdf(output_path)
    if vocal:
        print(f"Output saved to: {output_path}")


def run_custom_inference(
        data_source_path: str,
        output_path: str,
        n_iters: int,
        start_time: datetime.datetime,
        device: str = "cpu",
        ngpus: int = 0,
        vocal: bool = False
):
    """
    Run inference using a custom data source. If you want to use ERA5 data at a particular timestamp instead, use run_simple_inference.

    The custom data source must be formatted as an HDF5 file with the following structure:
    - data
        - time
        - channel
        - lat
        - lon

    For fcnv2_sm, the dimensions and their sizes (in order) must be (time: <any>, channel: 73, lon: 1440, lat: 721).

    The ordered list of 73 channels is as follows:
    [
        u10m, v10m, u100m, v100m, t2m, sp, msl, tcwv, 
        u50, u100, u150, u200, u250, u300, u400, u500, u600, u700, u850, u925, u1000, 
        v50, v100, v150, v200, v250, v300, v400, v500, v600, v700, v850, v925, v1000, 
        z50, z100, z150, z200, z250, z300, z400, z500, z600, z700, z850, z925, z1000, 
        t50, t100, t150, t200, t250, t300, t400, t500, t600, t700, t850, t925, t1000, 
        r50, r100, r150, r200, r250, r300, r400, r500, r600, r700, r850, r925, r1000
    ]

    Values must be standardized according to the formula: (value - mean) / std
    The mean and std values for each channel can be found in the model registry.

    --- HDF5 Data Sources (from initial_conditions.hdf5) ---

    Works with a directory structure like this::

        data.json
        subdirA/2018.h5
        subdirB/2017.h5
        subdirB/2016.h5

    data.json should have fields

        h5_path - the variable name of the data within the hdf5 file 
        dims - list of dimensions in the data
        dhours - timestep in hours (default 6 hours)
        attrs - dictionary of attributes to add to the xarray dataset, leave empty if no attributes
        coords.channel - list of channels
        coords.lat - list of lats
        coords.lon - list of lons

    An example data.json file would look like this:
        {
            "h5_path": "data",
            "dhours": 6,
            "attrs": {"source": "idealized experiment"},
            "dims": ["time", "channel", "lat", "lon"],
            "coords": {
                "channel": ["u10m", "v10m", ... , "r1000"],
                "lat": [90, 89.75, ... , -90],
                "lon": [0, 0.25, ... , 359.75]
            }
        }

    A similar file is provided in this directory. 

    Parameters:
        data_source_path (str): Path to the directory containing a data.json file and subdirectories with HDF5 files. See above for more information.
        output_path (str): Path to save the output netCDF file. If none, just return the xarray dataset.
        n_iters (int): Number of timesteps to run the model for. with "dhours" set to 6, n_iters=1 is 6 hours.
        start_time (datetime.datetime): The starting time for the model run.
        device (str): Device to run the model on. Default is "cpu", but "cuda" is faster if GPU is available.
        ngpus (int): Number of GPUs to use. Default is 0. Only relevant if device is set to "cuda".
        vocal (bool): If True, print out progress and results. Default is False.

    Returns:
        xarray.Dataset: The output dataset.

    Output:
        Saves the output netCDF file to the specified path.
    """

    package, sfno_inference_model, data_source = _setup(
        data_source_path, device, ngpus, vocal, mode="custom")

    ds = _run_inference(
        sfno_inference_model,
        n_iters,
        data_source,
        start_time,
        vocal
    )

    if output_path:
        _save_output(ds, output_path, vocal)

    return ds


def run_simple_inference(
    output_path: str,
    n_iters: int,
    start_time: datetime.datetime,
    device: str = "cpu",
    ngpus: int = 0,
    vocal: bool = False
):
    """
    Run inference using ERA5 data at a particular timestamp. If you want to use a custom data source instead, use run_custom_inference.

    Parameters:
        output_path (str): Path to save the output netCDF file.
        n_iters (int): Number of timesteps to run the model for. with "dhours" set to 6, n_iters=1 is 6 hours.
        start_time (datetime.datetime): The starting time for the model run. Will be used to get the initial conditions from ERA5 via the ECWMF CDS API.
        device (str): Device to run the model on. Default is "cpu", but "cuda" is faster if GPU is available.
        ngpus (int): Number of GPUs to use. Default is 0. Only relevant if device is set to "cuda".
        vocal (bool): If True, print out progress and results. Default is False.

    Returns:
        xarray.Dataset: The output dataset.

    Output:
        Saves the output netCDF file to the specified path.
    """

    package, sfno_inference_model, data_source = _setup(
        None, device, ngpus, vocal, mode="simple")

    ds = _run_inference(
        sfno_inference_model,
        n_iters,
        data_source,
        start_time,
        vocal
    )

    if output_path:
        _save_output(ds, output_path, vocal)

    return ds
