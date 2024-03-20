import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import scipy


def read_to_df(fort_output_dir: Path, f_in_name: str, nlat: int) -> tuple[pd.DataFrame, int]:

    # load the data
    f_in = pd.read_csv(fort_output_dir / f_in_name, header=0, index_col=None)

    # make sure given nlat is consistent with the input data
    assert f_in.shape[
        0] % nlat == 0, f"invalid dimensions: nlat x nlat != {f_in.shape[0]} (len(f_in))"
    nlev = f_in.shape[0] // nlat

    # remove leading/trailing whitespace from column names
    f_in.columns = [c.strip() for c in f_in.columns]

    return f_in, nlev


def read_metadata(metadata_dir: Path, lat_fname: str, lon_fname: str, lev_fname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # load latitude and vertical level data
    lat = np.load(metadata_dir / lat_fname)
    lon = np.load(metadata_dir / lon_fname)
    lev = pd.read_csv(metadata_dir / lev_fname,
                      header=0, index_col=None).values

    return lat, lon, lev.T


def compute_tcwv(q: np.ndarray, lev: np.ndarray) -> np.ndarray:
    # compute total column water vapor, see https://resources.eumetrain.org/data/3/359/print_2.htm for similar formula
    g = 9.80665  # m/s^2

    pa = 100 * lev  # convert hPa to Pa

    tcwv = -(1 / g) * scipy.integrate.trapezoid(q, pa, axis=1)

    return tcwv


def main(
        fort_output_dir: Path,
        f_in_name: str,
        stats_dir: Path,
        means_fname: str,
        stds_fname: str,
        metadata_dir: Path,
        lat_fname: str,
        lon_fname: str,
        lev_fname: str,
        channels_fname: str,
        output_to_dir: Path,
        f_out_name: str,
        nlat: int,
        keep_plevs: list[int]
):

    df, nlev = read_to_df(fort_output_dir, f_in_name, nlat)

    # extract vertical RH profile, convert to percentage
    rh = df["ZRH"].values[:nlev] * 100

    # remove RH from dataframe
    df = df.drop(columns=["ZRH"])

    # read latitude and vertical levels data
    lat, lon, (plev, etalev) = read_metadata(
        metadata_dir, lat_fname, lon_fname, lev_fname)
    plev, etalev = plev.T, etalev.T

    # compute total column water vapor
    q = df["ZQ"].values.reshape(nlat, nlev)
    tcwv = compute_tcwv(q, plev)

    # keep only the desired vertical levels
    keep_plevs = np.array(keep_plevs)
    keep_idxs = np.where(np.isin(plev, keep_plevs))[0]

    # similar to above, tile across lat to match the shape of the other variables
    rh = np.tile(rh[keep_idxs], (nlat, 1))

    # create pressure vars, both constant due to initiation at sea level
    sp = np.full_like(tcwv, 1013.25)
    msl = np.full_like(tcwv, 1013.25)

    # create xarray dataset
    ds_73 = xr.Dataset(
        {
            "tcwv": (["lat"], tcwv),
            "sp": (["lat"], sp),
            "msl": (["lat"], msl),
        },
        coords={
            "lat": lat
        }
    )

    # iteratively add pressure-coordinate necessary variables and levels to the dataset
    for lname, uname in [("u", "ZU"), ("v", "ZV"), ("t", "ZT"), ("z", "ZPHI_F"), (("r", "RH"), rh)]:

        # rh case, processing done already
        if isinstance(lname, tuple):
            sub = uname  # uname contains rh data
            (lname, uname) = lname  # lname contains both names

        # other vars
        else:
            # reshape to (nlat, nlev) and keep only the vertical levels needed for SFNO
            sub = df[uname].values.reshape(nlat, nlev)[:, keep_idxs]

        for i, plev in enumerate(keep_plevs.astype(str)):
            name = f"{lname}{plev}"
            ds_73[name] = (["lat"], sub[:, i])

    # add height level variables to dataset, all at lowest model level [0] or 1013.25 hPa
    u = df["ZU"].values.reshape(nlat, nlev)
    v = df["ZV"].values.reshape(nlat, nlev)
    t = df["ZT"].values.reshape(nlat, nlev)
    # lowest model level instead of 10 meter winds
    ds_73["u10m"] = (["lat"], u[:, 0])
    ds_73["v10m"] = (["lat"], v[:, 0])
    # lowest model level instead of 100 meter winds
    ds_73["u100m"] = (["lat"], u[:, 0])
    ds_73["v100m"] = (["lat"], v[:, 0])
    # lowest model level instead of 2 meter temperature
    ds_73["t2m"] = (["lat"], t[:, 0])

    # find channel order
    channels = np.loadtxt(metadata_dir / channels_fname, dtype=str)

    # convert from data variables to channel as coordinate
    arr_lst = [ds_73[ch].data for ch in channels]
    stacked = np.stack(arr_lst, axis=0)
    ds_73 = xr.DataArray(stacked, dims=["channel", "lat"], coords={"channel": channels, "lat": lat})

    # now standardize the dataset
    # means and stds are global and time invariant, unlike other versions of FCN
    means = np.load(stats_dir / means_fname).reshape(-1, 1)
    stds = np.load(stats_dir / stds_fname).reshape(-1, 1)

    ds_73 = (ds_73 - means) / stds
    # for i, (ch, m, s) in enumerate(zip(channels, means, stds)):
    #     print(f"{ch}: {m} {s}")

    # expand all variables along longitude dimension
    # while Bouvier et al. only outputs one meridional slice, we need the whole domain for SFNO
    ds_73 = ds_73.expand_dims({"lon": lon}, axis=1) # unsure which axis to expand along
    ds_73 = ds_73.to_dataset(name="data")

    # save to disk
    ds_73.to_netcdf(output_to_dir / f_out_name)

    # check
    # print(ds_73)
    # for i, l in enumerate(lat):
    #     if i % 10 == 0:
    #         print("")
    #     print(l, end=",")

    # print("-------------------")
    # for i, l in enumerate(lon):
    #     if i % 10 == 0:
    #         print("")
    #     print(l, end=",")



if __name__ == "__main__":

    data_dir = Path(
        "/N/slate/jmelms/projects/FCN_dynamical_testing/data/initial_conditions/")

    stats_dir = Path(
        "/N/u/jmelms/BigRed200/projects/dynamical-tests-FCN/data/"
    )

    metadata_dir = Path(
        "/N/u/jmelms/BigRed200/projects/dynamical-tests-FCN/metadata/")

    main(
        fort_output_dir=data_dir / "raw_fort_output",
        f_in_name="fields1.csv",

        stats_dir=stats_dir,
        means_fname="global_means.npy",
        stds_fname="global_stds.npy",

        metadata_dir=metadata_dir,
        lat_fname="latitude.npy",
        lon_fname="longitude.npy",
        lev_fname="p_eta_levels_full.txt",
        channels_fname="fcnv2_sm_channel_order.txt",

        output_to_dir=data_dir / "processed_ic_sets" / "test_data_source" / "idealized",
        f_out_name="data.h5",

        nlat=721,
        keep_plevs=[1000, 925, 850, 700, 600, 500,
                    400, 300, 250, 200, 150, 100, 50]  # 13 levels used for 73 ch SFNO
    )
