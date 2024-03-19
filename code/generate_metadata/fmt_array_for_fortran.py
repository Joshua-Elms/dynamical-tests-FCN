import numpy as np

def _fmtd_arr_to_fortran_str(array: list, name: str, precision):
    """
    Convert a (pseudo) 2D array to a string that can be used in Fortran code.
    """
    out = ""
    out += f"{name} = ["
    nchar = len(out)
    lines = [", ".join([f"{float(val):.{precision}f}" for val in sub if val != "."]) for sub in array]
    out += f", &\n&{' ' * (nchar - 1)}".join(lines)
    out += "]"

    return out

def numeric_arr_to_fort_str(arr: list, name: str, precision: int = 15, row_width: int = 3):
    arr += ["." for _ in range(row_width - (((len(arr)) % row_width)))]
    twoD_arr = np.array(arr).reshape(-1, row_width).tolist()
    return _fmtd_arr_to_fortran_str(twoD_arr, name, precision)


if __name__=="__main__":
    # Generate VETAF array
    full_arr = np.genfromtxt("levels_full.txt", delimiter=",", skip_header=1)
    eta_arr = full_arr[:, 1].tolist()
    fstr = numeric_arr_to_fort_str(eta_arr, "VETAF", row_width=2)
    print(fstr)

    # Generate GELAT_DEG and GELAT arrays
    arr = np.genfromtxt("lats_for_gen_IC.txt", delimiter=",", skip_header=1)
    deg = arr[:, 0].tolist()
    rad = arr[:, 1].tolist()
    deg_str = numeric_arr_to_fort_str(deg, "GELAT_DEG", row_width=8, precision=2)
    rad_str = numeric_arr_to_fort_str(rad, "GELAT", row_width=5, precision=8)

    # print(deg_str)
    # print(rad_str)