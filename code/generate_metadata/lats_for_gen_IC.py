import numpy as np

output_path = "lats_for_gen_IC.txt"
start, stop, step = (90, -90, -0.25)
lat_deg = np.arange(start, stop+step, step)
lat_rad = np.deg2rad(lat_deg)
merged = np.vstack((lat_deg, lat_rad)).T
np.savetxt(output_path, merged, delimiter=",", fmt="%.10f", header="lat_deg,lat_rad", comments="")