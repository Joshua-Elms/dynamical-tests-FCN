import numpy as np

def p_to_eta(p, ps: float = 1013.25):
    """
    Convert pressure levels to eta levels
    Formula: eta = (p/ps), where ps = surface pressure
    """
    return p/ps

def eta_to_p(eta, ps: float = 1013.25):
    """
    Convert eta levels to pressure levels
    Formula: p = eta*ps, where ps = surface pressure
    """
    return eta*ps

if __name__=="__main__":
    ps = 1013.25 # mb at lowest model level (surface)
    p_levels = np.array([1013.25, 1000, 925, 850, 750, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
    output_path = "levels.txt"

    eta_levels = p_to_eta(p_levels, ps)
    merged = np.vstack((p_levels, eta_levels)).T

    np.savetxt(output_path, merged, delimiter=",", fmt="%.10f", header="p,eta", comments="")