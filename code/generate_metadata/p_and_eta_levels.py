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

def write_levels(output_path, p_levels = None, eta_levels = None):
    """
    Use p or eta to find the other and write both to a file
    """
    if p_levels is None and eta_levels is None:
        raise ValueError("Either p_levels or eta_levels must be provided")
    
    elif p_levels is None:
        p_levels = eta_to_p(eta_levels)

    elif eta_levels is None:
        eta_levels = p_to_eta(p_levels)

    merged = np.vstack((p_levels, eta_levels)).T
    np.savetxt(output_path, merged, delimiter=",", fmt="%.15f", header="p,eta", comments="")
    print(f"Levels written to {output_path}")

if __name__=="__main__":
    p_levels = np.array([1013.25, 1000, 925, 850, 750, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
    output_path = "levels.txt"
    write_levels(output_path, p_levels=p_levels)