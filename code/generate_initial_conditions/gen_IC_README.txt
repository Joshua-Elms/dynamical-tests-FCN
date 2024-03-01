Details on the CLI for gen_init_conditions.out

gen_IC.F90 has been hardcoded (values for VETAF, GELAT_DEG, and GELAT) for FourCastNet, particularly the 73 channel SFNO version

Args in order:
    NLAT     - number of latitude ticks, must agree with GELAT_DEG and GELAT
    NLEV     - number of levels, must agree with VETAF
    ZN       - Jet width
    ZB       - Jet height
    ZRH0     - Surface level relative humidity (%)
    ZT0      - Average surface virtual temperature (K)
    ZU0      - works with ZB to affect amplitude of zonal mean wind speed (m/s)
    ZGAMMA   - Lapse rate (K/m)
    MOISTURE - 41 for dry run, 42 for moist
    FILENAME - Output location for csv file containing NLAT x NLEV rows and all fields needed to run FCN

Defaults: 
    NLAT     - 320
    NLEV     - 137
    ZN       - 3
    ZB       - 2.0
    ZRH0     - 0.8
    ZT0      - 288.0
    ZU0      - 35.0
    ZGAMMA   - 0.005
    MOISTURE - 42
    FILENAME - "fields.csv"

Default cmd: 
    ./gen_IC.out 320 137 3 2.0 0.8 288.0 35.0 0.005 42 fields.csv