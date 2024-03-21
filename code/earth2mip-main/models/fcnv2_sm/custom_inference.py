from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
from earth2mip.initial_conditions import cds, hdf5
from earth2mip import Settings, ModelRegistry, inference_ensemble
import numpy as np
import datetime
import os

# Set number of GPUs to use to 1 (using zero currently, check fcnv2_sm_load parameter "device")
os.environ["WORLD_SIZE"] = "1"
os.environ["MODEL_REGISTRY"] = "/N/u/jmelms/BigRed200/projects/dynamical-tests-FCN/code/earth2mip-main/models"

config = Settings(MODEL_REGISTRY=os.environ["MODEL_REGISTRY"])
registry = ModelRegistry(config.MODEL_REGISTRY)

# With the enviroment variables set now we import Earth-2 MIP

# Load model(s) from registry
package = registry.get_model("fcnv2_sm")
print(package.root)
print("loading FCNv2 small model, this can take a bit")
sfno_inference_model = fcnv2_sm_load(package, device="cpu")

time = datetime.datetime(1001, 1, 1)

data_source = hdf5.DataSource.from_path(root="/N/slate/jmelms/projects/FCN_dynamical_testing/data/initial_conditions/processed_ic_sets/test_data_source/")

ds = inference_ensemble.run_basic_inference(
    sfno_inference_model,
    n=2,
    data_source=data_source,
    time=time,
)

print(ds)
