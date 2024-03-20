from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
from earth2mip.initial_conditions import cds, hdf5
from earth2mip import Settings, ModelRegistry, inference_ensemble
import numpy as np
import datetime
import os

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
# model_registry = os.path.join(os.path.dirname(
#     os.path.realpath(os.getcwd())), "models")
# os.makedirs(model_registry, exist_ok=True)
# os.environ["MODEL_REGISTRY"] = model_registry
os.environ["MODEL_REGISTRY"] = "/N/u/jmelms/BigRed200/projects/dynamical-tests-FCN/code/earth2mip-main/models"

config = Settings(MODEL_REGISTRY=os.environ["MODEL_REGISTRY"])
registry = ModelRegistry(config.MODEL_REGISTRY)


# With the enviroment variables set now we import Earth-2 MIP

# Load model(s) from registry
# breakpoint()
package = registry.get_model("fcnv2_sm")
print(package.root)
print("loading FCNv2 small model, this can take a bit")
sfno_inference_model = fcnv2_sm_load(package, device="cpu")

# data_source = cds.DataSource(sfno_inference_model.in_channel_names)
# output = "path/"
# time = datetime.datetime(2018, 1, 1)

# data_source = hdf5.DataSource.from_path()

# ds = inference_ensemble.run_basic_inference(
#     sfno_inference_model,
#     n=1,
#     data_source=data_source,
#     time=time,
# )
