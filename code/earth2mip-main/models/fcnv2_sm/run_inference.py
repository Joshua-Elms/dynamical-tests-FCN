from inference_utils import run_custom_inference, run_simple_inference
import datetime

# make sure to set the environment variable MODEL_REGISTRY before running this script

run_custom_inference(
        data_source_path="/N/slate/jmelms/projects/FCN_dynamical_testing/data/initial_conditions/processed_ic_sets/test_data_source/",
        output_path="/N/slate/jmelms/projects/FCN_dynamical_testing/data/output/test2.nc",
        n_iters=3,
        start_time=datetime.datetime(1970, 1, 1),
        device="cpu",
        vocal=True
    )

# run_simple_inference(
#         output_path="/N/slate/jmelms/projects/FCN_dynamical_testing/data/output/test_era5ic_output.nc",
#         n_iters=1,
#         start_time=datetime.datetime(1970, 1, 1),
#         device="cpu",
#         vocal=True
# )