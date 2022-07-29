# Import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="6LJjZJNTzHyDYNiDimNxrA9MX",
    project_name="rams-ed",
    workspace="quan3a1dvt",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 2e-5,
    "steps": 100000,
    "batch_size": 8,
}
experiment.log_parameters(hyper_params)


# Run your code and go to /