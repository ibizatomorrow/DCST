from pathlib import Path
import signal
import numpy as np

from nni.experiment import Experiment

# Define search space
search_space = {
    'dataset': {'_type': 'choice', '_value': ['PEMSBAY']},
    'gpu_num': {'_type': 'choice', '_value': [0]},
    'kd_weight': {'_type': 'choice', '_value': [0.5]},
}

# Define experimental object
experiment = Experiment('local')

# run 'python train.py'
experiment.config.trial_command = 'python train.py'

# Save location
experiment.config.experiment_working_directory = '../experiments'

# Search space
experiment.config.search_space = search_space

# Search pattern
experiment.config.tuner.name = 'Random'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

# Parallel number
experiment.config.trial_concurrency = 1

# Name
experiment.config.experiment_name = "STGCN_KD"

# Run it!
experiment.run(port=50001, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()