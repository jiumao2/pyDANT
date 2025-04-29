from pyKilomatch import preprocess, motionEstimation, finalClustering, autoCuration
import hjson
import time
import numpy as np
import os

path_settings = r'./settings.json' # It should be the path to your settings.json file

with open(path_settings, 'r') as f:
    user_settings = hjson.load(f)

time_start = time.time()

preprocess(user_settings)
motionEstimation(user_settings)
finalClustering(user_settings)
autoCuration(user_settings)

# Save the run time
run_time_sec = time.time() - time_start
print(f"Total run time: {run_time_sec:.2f} seconds")
np.save(os.path.join(user_settings['output_folder'], 'RunTimeSec.npy'), run_time_sec)

