from pyKilomatch import preprocess, motionEstimation, computeWaveformFeatures, iterativeClustering, autoCuration
import json

with open('settings.json', 'r') as f:
    user_settings = json.load(f)

preprocess(user_settings)
motionEstimation(user_settings)
computeWaveformFeatures(user_settings)
iterativeClustering(user_settings)
autoCuration(user_settings)
