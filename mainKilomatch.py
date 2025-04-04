from pyKilomatch import preprocess, motionEstimation, computeWaveformFeatures, iterativeClustering, autoCuration
import hjson

path_settings = r'./settings.json' # It should be the path to your settings.json file

with open(path_settings, 'r') as f:
    user_settings = hjson.load(f)

preprocess(user_settings)
motionEstimation(user_settings)
computeWaveformFeatures(user_settings)
iterativeClustering(user_settings)
autoCuration(user_settings)
