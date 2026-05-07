from pyDANT import runDANTMultiShank
import hjson

path_settings = r'./settings.json' # It should be the path to your settings.json file

with open(path_settings, 'r') as f:
    user_settings = hjson.load(f)

runDANTMultiShank(user_settings)
