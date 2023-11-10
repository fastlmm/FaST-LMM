import os
os.environ["fastlmmuseanymkllib"]="1"
executablerel = "FastLmmC"
dirname = os.path.dirname(os.path.realpath(__file__))
executable = os.path.join(dirname,executablerel)