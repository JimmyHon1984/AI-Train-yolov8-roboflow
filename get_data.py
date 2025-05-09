# This dataset is flower

from roboflow import Roboflow
rf = Roboflow(api_key="fXWidy1XD85NHYANrSsh")
project = rf.workspace("research-kiwbb").project("annot-0abet")
version = project.version(1)
dataset = version.download("yolov8")

# or terminal
# curl -L "https://universe.roboflow.com/ds/Az7Nm6mc97?key=Gcijbs26Gz" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip