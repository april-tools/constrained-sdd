# Constrained-SDD
[![Python application](https://github.com/april-tools/constrained-sdd/actions/workflows/python-app.yml/badge.svg)](https://github.com/april-tools/constrained-sdd/actions/workflows/python-app.yml)

This repository contains an annotated version of the Stanford Drone Dataset[1]. We segmented the first 50 images from SDD and drew polygons for buildings, obstacles and offroad. Our trajectories only follow the roads/walking paths.

# Installation

Using this dataset is really easy! First, you can install this package using pypi:
```bash
 pip install constrained-sdd
```
then just use the provided class to download the dataset and load it into memory:
```python
import sdd.constrained_sdd as csdd

img_id = 0
folder = "data/sdd"
sdd = csdd.ConstrainedStanfordDroneDataset(0, sdd_data_path=folder, download=True)
train, val, test = sdd.get_dataset()
```
this creates a dataset with a list of trajectories in train/val/test, e.g. useful for position prediction.

You can also create the trajectory-prediction task via:
```python
train, val, test = sdd.get_trajectory_prediction_dataset(window_size, sampling_rate)
```


## Citation

Leander Kurscheidt, Paolo Morettin, Roberto Sebastiani, Andrea Passerini, Antonio Vergari, A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction, (to appear)

## Examples

![Image 12](imgs/img12.png)
![Image 2](imgs/img2.png)
![Image 0](imgs/img0.png)
![Image 18](imgs/img18.png)

An overview over all the images can be seen in `analysis/viz_data.ipynb`.

## Further Details

We annotate the first 50 pictures (scenes in the original dataset) with trajectories of the Stanford Drone Dataset[1], as extracted by P2T[2]. We select all trajectories following the road and delete obvious errors, like projecting a trajectory that slightly touches a building outwards and deleting trajectories that suddenly jump around. We annotate using three classes: Building, Obstacle and Offroad, which forms our constraint-set. Additionally, we have annotated the entrance of a building to differentiate trajectories that enter a building from those who not, but this is unused. All trajectories in our dataset are constraint-abiding.


[1]  A. Robicquet, A. Sadeghian, A. Alahi, S. Savarese, Learning Social Etiquette: Human Trajectory Prediction In Crowded Scenes in European Conference on Computer Vision (ECCV), 2016. 
[2] Nachiket Deo, Mohan M. Trivedi, Trajectory Forecasts in Unknown Environments Conditioned on Grid-Based Plans, 2021