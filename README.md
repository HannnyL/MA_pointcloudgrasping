# MA_Hanyu: Point Cloud Grasping

This repository contains the code, models, and results used for my
master's thesis on point cloud-based grasping.

## Table of Contents

-   Overview
-   Environment & Installation
-   Project Structure
-   Usage
    -   1.  Main Grasping Script

    -   2.  Runtime Measurement Script

    -   3.  Grasp Result Visualization Script

    -   4.  Grasp Result Analysis Script

    -   5.  Grasp Store Script
-   Outputs
-   Notes

## Overview

This project performs grasp planning based on 3D point clouds. The repository includes:  
- Code for grasp generation and evaluation 
- Predefined gripper configurations 
- Point cloud models of various objects 
- Evaluation results

## Environment & Installation

### Requirements

-   Python 3.9 or later
-   Recommended libraries:
    -   open3d
    -   numpy
    -   pyyaml
    -   opencv-python
    -   scikit-learn
    -   shapely

Install dependencies:

    pip install open3d numpy pyyaml opencv-python scikit-learn shapely

## Project Structure

    MA_Hanyu_pointcloudgrasping/
    ├── main_script_parallel_grasp.py
    ├── analyze_grasp_npz.py
    ├── grasp_store.py
    ├── measure_runtime.py
    ├── visualize_grasp_results.py
    ├── gripper_parameter/
    │   ├── franka_*.yaml
    │   └── schunk_*.yaml
    ├── object/
    │   ├── blender/
    │   ├── cubesat_space/
    │   └── validation_object/
    ├── result/
    │   ├── {time}-{gripper_name}-{object_name}-{plane_fitting_error}/
    |   |   ├── grasp_data/
    |   |   |   ├── *.csv
    |   |   |   └── *.npz
    |   |   └── image/
    │   └── runtime_record/
    └── README.md

## Usage

### 1. Main Grasping Script

    python main_script_parallel_grasp.py

### 2. Runtime Measurement Script

    python measure_runtime.py

### 3. Grasp Result Visualization Script

    python visualize_grasp_results.py

### 4. Grasp Result Analysis Script

    python analyze_grasp_npz.py

### 5. Grasp Store Script

Automatically used by other scripts.

## Outputs

-   Grasp `.npz` data
-   Visualization images
-   Runtime logs
-   Evaluation metrics

## Notes

- ```main_script_parallel_grasp.py```
  - Run ```main_script_parallel_grasp.py``` to execute the main function of the code.
  - Remember to configure the point cloud path, the gripper configuration file path, and other parameters under “Code parameters”, which determine the performance and behavior of the pipeline. There are also several variables under “image console”, which control whether visualization is enabled, which images to display, and whether to save them.
  - Point cloud files (```.pcd```) can be found in:
    -  ```object/blender``` — ideal object models
    -  ```object/cubesat``` — CubeSat components and the full assembly
    - ```object/validation_object``` — the 8 evaluation objects
  - Gripper configuration files can be found in ```gripper_parameter/.``` Configuration files for Franka Hand and Schunk are provided.

  - The results generated during execution (images and grasp point data) will be saved in the ```result/``` directory under a timestamped folder.

- ```measure_runtime.py```
  - This script is used to record the runtime of the main script. It logs both the total runtime and the 20 functions with the longest execution time. The results will be stored in  ```result/runtime_record/```.
  - Simply run this script, and it will automatically execute the main script. Remember to set the correct ```.pcd``` and ```.yaml``` file paths in the main script in advance.

- ```visualize_grasp_results.py```
  - This script visualizes grasp point results. Before execution, set the necessary paths and parameters at the top of the script under “Set all necessary paths and TARGET before execution.”
  - Provide the ```.npz``` grasp data file and its corresponding ```.pcd``` file (the ```.pcd``` filename is typically included in the ```.npz``` filename).
  - The ```TARGET``` parameter determines which grasp point(s) to visualize, including specific selections such as ```target_pair```, ```target_edge```, or ```target_point````.

- ```analyze_grasp_npz.py```
  - This script computes evaluation metrics. Set the paths at the bottom of the script: ```FILE_PATH```, ```PCD_PATH```, and ```FILE_LIST```, and specify the execution mode in ```choice```.
  - It can compute:
    -   the positional deviation of the top-k grasp points for an object,
    -   the repeatability deviation of the best grasp point across multiple ```.npz``` files,
    -   and the accuracy deviation of the best grasp point.
- ```grasp_store.py```
  - This script processes grasp data and is automatically executed by other scripts.



