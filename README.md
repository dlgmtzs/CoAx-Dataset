# Utility Repository for [CoAx-Dataset](https://dlgmtzs.github.io/dataset-coax)

## About this Repository
This repository provides an introduction on how to work with and load the dataset and its derived data. Short intro and hands-on by showing a code snippet using a [jupyter notebook](./notebooks/DatasetPreview.ipynb) . Utility functions are encapsulated in a `util.py`, that can be located in the `src` directory of this GitHub-Repository.

### What utilities / functionality is provided?
- [x] Loading of the CoAx dataset
- [x] Extraction of camera parameters intrinsics and extrinsics.
- [x] Naive calculation of a 3D bounding box based on two 3d points.
- [x] Extraction of Pointcloud and Pixel Image for a specified identifier (subject,task,take and frame)
- [x] Plotting of candidate Pointcloud and Pixel Image (frame) with labeled 2d/3d bounding boxes and center points for each object

## How to get the dataset?
You can find and download the [CoAx-Dataset](https://dlgmtzs.github.io/dataset-coax) following the highlighted link.

## Setup Repository


- Clone Repository

```cmd
git clone https://github.com/dlgmtzs/coax-dataset.git 
cd coax-dataset/
```

- Create `data` directory and symbolically link the dataset

```cmd
mkdir data
ln -s PATH_TO_CoAx_DATASET/* ./data/.
```

- Create `python env` and build coax package, install requirements

```cmd
python3 -m venv venv
source venv/bin/activate

pip install --editable .

: install requirements via cmd 
pip install notebook matplotlib open3d pyyaml pandas numpy opencv-python transformations

: OR 

: install requirements via "requirements" file
pip install -r requirements.txt 
```

- with activated `python env` run your notebook

```cmd
jupyter notebook
```

---

Feel free to open issues or PR to contribute and help improve this work
