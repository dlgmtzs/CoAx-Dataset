# -*- coding: utf-8 -*-
"""
Helper class for providing utility function to handle the coax dataset. 
See "../../notebooks/DatasetPreview.ipynb" for usage examples.
"""
__author__ = "Dimitrios Lagamtzis"
__copyright__ = "Dimitrios Lagamtzis"
__license__ = "mit"

import glob
import os
import time

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from transformations import *

# Global variable for colors used in bounding box plots
colors_ = [
    'darkslategray', 'midnightblue', 'darkgreen', 'mediumturquoise', 'red',
    'orange', 'yellow', 'mediumvioletred', 'mediumblue', 'lime',
    'mediumspringgreen', 'thistle', 'fuchsia', 'dodgerblue', 'khaki', 'salmon'
]


def get_sequence_identifier(subject: int, task: int, take: int):
    '''Sequence Identifier -> Subject, Task, Take 
        -->Sequence Identifier = Subject + Task + Take
    Input
      subject: id of subject
      task: id of task
      take: id of take
    Output
      sequence identifier as str 
    '''
    return 'subject_'+str(subject)+'/task_'+str(task)+'/take_'+str(take)+'/'


def get_camera_information(cam_param_path: str, calib_param_path: str):
    '''Extraction of camera parameters intrinsics and extrinsics

    Input
      cam_param_path: str path of camera parameter json
      calib_param_path: str path of calibration yaml
    Output
      camParams: camera parameters in open3d format
    '''
    camera_info = pd.read_json(cam_param_path, lines=True)

    camParams = o3d.camera.PinholeCameraParameters()

    camParams.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=camera_info['width'][0], height=camera_info['height'][0],
        fx=camera_info["fx"][0], fy=camera_info["fy"][0],
        cx=camera_info["ppx"][0], cy=camera_info["ppy"][0])

    camParams.extrinsic = get_extrinsic_from_calib_values(
        calib_path=calib_param_path)

    return camParams


def get_extrinsic_from_calib_values(calib_path: str):
    '''Extraction of camera extrinsic parameters based on calibration parameters of the camera.

    Input
      calib_path: str path of calibration yaml
    Output
      camera extrinsic matrix
    '''
    with open(calib_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

        qw = data["transformation"]["qw"]
        qx = data["transformation"]["qx"]
        qy = data["transformation"]["qy"]
        qz = data["transformation"]["qz"]
        x = data["transformation"]["x"]
        y = data["transformation"]["y"]
        z = data["transformation"]["z"]

        f.close()

    # Quaternions w+ix+jy+kz are represented as [w, x, y, z].
    # Careful with older implementations (ros tf), where q is build in different order
    q = np.array([qw, qx, qy, qz])
    cam2world = quaternion_matrix(q)  # rotation
    cam2world[:, 3] = np.array([x, y, z, 1])  # translation

    return np.linalg.inv(cam2world)


def get_3d_bounding_box(point1: list, point2: list):
    '''Naive calculation of a 3d bounding box based on two 3d points.

    Input
      point1: 3d point 1 of 2 that spanning a 3d bounding box: list with x,y,z coordinate
      point2: 3d point 1 of 2 that spanning a 3d bounding box: list with x,y,z coordinate
    Output
      points: 3d points of 3d bounding box based on point1 and point2
      edges: edges between 3d points of 3d bounding box based on point1 and point2
    '''
    # Positional naming of all points of a 3d bounding box
    # point_*_*_* --> rear/front _ top/bottom _ left/right
    point_r_t_l = point1
    point_f_b_r = point2
    point_r_t_r = [point_f_b_r[0], point_r_t_l[-2], point_r_t_l[-1]]
    point_r_b_r = [point_f_b_r[0], point_f_b_r[1], point_r_t_l[-1]]
    point_f_t_l = [point_r_t_l[0], point_r_t_l[1], point_f_b_r[2]]
    point_f_b_l = [point_r_t_l[0], point_f_b_r[1], point_f_b_r[2]]
    point_f_t_r = [point_r_t_r[0], point_r_t_r[1], point_f_b_r[2]]
    point_r_b_l = [point_r_t_l[0], point_r_b_r[1], point_r_b_r[2]]
    points = [point_r_t_l, point_f_b_r, point_r_t_r, point_r_b_r,
              point_f_t_l, point_f_b_l, point_f_t_r, point_r_b_l]

    # Positional naming of all edges devided in top,bottom,middle layers of edges
    t_on = np.array([np.array(point_r_t_l), np.array(point_r_t_r)])
    t_tw = np.array([np.array(point_r_t_l), np.array(point_f_t_l)])
    t_th = np.array([np.array(point_r_t_r), np.array(point_f_t_r)])
    t_fo = np.array([np.array(point_f_t_l), np.array(point_f_t_r)])

    b_on = np.array([np.array(point_r_b_l), np.array(point_r_b_r)])
    b_tw = np.array([np.array(point_r_b_l), np.array(point_f_b_l)])
    b_th = np.array([np.array(point_r_b_r), np.array(point_f_b_r)])
    b_fo = np.array([np.array(point_f_b_l), np.array(point_f_b_r)])

    m_on = np.array([np.array(point_f_t_l), np.array(point_f_b_l)])
    m_tw = np.array([np.array(point_r_t_l), np.array(point_r_b_l)])
    m_th = np.array([np.array(point_r_t_r), np.array(point_r_b_r)])
    m_fo = np.array([np.array(point_f_t_r), np.array(point_f_b_r)])

    edges = [t_on, t_tw, t_th, t_fo, b_on, b_tw,
             b_th, b_fo, m_on, m_tw, m_th, m_fo]

    return points, edges


def set_axes_equal(ax):
    '''source: stackoverflow /questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    --> altered by author.
    '''
    '''Make axes of 3d plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3d.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.35 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.view_init(elev=26, azim=-39)


def frameWrapper(frm, dim: int):
    '''Processing candidate frame's derived data. Either with 2d or 3d information.

    Input
      frm: derived data of given frame 
      dim: 2d or 3d identifier
    Output
      tmp: processed dataframe with necessary columns
    '''
    assert (dim == 2 or dim == 3), "Dimension has to be either 2d or 3d"
    cols = ['bounding_box', 'x', 'y', 'z', 'class',
            'class_name', 'class_instance', 'certainty']
    if dim == 2:
        cols.remove('z')
    tmp = []
    for index, row in frm.iterrows():
        cnt = row.center_point
        tmp_dict = dict({
            'bounding_box': row.bounding_box,
            'x': cnt['x'],
            'y': cnt['y'],
            'class': row.class_index,
            'class_name': row.class_name,
            'class_instance': row.instance_name,
            'certainty': row.certainty
        })
        if dim == 3:
            pos_ = list(tmp_dict.keys()).index('y')
            items = list(tmp_dict.items())
            items.insert(pos_, ('z', cnt['z']))
            tmp_dict = dict(items)
        tmp.append(tmp_dict)

    return pd.DataFrame(tmp)


def get_rgb_xyz_for_frame(frame: int, images: list, depth_images: list, camParams):
    '''Extraction of Pointcloud given frame, rgb image and depth image

    Input
      frame: int frame identifier
      images: list of images for sequence identifier
      depth_images: list of depth images for sequence identifier
      camParams: camera parameters (intrinsics and extrinsics)
    Output
      rgbxyz: Pointcloud
    '''
    image_test = images[frame]
    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    depth_image_test = depth_images[frame]
    depth_image = o3d.geometry.Image(
        (depth_image_test / 1000).astype(np.float32))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_image,
        intrinsic=camParams.intrinsic,
        extrinsic=camParams.extrinsic,
        depth_scale=1.0,
        depth_trunc=1000.0,
        stride=1,
        project_valid_depth_only=False)

    pcd_array = np.asarray(pcd.points)

    height = depth_image_test.shape[0]
    width = depth_image_test.shape[1]
    rgbxyz = np.zeros(shape=(height, width, 6))

    rgbxyz[:, :, 0:3] = image_test / 255
    rgbxyz[:, :, 3] = pcd_array[:, 0].reshape(
        (rgbxyz.shape[0], rgbxyz.shape[1]))  # x
    rgbxyz[:, :, 4] = pcd_array[:, 1].reshape(
        (rgbxyz.shape[0], rgbxyz.shape[1]))  # y
    rgbxyz[:, :, 5] = pcd_array[:, 2].reshape(
        (rgbxyz.shape[0], rgbxyz.shape[1]))  # z

    return rgbxyz


def get_2d_processed_dfs_per_frame(derived_data_path: str, identifier: str):
    '''Extraction of dataframes per grouped by object and frames for all frames of sequence identifier;
       using 2d derived data (2d_objects)

    Input
      derived_data_path: path of derived data
      identifier: sequence identifier
    Output
          dfs: m (number of frames for take) dataframes --> dataframes consisting of x [number of objects] rows
      obj_dfs: n (number of objects for task) dataframes --> object dataframes consisting of y [number of frames] rows
    '''
    candids = []
    for root, subdirs, files in os.walk(derived_data_path):
        if root.split("/")[-1] == "2d_objects":
            candids.append(root)
    candids = sorted(candids)

    candid_path = [s for s in candids if identifier in s][0]

    candid_frames = sorted(glob.glob(candid_path + '/*.json'))

    data_frames = []
    for frame in candid_frames:
        tmp = pd.read_json(frame)
        tmp.insert(
            0, "center_point",
            tmp.apply(lambda row: dict(
                {
                    "x": np.mean([row.bounding_box["x"], row.bounding_box["x"]+row.bounding_box["w"]]),
                    "y": np.mean([row.bounding_box["y"], row.bounding_box["y"]+row.bounding_box["h"]])
                }),
                axis=1))
        data_frames.append(tmp)

    dfs = []
    for it in data_frames:
        tmp = frameWrapper(it, dim=2)
        dfs.append(tmp)

    df = pd.concat(dfs)
    df['frame'] = df.index
    df['frame'] = df['frame'].diff().ne(1).cumsum() - 1
    df['data_id'] = '-'.join(candid_path.split('/')[3:-1])
    df = df.reset_index(drop=True)

    grouped = df.groupby(df.class_name)
    groups = list(grouped.groups.keys())
    obj_dfs = [grouped.get_group(grp).reset_index(drop=True) for grp in groups]
    obj_dfs = [obj_df.set_index(obj_df.frame) for obj_df in obj_dfs]

    return dfs, obj_dfs


def plot_rgb_with_bounding_boxes(frame: int, images: list, obj_dfs: list, save=False):
    '''Plot candidate image (frame) with labeled 2d bounding boxes and center points for each object

    Input
      frame: int id of frame in take
      images: list of rgb images
      obj_dfs: n (number of objects for task) dataframes --> object dataframes consisting of y [number of frames] rows
      save: default set parameter to save plot -> named after its type + timestamp (human readable)
    '''

    rgb = [list(np.round(matplotlib.colors.hex2color(col), 5))
           for col in colors_]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.imshow(cv2.cvtColor(images[frame], cv2.COLOR_BGR2RGB))
    for it, obj_ in enumerate(obj_dfs):
        label_ = obj_.class_name.unique()[0]
        if not obj_[obj_.frame == frame].empty:
            ax.plot(obj_.loc[frame:frame, "x"].values,
                    obj_.loc[frame:frame, "y"].values,
                    'o-',
                    color=rgb[it],
                    ms=5,
                    label=label_)
        try:
            specd = obj_[obj_.frame == frame]
            bbox = specd["bounding_box"].values[0]
            ax = plt.gca()

            x = bbox["x"]
            y = bbox["y"]
            w = bbox["w"]
            h = bbox["h"]
            if (w == 1) or (h == 1):
                x -= 10
                y -= 10
                w = 20
                h = 20
            rect = Rectangle((x, y), w, h, linewidth=1,
                             edgecolor=rgb[it], facecolor='none')
            ax.add_patch(rect)
        except:
            pass
    ax.legend(bbox_to_anchor=(0.2, 0.6))
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        date_str = '_'.join(map(str, time.localtime()[:5]))
        fig.savefig('./'+"rgb_"+date_str+'.png', dpi=300, bbox_inches="tight")


def get_3d_processed_dfs_per_frame(derived_data_path: str, identifier: str):
    '''Extraction of dataframes per grouped by object and frames for all frames of sequence identifier;
       using 3d derived data (3d_objects)

    Input
      derived_data_path: path of derived data
      identifier: sequence identifier
    Output
          dfs: m (number of frames for take) dataframes --> dataframes consisting of x [number of objects] rows
      obj_dfs: n (number of objects for task) dataframes --> object dataframes consisting of y [number of frames] rows
    '''
    candids = []
    for root, subdirs, files in os.walk(derived_data_path):
        if root.split("/")[-1] == "3d_objects":
            candids.append(root)
    candids = sorted(candids)

    candid_path = [s for s in candids if identifier in s][0]

    candid_frames = sorted(glob.glob(candid_path + '/*.json'))

    data_frames = []
    for frame in candid_frames:
        tmp = pd.read_json(frame)
        tmp.insert(
            0, "center_point",
            tmp.apply(lambda row: dict(
                {
                    "x": np.mean([row.bounding_box["x0"], row.bounding_box["x1"]]),
                    "y": np.mean([row.bounding_box["y0"], row.bounding_box["y1"]]),
                    "z": np.mean([row.bounding_box["z0"], row.bounding_box["z1"]])
                }),
                axis=1))
        data_frames.append(tmp)

    dfs = []
    for it in data_frames:
        tmp = frameWrapper(it, dim=3)
        dfs.append(tmp)

    df = pd.concat(dfs)
    df['frame'] = df.index
    df['frame'] = df['frame'].diff().ne(1).cumsum() - 1
    df['data_id'] = '-'.join(candid_path.split('/')[3:-1])
    df = df.reset_index(drop=True)

    grouped = df.groupby(df.class_name)
    groups = list(grouped.groups.keys())
    obj_dfs = [grouped.get_group(grp).reset_index(drop=True) for grp in groups]
    obj_dfs = [obj_df.set_index(obj_df.frame) for obj_df in obj_dfs]
    return dfs, obj_dfs


def plot_rgbxyz_with_bounding_boxes(frame: int, rgbxyz, obj_dfs: list, save=False):
    '''Plot candidate pointcloud (frame) with labeled 3d bounding boxes and center points for each object

    Input
      frame: int id of frame in take
      images: list of rgb images
      obj_dfs: n (number of objects for task) dataframes --> object dataframes consisting of y [number of frames] rows
      save: default set parameter to save plot -> named after its type + timestamp (human readable)
    '''

    rgb = [list(np.round(matplotlib.colors.hex2color(col), 5))
           for col in colors_]

    r = rgbxyz[::3, ::3, 0].flatten()
    g = rgbxyz[::3, ::3, 1].flatten()
    b = rgbxyz[::3, ::3, 2].flatten()
    x = rgbxyz[::3, ::3, 3].flatten()
    y = rgbxyz[::3, ::3, 4].flatten()
    z = rgbxyz[::3, ::3, 5].flatten()

    # cut to interesting region
    to_cut = x > -1.5
    x = x[to_cut]
    y = y[to_cut]
    z = z[to_cut]
    r = r[to_cut]
    g = g[to_cut]
    b = b[to_cut]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, marker='.', s=0.5, c=np.array([r, g, b]).T)
    for it, obj_ in enumerate(obj_dfs):
        label_ = obj_.class_name.unique()[0]
        if not obj_[obj_.frame == frame].empty:
            ax.plot3D(obj_.loc[frame:frame, "x"].values,
                      obj_.loc[frame:frame, "y"].values,
                      obj_.loc[frame:frame, "z"].values,
                      'o-',
                      color=rgb[it],
                      ms=10,
                      label=label_)

        try:
            specd = obj_[obj_.frame == frame]
            bbox = specd["bounding_box"].values[0]
            if bbox["x0"] != bbox["x1"]:
                pt1 = [
                    bbox["x0"], bbox["y0"], bbox["z0"]
                ]
                pt2 = [
                    bbox["x1"], bbox["y1"], bbox["z1"]
                ]

                pts, kts = get_3d_bounding_box(pt1, pt2)

                for ite, pt in enumerate(pts):
                    ax.scatter(pt[0], pt[1], pt[2], color=rgb[it])

                for ite, kt in enumerate(kts):
                    ax.plot3D(kt[:, 0], kt[:, 1], kt[:, 2],
                              'o-', color=rgb[it])
        except:
            pass

    set_axes_equal(ax)
    ax.legend(bbox_to_anchor=(0.2, 0.6))
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        date_str = '_'.join(map(str, time.localtime()[:5]))
        fig.savefig('./'+"pcl_"+date_str+'.png', dpi=300, bbox_inches="tight")
