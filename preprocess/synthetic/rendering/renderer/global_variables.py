#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
g_render4cnn_root_folder = os.path.dirname(os.path.abspath(__file__))
# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
g_blender_executable_path = '/home/eecs/shubhtuls/Downloads/blender-2.71/blender' #!! MODIFY if necessary
g_shapenet_root_folder = '/data1/shubhtuls/cachedir/Datasets/shapeNetCoreV1'
g_blank_blend_file_path = os.path.join(g_render4cnn_root_folder, 'blank.blend')
g_blender_python_script = os.path.join(g_render4cnn_root_folder, 'render_model_views.py')