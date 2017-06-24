#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
render_model_views.py
brief:
	render projections of a 3D model from viewpoints specified by an input parameter file

author: hao su, charles r. qi, yangyan li
changed: abhishek kar, shubham tulsiani
'''

import os
import bpy
import sys
import math
import random
import time
import numpy as np
import pickle
from mathutils import Matrix

## rendering parameters
# render_model_views
g_syn_light_num_lowbound = 0
g_syn_light_num_highbound = 0
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 12
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = -90
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 2
g_syn_light_energy_std = 1
g_syn_light_environment_energy_lowbound = 2
g_syn_light_environment_energy_highbound = 2.01

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = np.array(
        [[alpha_u, skew,    u_0],
        [    0  ,  alpha_v, v_0],
        [    0  ,    0,      1 ]])
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_4x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
         ))
    
    motionMat = np.zeros((4,4))
    motionMat[3][3] = 1
    motionMat[0:3,0:4] = np.asarray(RT)
    return motionMat

def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)


# Preferences
#bpy.context.user_preferences.system.compute_device_type = "CUDA"
#bpy.context.user_preferences.system.compute_device = "CUDA_MULTI_2"
# bpy.context.scene.cycles.device = 'GPU'

# Rendering parameters
im_x, im_y = 224, 224
print("starting")
bpy.context.scene.render.resolution_percentage = 50
bpy.context.scene.render.resolution_x = im_x*(100/bpy.context.scene.render.resolution_percentage)
bpy.context.scene.render.resolution_y = im_y*(100/bpy.context.scene.render.resolution_percentage)
bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
#bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.use_zbuffer = True
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 8

# bpy.context.scene.render.subsurface_scattering.use = True
# bpy.context.scene.render.use_shadows = False
# bpy.context.scene.render.use_raytrace = False

###### Camera settings ######
camObj = bpy.data.objects['Camera']
camObj.data.lens = 60 # 60 mm focal length
camObj.data.sensor_height = 32.0
camObj.data.sensor_width = float(camObj.data.sensor_height)/im_y*im_x
# camObj.data.type = 'ORTHO'
# camObj.data.ortho_scale = 1
# camObj.data.lens_unit = 'FOV'
# camObj.data.angle = 0.2

###### Compositing node ######
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
for n in tree.nodes:
    tree.nodes.remove(n)
rl = tree.nodes.new(type="CompositorNodeRLayers")
composite = tree.nodes.new(type = "CompositorNodeComposite")
composite.location = 200,0
links.new(rl.outputs['Image'],composite.inputs['Image'])
links.new(rl.outputs['Z'],composite.inputs['Z'])

###### Load rendering light parameters ######
#BASE_DIR = os.getcwd()
#sys.path.insert(0,BASE_DIR)

#from global_variables import *
light_num_lowbound = g_syn_light_num_lowbound
light_num_highbound = g_syn_light_num_highbound
light_dist_lowbound = g_syn_light_dist_lowbound
light_dist_highbound = g_syn_light_dist_highbound

###### Input parameters ######
shape_file = sys.argv[-4]
shape_view_params_file = sys.argv[-3]
out_prefix = sys.argv[-2]
syn_images_folder = sys.argv[-1]

if not os.path.exists(syn_images_folder):
    os.mkdir(syn_images_folder)

view_params = [[float(x) for x in line.strip().split(' ')] for line in open(shape_view_params_file).readlines()]

if not os.path.exists(syn_images_folder):
    os.makedirs(syn_images_folder)

bpy.ops.import_scene.obj(filepath=shape_file)
#print(shape_view_params_file)
#print("loaded shape")
#print(view_params)

###### Set lights ######
bpy.data.objects['Lamp'].data.energy = 0
bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True # remove default light
bpy.ops.object.delete()

for param in view_params:
    azimuth_deg = param[0]
    elevation_deg = param[1]
    theta_deg = -1 * param[2] # ** multiply by -1 to match pascal3d annotations **
    rho = param[3]

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # set environment lighting
    #bpy.context.space_data.context = 'WORLD'
    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
    bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

    # set point lights
    for i in range(random.randint(light_num_lowbound,light_num_highbound)):
        light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
        light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
        light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
        lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
        bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)

    # set camera location
    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    camObj.location[0] = cx
    camObj.location[1] = cy
    camObj.location[2] = cz
    camObj.rotation_mode = 'QUATERNION'
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]
    
    # render and write
    syn_image_file = './%s_a%03d_e%03d_t%03d_d%03d.exr' % (out_prefix, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
    K_file = './K_%s_a%03d_e%03d_t%03d_d%03d.txt' % (out_prefix, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
    CamFile = './Cam_%s_a%03d_e%03d_t%03d_d%03d.pkl' % (out_prefix, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
    bpy.context.scene.render.filepath = os.path.join(syn_images_folder, syn_image_file)
    bpy.ops.render.render(write_still=True)
    
    # get camera matrix and write to file
    print(bpy.data.objects['Camera'].data)
    K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
    print(K)
    #print(bpy.data.objects['Camera'].matrix_world.inverted())
    print(bpy.data.objects['Camera'].matrix_world)
    extrinsic = get_4x4_RT_matrix_from_blender(bpy.data.objects['Camera'])
    np.savetxt(os.path.join(syn_images_folder, K_file), K)
    
    # save camera coordinates
    #np.savetxt(os.path.join(syn_images_folder, CamFile), np.array([cx,cy,cz]))
    pickle.dump( {'quat':q, 'pos':np.array([cx,cy,cz]), 'K':K, 'extrinsic':extrinsic}, open(os.path.join(syn_images_folder, CamFile), "wb" ) , protocol=2)
    #scipy.io.savemat(os.path.join(syn_images_folder, CamFile),{'quat':q, 'pos':np.array([cx,cy,cz])})
    
    