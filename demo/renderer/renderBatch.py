import bpy
import os
import sys

modelpath = sys.argv[6]
pngpath = sys.argv[7]

bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = 0.4
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

bpy.ops.import_scene.obj(filepath = modelpath)
bpy.data.scenes['Scene'].render.filepath = pngpath
print('rendering')
bpy.ops.render.render( write_still=True)
print('rendered')
sys.exit(0) # exit python and blender
print('exited')
