import numpy as np
import sys, glob
from global_variables import *
import os
import subprocess
import pickle
import scipy.io
import scipy.misc

import OpenEXR, Imath
import png


def saveUint16(z, path):
    # Use pypng to write zgray as a grayscale PNG.
    with open(path, 'wb') as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
        zgray2list = z.tolist()
        writer.write(f, zgray2list)

def depthToint16(dMap, minVal=0, maxVal=10):
    dMap[dMap>maxVal] = maxVal
    dMap = ((dMap-minVal)*(pow(2,16)-1)/(maxVal-minVal)).astype(np.uint16)
    return dMap

def loadDepth(dFile, minVal=0, maxVal=10):
    dMap = scipy.misc.imread(dFile)
    dMap = dMap.astype(np.float32)
    dMap = dMap*(maxVal-minVal)/(pow(2,16)-1) + minVal
    return dMap

def normalize(ch):
    return ch/ch.max()

def path2label(path):
    parts = os.path.basename(path).split('_')
    azimuth = int(parts[2][1:]) % 360
    elevation = int(parts[3][1:]) % 360
    tilt = int(parts[4][1:]) % 360
    return (azimuth, elevation, tilt)

def fromstr(s,width, height):
    mat = np.fromstring(s, dtype=np.float32)
    mat = mat.reshape(height,width)
    return mat

class Renderer:
    def __init__(self, modelNames):
        self.models = modelNames        
        self.modelIndex = 0
        
    def setModelInd(self, i):
        self.modelIndex = i
        
    def renderViews(self, poseSamples, outDir):
        dist_low, dist_high = 2,2
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        viewFile = os.path.join(outDir, 'view.txt')
        viewFout = open(viewFile,'w')
        
        ## render from blender
        for theta in poseSamples:
            eulers = theta.getVar('rot')
            az = eulers[0]
            el = eulers[1]
            tilt = eulers[2]
            dist = (dist_high-dist_low)*np.random.random() + dist_low
            viewFout.write(' '.join(map(str,[az, el, tilt, dist])))
            viewFout.write('\n')
        viewFout.close()
        prefix = 'init'
        render_cmd = '%s %s --background --python %s -- %s %s %s %s' % (g_blender_executable_path, g_blank_blend_file_path, g_blender_python_script, self.models[self.modelIndex], viewFile,prefix, outDir)
        render_cmd_debug = '%s %s --python %s -- %s %s %s %s' % (g_blender_executable_path, g_blank_blend_file_path, g_blender_python_script, self.models[self.modelIndex], viewFile,prefix, outDir)
        #print render_cmd
        os.system(render_cmd)
        
        ## augment normals
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        viewParams = [[float(x) for x in line.strip().split(' ')] for line in open(viewFile).readlines()]
        
        for ix,view in enumerate(viewParams):
            azimuth_deg = view[0]
            elevation_deg = view[1]
            theta_deg = -1 * view[2] # ** multiply by -1 to match pascal3d annotations **
            rho = view[3]
            imFile = '%s/%s_a%03d_e%03d_t%03d_d%03d.exr' % (outDir, prefix, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
            KFile = '%s/K_%s_a%03d_e%03d_t%03d_d%03d.txt' % (outDir, prefix, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
            
            exrimage = OpenEXR.InputFile(imFile)
            dw = exrimage.header()['dataWindow']
            (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            (a, b, g, r, dmap) = [fromstr(s, width, height) for s in exrimage.channels('ABGRZ', pt)]
            mask = dmap<1000
            dmap[np.logical_not(mask)] = 1000
            K = np.loadtxt(KFile)
            
            dFile = os.path.join(outDir, 'depth_{}.png'.format(ix))
            rgbFile = os.path.join(outDir, 'render_{}.png'.format(ix))
            im_rgb = np.dstack([normalize(np.dstack([r,g,b])), a])
            
            saveUint16(depthToint16(dmap), dFile)
            scipy.misc.imsave(rgbFile, im_rgb)
            
            exrimage.close()
            
        ## remove unwanted files
        for ix,view in enumerate(viewParams):
            azimuth_deg = view[0]
            elevation_deg = view[1]
            theta_deg = -1 * view[2] # ** multiply by -1 to match pascal3d annotations **
            rho = view[3]
            CamFile = '%s/Cam_%s_a%03d_e%03d_t%03d_d%03d.pkl' % (outDir, prefix, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
            with open(CamFile, 'rb') as handle:
                cam = pickle.load(handle)
            CamFileNew = os.path.join(outDir, 'camera_{}.mat'.format(ix))
            scipy.io.savemat(CamFileNew, cam)
            
        ims = sorted(glob.glob(outDir + '/{:s}*.exr'.format(prefix)))
        Ks = sorted(glob.glob(outDir + '/K_{:s}*.txt'.format(prefix)))
        Cams = sorted(glob.glob(outDir + '/Cam_{:s}*.pkl'.format(prefix)))
        
        rmims = [os.remove(f) for f in ims]
        rmfs = [os.remove(f) for f in Ks]
        rmcs = [os.remove(f) for f in Cams]
        
    def visRenderedViews(self,outDir,nViews=0):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        renders = sorted(glob.glob(outDir + '/render_*.png'))
        if (nViews > 0) and (nViews < len(renders)):
            renders = [renders[ix] for ix in range(nViews)]
        
        for render in renders:
            print render
            rgbIm = scipy.misc.imread(render)
            dMap = loadDepth(render.replace('render_','depth_'))
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.imshow(rgbIm)
            dMap[dMap>=10] = np.nan
            plt.subplot(122)
            plt.imshow(dMap)
            print(np.nanmax(dMap),np.nanmin(dMap))
            plt.show()
            
    def reInit(self, modelNames):
        self.models = modelNames
        self.modelIndex = 0