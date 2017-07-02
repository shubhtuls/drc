import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

add_path(os.getcwd())

def params():
    config = {}
    config['basedir'] = os.getcwd()
    config['shapenetDir'] = '/data1/shubhtuls/cachedir/Datasets/shapeNetCoreV1/'
    config['renderPrecomputeDir'] = osp.join(config['basedir'],'..','..','..','cachedir','blenderRenderPreprocess')
    return config
