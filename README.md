# Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency 
Shubham Tulsiani, Tinghui Zhou, Alexei A. Efros, Jitendra Malik. In CVPR, 2017.
[Project Page](https://shubhtuls.github.io/drc/)

![Teaser Image](https://shubhtuls.github.io/drc/resources/images/formulation.png)

## Demo and Pre-trained Models

Please check out the [interactive notebook](../demo/demo.ipynb) which shows reconstructions using the learned models. You'll need to - 
- Install a working implementation of torch and itorch.
- Download the pre-trained models for [Pascal3D (490MB)](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/drc/snapshots/pascalModels.tar.gz) and [ShapeNet (250MB)](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/drc/snapshots/shapenetModels.tar.gz). Extract the pretrained models to 'cachedir/snapshots/{pascal,shapenet}/'
- Edit the path to the blender executable in the demo script.

## Loss Function Compilation

To use our proposed loss function for training, we need to compile the C implementation so it can be used in Torch.
```
cd drcLoss
luarocks make rpsem-alpha-1.rockspec
```

## Training and Evaluating
For training your own models and evaluating those, or for reproducing the main experiments in the paper, please see the detailed README files for [PASCAL3D](docs/pascal.md) or ShapeNet(docs/snet.md).

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{drcTulsiani17,
  title={Multi-view Supervision for Single-view Reconstruction
  via Differentiable Ray Consistency},
  author = {Shubham Tulsiani
  and Tinghui Zhou
  and Alexei A. Efros
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2017}
}
```