# ShapeNet Experiments

## Demo and Pre-trained Models

Please check out the [interactive notebook](../demo/demo.ipynb) which shows reconstructions using the learned models. You'll need to - 
- Install a working implementation of torch and itorch.
- Download the pre-trained ShapeNet models [here (250MB)](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/drc/snapshots/shapenetModels.tar.gz) and extract them to cachedir/snapshots/
- Edit the path to the blender executable in the demo script.

## Training
To train your own models and baselines on ShapeNet data, say for the class 'chair', run
```
#Training all models and baselines for 'chairs'
cd experiments
class=chair gpu=1 th synthetic/experimentScripts.lua | bash
```
Note that before running this, you'll need to render the ShapeNet images. To train the baselines, we also need to precompute ground-truth voxels and the depth fusion based voxelizations. You can modify the [training script](../experiments/synthetic/experimentScripts.lua) if you want to train only some of the models/baselines.

## Evaluation
To evaluate the trained (or downloaded) models, run
```
#Evaluating all trained models.
cd benchmark
#first predict and save
gpu=1 th synthetic/evalScripts.lua | bash
#evaluate using matlab
cd synthetic; matlab -nodesktop -nosplash
>> evalScripts
```
Note that before running this, you'll need to render the ShapeNet images and precompute ground-truth voxels. This script will evaluate all models for all classes but if you need only a subset evaluated, please modify the [evaluation script](../benchmark/synthetic/evalScripts.lua).

## Rendering
To render RGB and depth images for ShapeNet models, specify the ShapeNetV1 folder [here](../preprocess/synthetic/rendering/startup.py) and the path to blender [here](../preprocess/synthetic/rendering/renderer/global_variables.py). Then, run
```
#Rendering chairs, cars and aeroplanes (takes about a day)
cd preprocess/synthetic/rendering
python renderPreprocessShapenet.py
```
Some experiments also need a noisy depth. If you need to train the models with noisy data, after rendering the images as above, the noisy images can be saved as follows.
```
cd preprocess
synset=3001627 th synthetic/noisyDepth.lua #chairs
synset=2958343 th synthetic/noisyDepth.lua #cars
synset=2691156 th synthetic/noisyDepth.lua #aero
```

## Computing Voxelizations
For evaluation and training the 3D-supervised baseline, we need to compute the groun-truth 3D voxelizations. First, modify the path to ShapeNetV1 [here](../preprocess/synthetic/voxelization/startup.m) and then run
```
#Computing Gt Voxelizations
cd preprocess/synthetic/voxelization
matlab -nodesktop -nosplash
>> precomputeVoxels
```

## Pre-processing for Fusion Baseline
To compute the fused volumes required to train the fusion baselines, run
```
cd preprocess;
#Sample fusion preprocessing script for chairs using clean depth.
#You'll need to repeat this with and without noise for all classes.
useNoise=0 synset=3001627 th synthetic/fusion/shapenetFusion.lua
```
