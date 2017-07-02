# Pascal VOC Experiments

## Demo and Pre-trained Models

Please check out the [interactive notebook](../demo/demo.ipynb) which shows reconstructions using the learned models. You'll need to - 
- Install a working implementation of torch and itorch.
- Download the pre-trained models [here (490MB)](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/drc/snapshots/pascalModels.tar.gz) and extract them to cachedir/snapshots/pascal/
- Edit the path to the blender executable in the demo script.

## Pre-processing
Download the annotations for segmentation masks for objects in PASCAL3D from [here (80MB)](https://people.eecs.berkeley.edu/~shubhtuls/cachedir/drc/pascalData.tar.gz) - the segmentations for instances in PASCAL VOC are from annotations by Hariharan et. al. and the ones for ImageNet instances were computed using a pre-trained 'Iterative Instance Segmentation (IIS)' model. Extract the contents of this tar file in 'cachedir/pascal/'.

For training, we need to extract orthographic cameras from the PASCAL3D dataset. For evaluation, we need the ground-truth PASCAL3D models. To compute these, first modify the dataset paths in the [startup file](../preprocess/pascal/startup.m) and then run in matlab from within the directory 'preprocess/pascal':
```
% This might take an hour
classes = {'aeroplane','car','chair'};
for c=1:3
    computeOrthographicCam(classes{c});
    computeOrthographicCamImagenet(classes{c});
end
precomputeVoxelsP3d;
```

## Evaluation
To evaluate the trained (or downloaded) models, run
```
#Evaluating all trained models.
cd benchmark
#first predict and save
gpu=1 th pascal/evalScripts.lua | bash
#evaluate using matlab
cd pascal; matlab -nodesktop -nosplash
>> evalScripts
```
Note that the baseline trained on only ShapeNet using realistic renderings actually performs worse on real data when trained for much longer. So, while we report performance after 20,000 iterations for other models, we evaluate this baseline at 10,000 iterations (the evaluation script also evaluates this baseline at 25,000 iterations but the performance reported in the paper corresponds to the earlier snapshot).

## Training
To train the models and the baselines reported in the paper, we first need to download SUN2012 dataset as we'll use these images as random background textures for synthetic data. We used the SUN images in PASCAL format downloaded from [here](http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz). After downloading these, please edit the paths in the top line of training files [here](../experiments/pascal/pascal.lua) and [here](../experiments/pascal/pascalEncoderFinetune.lua). We initialize using a pretrained ResNet-18 model which can be downloaded from [here](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained). Place the ResNet model in 'cachedir/initModels/resnet-18.t7'. We also need the voxelized ShapeNet models and rendered ShapeNet images for these experiments. Please see the [ShapeNet experiments documenation](snet.md) for instructions to compute these. After all the preprocessing is complete, we can train the models.

- Baseline using only Synthetic data with full 3D supervision.
```
cd experiments; disp=0 gpu=1 useResNet=1 batchSizeShapenet=16 shapenetWeight=1 pascalWeight=0 name=SNet th pascal/pascal.lua
```

- DRC Model using only PASCAL3D (mask + pose annotations).
```
cd experiments; disp=0 gpu=1 useResNet=1 batchSizePascal=16 shapenetWeight=0 bgWtPascal=0.2 pascalWeight=1 name=p3d th pascal/pascal.lua
```

- DRC Joint Model using Synthetic Data and PASCAL3D. We finetune the synthetic data model from an intermediate snapshot.
```
cd experiments; disp=0 gpu=1 pretrainNet=SNet numPreTrainIter=10000 batchSizeShapenet=16 batchSizePascal=8 shapenetWeight=1 bgWtPascal=0.2 pascalWeight=1 name=p3dSNetCombined th pascal/pascalEncoderFinetune.lua
```
