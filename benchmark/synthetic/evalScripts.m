%% Run this script in matlab after running benchmark/synthetic/evalScripts.lua
classes = {'aero','car','chair'};
netNameSuffixes = {'voxels', 'mask_nIm5', 'fuse_nIm5', 'depth_nIm5', 'fuse_nIm5_noise',  'depth_nIm5_noise'};
addpath('./matUtils');

numTrainIters = {10000};
perfs = zeros(length(classes), length(netNameSuffixes), length(numTrainIters));
threshesOpt = zeros(length(classes), length(netNameSuffixes), length(numTrainIters));
valThreshes = 0.01:0.01:0.99;

for cx = 1:length(classes)
    for nx = 1:length(netNameSuffixes)
        for ix = 1:length(numTrainIters)
            expName = [classes{cx} '_' netNameSuffixes{nx} '_' num2str(numTrainIters{ix})];
            predVolDirVal = fullfile('../../cachedir/resultsDir/shapenet/', [expName '_val']);
            predVolDirTest = fullfile('../../cachedir/resultsDir/shapenet/', [expName '_test']);
            [~, threshOpt] = iouBenchmark(predVolDirVal,  valThreshes);
            perf = iouBenchmark(predVolDirTest,  [threshOpt]);
            perfs(cx, nx, ix) = perf;
            threshesOpt(cx, nx, ix) = threshOpt;
        end
    end
end

perfs = (max(perfs,[],3));
disp(perfs);
mkdirOptional('../../cachedir/resultsDir/tables/');
save('../../cachedir/resultsDir/tables/snet.mat','perfs','netNameSuffixes','threshesOpt');