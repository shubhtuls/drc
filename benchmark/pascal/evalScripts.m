classes = {'aeroplane','car','chair'};
netNameSuffixes = {'cs3d', 'p3d_20000', 'SNet_10000', 'SNet_25000', 'p3dSNetCombined_20000'};

perfs = zeros(length(classes), length(netNameSuffixes));
threshesOpt = zeros(length(classes), length(netNameSuffixes));
valThreshes = 0.01:0.01:0.99;

for cx = 1:length(classes)
    for nx = 1:length(netNameSuffixes)
        expName = [netNameSuffixes{nx} '_' classes{cx}];
        disp(expName);
        predVolDirVal = fullfile('../../cachedir/resultsDir/pascal/', [expName '_train']);
        predVolDirTest = fullfile('../../cachedir/resultsDir/pascal/', [expName '_val']);
        [~, threshOpt] = iouBenchmark(predVolDirVal,  valThreshes);
        perf = iouBenchmark(predVolDirTest,  [threshOpt]);
        perfs(cx, nx) = perf;
        threshesOpt(cx, nx) = threshOpt;
    end
end

disp(perfs');
addpath('./matUtils');
mkdirOptional('../../cachedir/resultsDir/tables/');
save('../../cachedir/resultsDir/tables/pascal.mat','perfs','netNameSuffixes','threshesOpt');