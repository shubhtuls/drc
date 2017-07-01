function [iouVal, threshVal] = iouBenchmark(predVolDir,  threshes)
    addpath('./matUtils');
    nThreshes = length(threshes);
    predFiles = getFileNamesFromDirectory(predVolDir,'types',{'.mat'});
    nModels = length(predFiles);
    ious = zeros(nModels,nThreshes);
    for p = 1:nModels
        if(mod(p,200)==0)
            disp(p);
        end
        predModel = load(fullfile(predVolDir,predFiles{p}));
        for nt=1:nThreshes
            thresh = threshes(nt);
            predVol = double(predModel.volume > thresh);

            gtVol = double(predModel.gtVol > 0.5);
            intersection = sum(predVol(:).*gtVol(:));
            total = sum(predVol(:)) + sum(gtVol(:));
            ious(p,nt) = intersection/(total-intersection);
        end
    end
    mIous = mean(ious);
    [iouVal,threshInd] = max(mIous);
    threshVal = threshes(threshInd);
end