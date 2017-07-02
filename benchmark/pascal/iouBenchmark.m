function [iouVal, threshVal] = iouBenchmark(predVolDir,  threshes)
    addpath('../synthetic/matUtils');
    nThreshes = length(threshes);
    predFiles = getFileNamesFromDirectory(predVolDir,'types',{'.mat'});
    nModels = length(predFiles);
    ious = zeros(nModels,nThreshes);
    for p = 1:nModels
        predModel = load(fullfile(predVolDir,predFiles{p}));
        for nt=1:nThreshes
            thresh = threshes(nt);
            predVol = double(predModel.volume > thresh);

            gtVol = double(predModel.gtVol > 0.5);
            intersection = sum(predVol(:).*gtVol(:));
            total = sum(predVol(:)) + sum(gtVol(:));
            ious(p,nt) = intersection/(total-intersection);

            %gtPoints = occupiedPoints(predModel.gtVol, 0.5);
            %predPoints = occupiedPoints(predModel.volume, thresh);
            %figure(1);
            %plot3(predPoints(:,1),predPoints(:,2),predPoints(:,3),'g.');
            %axis equal;xlabel('x');ylabel('y');zlabel('z');
            %figure(2);
            %plot3(gtPoints(:,1),gtPoints(:,2),gtPoints(:,3),'r.');
            %axis equal;xlabel('x');ylabel('y');zlabel('z');
            %title(errs(p))
            %pause();
            %%close all;
        end
    end
    mIous = mean(ious);
    [iouVal,threshInd] = max(mIous);
    threshVal = threshes(threshInd);
    %disp(mIous);
    %disp(iouVal);
    %disp(threshVal);
end

function [points] = occupiedPoints(volume, thresh)
    occPoints = volume >= thresh;
    [Xs,Ys,Zs] = ind2sub(size(volume),find(occPoints));
    Xs = (Xs)/(size(volume,1))-0.5;
    Ys = (Ys)/(size(volume,2))-0.5;
    Zs = (Zs)/(size(volume,3))-0.5;
    points = [Xs Ys Zs];
end
