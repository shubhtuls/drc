function [] = computeOrthographicCam(className)
    startup;
    pascalInstances = extractPascalAnnotations(className);

    addpath('matUtils');
    addpath(fullfile(basedir, '..', 'synthetic', 'voxelization'));

    valFile = fullfile(cachedir, 'pascal', 'val.txt');
    valId = fopen(valFile);
    sNetToP3d = [0 0 -1; -1 0 0 ; 0 1 0];
    valNames = textscan(valId,'%s');valNames = valNames{1};fclose(valId);

    trainFile = fullfile(cachedir, 'pascal', 'train.txt');
    trainId = fopen(trainFile);
    trainNames = textscan(trainId,'%s');trainNames = trainNames{1};fclose(trainId);

    cameraStatesDir = fullfile(cachedir,'pascal','camera',className);
    synset = shapenetClassSynset(className);
    synsetDir = fullfile(shapenetDir, synset);
    shapenetModelNames = getFileNamesFromDirectory(synsetDir,'types',{''});
    shapenetModelNames = shapenetModelNames(3:end);

    preloadedModels = {};
    nModels = 30;
    parfor i = 1:nModels
        modelFile = fullfile(synsetDir,shapenetModelNames{i},'model.obj');
        [Shape] = parseObjMesh(modelFile);
        surfaceSamples = uniform_sampling(Shape, 1000);
        surfaceSamples = surfaceSamples';
        preloadedModels{i} = surfaceSamples;
    end

    mkdirOptional(cameraStatesDir);
    mkdirOptional(fullfile(cameraStatesDir,'val'));
    mkdirOptional(fullfile(cameraStatesDir,'train'));

    costs = zeros(nModels);
    scales = zeros(nModels);
    translations = zeros(nModels,2);

    for ix=1:length(pascalInstances)
        if(pascalInstances(ix).truncated || pascalInstances(ix).occluded)
            continue;
        end
        voc_image_id = pascalInstances(ix).voc_image_id;
        im = imread(fullfile(pascalImgsDir, [voc_image_id '.jpg']));
        mask = roipoly(im, pascalInstances(ix).poly_x, pascalInstances(ix).poly_y);
        im_mask = im.*repmat(uint8(mask),[1,1,3])*0.8 + 0.2*im;
        rot = pascalInstances(ix).rot * sNetToP3d;
        bbox = pascalInstances(ix).bbox;
        for nm=1:nModels
            rotatedPoints = (rot*preloadedModels{nm}')';
            [scale, trans, cost] = solveOrthographic(bbox, rotatedPoints);
            scales(nm) = scale;
            translations(nm,:) = trans;
            costs(nm) = cost;
            transformedPts = scale*rotatedPoints + repmat([trans,0],[size(rotatedPoints,1), 1]);
        end
        [~,bestInds] = sort(costs);
        bestInds = bestInds(1:5);
        state = struct();
        
        state.cameraRot = rot;
        state.cameraScale = median(scales(bestInds));
        state.translation = median(translations(bestInds,:));
        state.bbox = bbox;
        state.bbox(3:4) = bbox(3:4) - bbox(1:2) + 1;
        state.mask = mask;
        state.im = im;
        state.subtype = pascalInstances(ix).subtype;
        state.euler = pascalInstances(ix).euler;
        mId = [pascalInstances(ix).voc_image_id, '_', num2str(pascalInstances(ix).voc_rec_id)];
        isTrain = ismember({mId(1:11)}, trainNames);
        isVal = ismember({mId(1:11)}, valNames);
        assert(isTrain || isVal);
        if(isTrain)
            evalSet = 'train';
        end
        if(isVal)
            evalSet = 'val';
        end
        save(fullfile(cameraStatesDir,evalSet,mId), '-struct', 'state');
    end
end

function [scale, trans, cost] = solveOrthographic(bbox, rotatedPoints)
    maxProj = max(rotatedPoints, [], 1);
    minProj = min(rotatedPoints, [], 1);
    A = [
         minProj(1) 1 0;
         minProj(2) 0 1;
         maxProj(1) 1 0;
         maxProj(2) 0 1;
        ];
    fit = A \ bbox';
    cost = sum((bbox' - A*fit).^2);
                
    fit = fit';
    scale = fit(1);trans = fit(2:3);
    %disp(minProj);disp(maxProj);
    %disp(bbox);disp((A*fit')');disp(scale);disp(trans);
end
        
function [samples] = uniform_sampling(Shape, numSamples)
% Perform uniform sampling of a mesh, with the numSamples samples
% samples: a matrix of dimension 3 x numSamples

t = sort(rand(1, numSamples));
faceAreas = tri_mesh_face_area(Shape);
numFaces = length(faceAreas);
for i = 2:numFaces
    faceAreas(i) = faceAreas(i-1) + faceAreas(i);
end
samples = zeros(3, numSamples);

paras = rand(2, numSamples);

faceId = 1;
for sId = 1:numSamples
    while t(sId) > faceAreas(faceId)
        faceId = faceId + 1;
    end
    faceId = min(faceId, numFaces);
    p1 = Shape.vertexPoss(:, Shape.faceVIds(1, faceId));
    p2 = Shape.vertexPoss(:, Shape.faceVIds(2, faceId));
    p3 = Shape.vertexPoss(:, Shape.faceVIds(3, faceId));
    
    r1 = paras(1, sId);
    r2 = paras(2, sId);
    t1 = 1-sqrt(r1);
    t2 = sqrt(r1)*(1-r2);
    t3 = sqrt(r1)*r2;
    samples(:, sId) = t1*p1 + t2*p2 + t3*p3;
end

end

function [faceAreas] = tri_mesh_face_area(Shape)
%
p1 = Shape.vertexPoss(:, Shape.faceVIds(1,:));
p2 = Shape.vertexPoss(:, Shape.faceVIds(2,:));
p3 = Shape.vertexPoss(:, Shape.faceVIds(3,:));

e12 = p1 - p2;
e23 = p2 - p3;
e31 = p3 - p1;

a2 = sum(e12.*e12);
b2 = sum(e23.*e23);
c2 = sum(e31.*e31);

areas = 2*(a2.*(b2+c2)+b2.*c2)-a2.*a2-b2.*b2-c2.*c2;
areas = sqrt(max(0, areas))/4;
faceAreas = areas/sum(areas);
end