function [] = precomputeVoxels()
globals;
shapenetDir = shapenetDir;
cachedir = cachedir;
params = params;
%loop over sysnets
synsetNames = {'03001627','02691156','02958343'}; %chair, aero, car
gridSize=32;

for s = 1:length(synsetNames)
    fprintf('synset : %d/%d\n\n',s,length(synsetNames));
    synset = synsetNames{s};
    modelsDir = fullfile(shapenetDir,synset);
    voxelsDir = fullfile(cachedir,'shapenet','modelVoxels',synset);
    mkdirOptional(voxelsDir);
    modelNames = getFileNamesFromDirectory(modelsDir,'types',{''});
    modelNames = modelNames(3:end); %fist two are '.' and '..'

    nModels = length(modelNames);
    %nModels = 10;%for debugging we'll use about 20 instances only
    pBar = TimedProgressBar( nModels, 40, 'Time Remaining : ', ' Percentage Completion ', 'Tsdf Extraction Completed.');

    parfor i = 1:nModels
        %% Check if computation is really needed
        modelFile = getFileNamesFromDirectory(fullfile(modelsDir,modelNames{i}),'types',{'.obj'});
        voxelsFile = fullfile(voxelsDir,[modelNames{i} '.mat']);
        if(exist(voxelsFile,'file'))
            continue;
        end
        if(isempty(modelFile))
            continue;
        end

        %% Read model
        modelFile = fullfile(modelsDir,modelNames{i},modelFile{1});
        [Shape] = parseObjMesh(modelFile);
        %ignore bad meshes
        if(isempty(Shape.vertexPoss) || isempty(Shape.faceVIds))
            continue;
        end
        %% Compute Voxels
        faces = Shape.faceVIds';
        vertices = Shape.vertexPoss';
        %keyboard;
        FV = struct();
        FV.faces = faces;
        FV.vertices = (gridSize)*(vertices+0.5) + 0.5;

        Volume=polygon2voxel(FV,gridSize,'none',false);
        savefunc(voxelsFile, Volume);
        pBar.progress();
    end
    pBar.stop();
end

end

function savefunc(voxelsFile, Volume)
save(voxelsFile,'Volume');
end
