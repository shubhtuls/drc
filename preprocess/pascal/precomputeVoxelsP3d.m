function [] = precomputeVoxelsP3d()
startup;
addpath('matUtils');
addpath(fullfile(basedir, '..', 'synthetic', 'voxelization'));

%loop over classes
classes = {'car','aeroplane','chair'};
gridSize=32;

for s = 1:length(classes)
    fprintf('class : %d/%d\n\n',s,length(classes));
    class = classes{s};
    modelsAll = load(fullfile(pascal3dDir, 'CAD', class));
    modelsAll = getfield(modelsAll,class);
    voxelsDir = fullfile(cachedir,'pascal','modelVoxels',class);
    mkdirOptional(voxelsDir);

    nModels = length(modelsAll);
    for i = 1:nModels
        voxelsFile = fullfile(voxelsDir,[num2str(i) '.mat']);

        %% Compute Voxels
        faces = modelsAll(i).faces;
        vertices = modelsAll(i).vertices;
        vertices = vertices(:,[2 1 3]);
        vertices(:,1) = -vertices(:,1);

        %keyboard;
        FV = struct();
        FV.faces = faces;
        FV.vertices = (gridSize)*(vertices+0.5) + 0.5;

        Volume=polygon2voxel(FV,gridSize,'none',false);
        savefunc(voxelsFile, Volume);
    end
end

end

function savefunc(voxelsFile, Volume)
save(voxelsFile,'Volume');
end
