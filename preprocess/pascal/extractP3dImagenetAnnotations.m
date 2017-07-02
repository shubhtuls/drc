function [pos] = extractP3dImagenetAnnotations(cls)
startup;
addpath('matUtils');

annoDir = fullfile(pascal3dDir,'Annotations',[cls '_imagenet']);
posNames = getFileNamesFromDirectory(annoDir,'types',{'.mat'});
for i=1:length(posNames)
    posNames{i} = posNames{i}(1:end-4);
end
pos = [];

%% Create one entry per bounding box in the pos array
numpos = 0;
for j = 1:length(posNames)
    
    load(fullfile(annoDir,posNames{j}));
    for k=1:length(record.objects)
        if(strcmp(record.objects(k).class,cls) && ~isempty(record.objects(k).viewpoint) && ~record.objects(k).truncated && ~record.objects(k).occluded)
            bbox   = round(record.objects(k).bbox);
            imsize = record.imgsize(1:2);
            bbox(1:2) = max(bbox(1:2),1);
            bbox(3) = min(bbox(3),imsize(1));
            bbox(4) = min(bbox(4),imsize(2));
            if(strcmp(posNames{j}, 'n03790512_11192'))
                bbox(1) = 1;bbox(2) = 1;
            end
            numpos = numpos + 1;
            pos(numpos).imsize = imsize;
            pos(numpos).voc_image_id = posNames{j};
            pos(numpos).voc_rec_id = k;
            pos(numpos).bbox   = bbox;
            pos(numpos).mask = [];
            pos(numpos).class       = cls;
            pos(numpos).rot = [];
            pos(numpos).euler = [];
            pos(numpos).occluded = record.objects(k).occluded;
            pos(numpos).truncated = record.objects(k).truncated;

            %% Getting camera
            objectInd = k;
            viewpoint = record.objects(objectInd).viewpoint;
            [rot,euler]=viewpointToRots(viewpoint);
            pos(numpos).rot=rot;
            pos(numpos).euler=euler;

            pos(numpos).objectInd = objectInd;
            pos(numpos).dataset = 'imagenet';
            pos(numpos).subtype = record.objects(k).cad_index;
        end
    end
end


end


function [R,euler] = viewpointToRots(vp)
    if(~isfield(vp,'azimuth'))
        vp.azimuth = vp.azimuth_coarse;
    end
    
    if(~isfield(vp,'elevation'))
        vp.elevation = vp.elevation_coarse;
    end
    
    if(~isfield(vp,'theta'))
        vp.theta = 0;
    end
    euler = [vp.azimuth vp.elevation vp.theta]' .* pi/180;
    R = angle2dcm(pi-euler(1), -pi/2 + euler(2), euler(3),'ZXZ'); %empirically computed
    euler = euler([3 2 1]);
end
