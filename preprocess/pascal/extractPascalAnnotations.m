function [pos] = extractPascalAnnotations(className)
    startup;
    segKps = load(fullfile(cachedir, 'pascal', 'segkps', className));
    pos = [];
    annoDir = fullfile(pascal3dDir,'Annotations',[className '_pascal']);
    for numpos = 1:length(segKps.segmentations.voc_image_id)
        voc_image_id = segKps.segmentations.voc_image_id{numpos};
        k = segKps.segmentations.voc_rec_id(numpos);
        load(fullfile(annoDir,voc_image_id));
        bbox  = round(record.objects(k).bbox);
        pos(numpos).imsize = record.imgsize(1:2);
        pos(numpos).voc_image_id = voc_image_id;
        pos(numpos).voc_rec_id = k;
        pos(numpos).bbox   = bbox;
        pos(numpos).view    = '';
        pos(numpos).poly_x      = segKps.segmentations.poly_x{numpos};
        pos(numpos).poly_y      = segKps.segmentations.poly_y{numpos};
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
        pos(numpos).dataset = 'pascal';
        pos(numpos).subtype = record.objects(k).cad_index;
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
