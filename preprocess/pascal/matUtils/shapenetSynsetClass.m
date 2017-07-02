function [synset] = shapenetClassSynset(class)
%SHAPENETCLASSSYNSET Summary of this function goes here
%   Detailed explanation goes here

synsetNamePairs = {'02691156', 'aeroplane';
     '02834778', 'bicycle';
     '02858304', 'boat';
     '02876657', 'bottle';
     '02924116', 'bus';
     '02958343', 'car';
     '03001627', 'chair';
     '04379243', 'diningtable';
     '03790512', 'motorbike';
     '04256520', 'sofa';
     '04468005', 'train';
     '03211117', 'tvmonitor'};
synset = synsetNamePairs(ismember(synsetNamePairs(:,2),class),1);
if(~isempty(synset))
    synset = synset{1};
end


end
