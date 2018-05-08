%% Calculate GIST features
addpath('gistdescriptor');

% Filepath need to be changed accordingly
load('../../data/sun_attributes/images.mat')
image_folder = '../../data/sun_images/';
feature_folder = '../../features/gist/';

% GIST Parameters
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

for i = 1:length(images)
    fprintf('Image : %i/%i\n', i, length(images));
    
    tic
    image_path = [image_folder, images{i}];
    img = imread(image_path);
    
    % Computing gist requires 1) prefilter image, 2) filter image and collect
    % output energies
    [gist, ~] = LMgist(img, '', param);
    toc
    
    feature_path = [feature_folder, sprintf('%05i.mat',(i-1))];
    save(feature_path, 'gist');
end