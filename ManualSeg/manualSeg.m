close all;

inFolderName = './test/';

% Name change
% fnames = getAllFilesInFolder(inFolderName);
% newFolderName = strcat(inFolderName, 'New/');
% if ~exist(newFolderName, 'dir')
%     mkdir(newFolderName);
% end
% 
% for i = 1 : 120
%     newFileName = strcat('img', sprintf('%04d', i), '_', num2str(2), '.png');
%     movefile(fullfile(inFolderName, fnames{i}), fullfile(newFolderName, newFileName));    
% end
% 
% for i = 121:360
%     [path, name, extension] = fileparts(fnames{i}); 
%     temp = strsplit(name, '_');
%     extend = strsplit(temp{3}, '.');
%     newFileName = strcat(temp{1}, '_', num2str(2), '_', extend{1}, '.png');
%     movefile(fullfile(inFolderName, fnames{i}), fullfile(newFolderName, newFileName));   
% end

%% Read and compress images
% Create output folder for raw images
rawFolderName = strcat(inFolderName, 'Raw/');
if ~exist(rawFolderName, 'dir')
    mkdir(rawFolderName);

    % Specify the working folder and get all image files in it
    fnames = getAllFilesInFolder(inFolderName); % getAllFilesInFolder.m can be customed to filter out some file types

    % Format arbitrary image file names to img000N_X and move to Raw folder
    % where N = image No. and X = 0(top)/1(front)/2(side) 
    % The filename formatting should only be done once for a folder
    % Get the file extension
    [path, name, extension] = fileparts(fnames{1}); 
    for i = 1 : length(fnames)
        newFileName = strcat('img', sprintf('%04d', ceil(i / 3)), '_', num2str(mod(i, 3)), extension); % for triplet images
        % view = 1; % 0(top)/1(front)/2(side) 
        % newFileName = strcat('img', sprintf('%04d', i), '_', num2str(2), extension); % For individual set

        % Rename files and put them under "Raw" folder
        movefile(fullfile(inFolderName, fnames{i}), fullfile(rawFolderName, newFileName));     
    end
end

% Avoid having too large image file and long running time, compress the files
% Create output folder for compressed images
compressFolderName = strcat(inFolderName, 'Compressed/');
compress_size = 1024;
if ~exist(compressFolderName, 'dir')
    mkdir(compressFolderName);
    
    fnames = getAllFilesInFolder(rawFolderName);
    for i = 1 : length(fnames)
        img = imread(fullfile(rawFolderName, fnames{i}));

        % Fix a default dimension of 1024
        [h,w,d] = size(img);
        if h > w
            img = imresize(img, [compress_size NaN]);
        else
            img = imresize(img, [NaN compress_size]);
        end

        % Save compressed image
        [path, name, extension] = fileparts(fnames{i});
        imwrite(img, fullfile(compressFolderName, strcat(name, '.png'))); % @note: DON'T save as .jpg! This will lead to lossful compression and the segmentation results will be messed up! use PNG
    end

end

%% Use imageSegmenter to manually complete the segmentation
% Locate input files
compressFolderName = strcat(inFolderName, 'Compressed/');
fnames = getAllFilesInFolder(compressFolderName);

% Create output folder
segFolderName = strcat(inFolderName, 'Segmentation/');
if ~exist(segFolderName, 'dir')
    mkdir(segFolderName);
end

for i = 1 : length(fnames)
    filename = fullfile(compressFolderName, fnames{i});
    img = imread(filename);
    imageSegmenter(img);
    pause;
    imageSegmenter close;
    
    % Obtain region properties and geometric features and save to disk
    [Label,N] = bwlabel(BW, 4); % N is the number of regions
    stats = regionprops(Label, 'all'); 
    allArea = [stats.Area];
    allBoundingBox = [stats.BoundingBox];

    if N < 2
        error('Segmentation failed...');
    end

    % sort by the region area in descending order to distinguish ball and rock. can also use circular Hough transform to recognize ball 
    [~, index] = sort(allArea, 'descend'); 
    rockIdx = index(1);
    ballIdx = index(2);

    % Get the mask of each object
    rockMask = ismember(Label, rockIdx);
    ballMask = ismember(Label, ballIdx);

    rockCrop = imcrop(rockMask,allBoundingBox(4*(rockIdx-1)+1:4*(rockIdx-1)+4));
    ballCrop = imcrop(ballMask,allBoundingBox(4*(ballIdx-1)+1:4*(ballIdx-1)+4));
    
    % Get the perimeter of each object and burn it onto the raw image
    mark = img;
    rockBoundary = bwperim(rockMask, 4); 
    ballBoundary = bwperim(ballMask, 4);
    mark = imoverlay(mark, rockBoundary, 'red');
    mark = imoverlay(mark, ballBoundary, 'yellow');
    [path, name, extension] = fileparts(filename); % path is to "Compressed folder"
    path = erase(path, '/Compressed'); % direct to the upper folder

    imwrite(mark, fullfile(path, 'Segmentation/', strcat(name, '.png')));
    imwrite(rockCrop, fullfile(path, 'Segmentation/', strcat('t', name, '_rock', '.png')));
    imwrite(ballCrop, fullfile(path, 'Segmentation/', strcat('t', name, '_ball', '.png')));

end