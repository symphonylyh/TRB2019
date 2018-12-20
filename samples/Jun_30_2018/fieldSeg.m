%% File Dependencies: 
% Core: illiSeg.m, reconstruct3D.m
% Accessory: rgb2lab.m, tight_subplot.m, getAllFilesInFolder.m
%
% Flowchart: 
% Pair images into triplets, rename and compress the image files --- READ module
% 

%% Control panel
READ = false; COMPRESS = true; compress_size = 1024; % Rename image file (one-time only), compress image file (if the resolution remains the same, turn off the switch)
SEGMENT = true;
RECONSTRUCT = false;

% User define folder name here
inFolderName = './samples/Jul_12_2018/'; 

%% READ: Read image files
if READ  
    % Create output folder for raw images
    rawFolderName = strcat(inFolderName, 'Raw/');
    if ~exist(rawFolderName, 'dir')
        mkdir(rawFolderName);
    end   
    
    % Specify the working folder and get all image files in it
    fnames = getAllFilesInFolder(inFolderName); % getAllFilesInFolder.m can be customed to filter out some file types
%     if (mod(length(fnames),3) ~= 0) % check if the images are correctly taken 3 views of each particle
%         error("Images are not paired in triplet...Please check if some views are missing!");
%     end

    % Format arbitrary image file names to img000N_X and move to Raw folder
    % where N = image No. and X = 0(top)/1(front)/2(side) 
    % Note: raw images should be taken in sequence front-->side-->top!
    % The filename formatting should only be done once for a folder
    % Get the file extension
    [path, name, extension] = fileparts(fnames{1}); 
    for i = 1 : length(fnames)
        % newFileName = strcat('img', sprintf('%04d', ceil(i / 3)), '_', num2str(mod(i, 3)), extension); % for triplet images
        % For individual set
        view = 0; % 0(top)/1(front)/2(side) 
        newFileName = strcat('img', sprintf('%04d', i), '_', num2str(view), extension);
        
        % Rename files and put them under "Raw" folder
        movefile(fullfile(inFolderName, fnames{i}), fullfile(rawFolderName, newFileName));     
    end
    
    % Avoid having too large image file and long running time, pre-processing the files
    if COMPRESS
        % Create output folder for compressed images
        compressFolderName = strcat(inFolderName, 'Compressed/');
        if ~exist(compressFolderName, 'dir')
            mkdir(compressFolderName);
        end
        
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
    
end

%% SEGMENT: Particle and calibration ball segmentation
if SEGMENT
    close all;
    
    % Locate input files
    compressFolderName = strcat(inFolderName, 'Compressed/');
    fnames = getAllFilesInFolder(compressFolderName);
    
%     rawFolderName = strcat(inFolderName, 'Raw/');
%     fnames = getAllFilesInFolder(rawFolderName);
    
    % Create output folder
    segFolderName = strcat(inFolderName, 'Segmentation/');
    if ~exist(segFolderName, 'dir')
        mkdir(segFolderName);
    end
    
    % Group segmentation or single segmentation based on user's option
    DEBUG = true; object = 3; view = 1; % designate the object & view (1-top;2-front;3-side) to debug
    if DEBUG
        % Create debug folder or clear existing folder
        debugFolderName = strcat(segFolderName, 'Debug/');
        if ~exist(debugFolderName, 'dir')
            mkdir(debugFolderName);
        else
            % Clear existing files in debug folder
            % delete(strcat(debugFolderName, '*'));
        end
        % Debug individual image (either illiSeg or illiSeg_old)
        % For images organized in triplet sequence: 
        % result = illiSeg(fullfile(compressFolderName, fnames{(object - 1) * 3 + view}), DEBUG);
        
        % For batch processing of each view (phone) images separately:
        result = illiSeg(fullfile(compressFolderName, strcat('img', sprintf('%04d', object), '_', num2str(view - 1), '.png')), DEBUG);
    else
        % For batch processing of each view (phone) images separately:
        for i = 1 : length(fnames)
            illiSeg(fullfile(compressFolderName, fnames{i}), DEBUG);
        end
        
        % For images organized in triplet sequence: 
%         summary = [];
%         for object = 1 : length(fnames) / 3
%             results = [];
%             for i = 1 : 3 % image triplet of top-front-side views of an object
%                 views{i} = fullfile(compressFolderName, fnames{(object - 1) * 3 + i});
%                 results = cat(2, results, illiSeg(views{i}, DEBUG));
%             end
%             summary(2*(object-1)+1:2*(object-1)+2, :) = results;
%             % summary compiles particle information of all the segmented images
%             % Columns are the hole ratios of the rock particle and the
%             % equivalent diameters of the calibration ball, interleaved following
%             % top-front-side sequence. Rows are interleaved as rock-ball pair.
%             % Example:
%             % ratio1_top ratio1_front ratio1_side ---- rock1
%             % diamt1_top diamt1_front diamt1_side ---- ball1
%             % ratio2_top ratio2_front ratio2_side ---- rock2
%             % diamt2_top diamt2_front diamt2_side ---- ball2
%             % ratio3_top ratio3_front ratio3_side ---- rock3
%             % diamt3_top diamt3_front diamt3_side ---- ball3
%             % ...
%         end
%         % Save the summary info to disk
%         save(fullfile(segFolderName, 'summary.mat'), 'summary');
    end
      
end

%% RECONSTRUCT: 3D reconstruction and volume estimation
if RECONSTRUCT
    close all;
    
    % Read measured weight/volume information
    file = fopen(fullfile(inFolderName, 'measure.txt'));
    line1 = textscan(fgetl(file), '%d %d', 'Delimiter', ' '); % volume/weight indicator: integer 0 1 
    volumeFlag = line1{1};
    weightFlag = line1{2};
    line2 = textscan(fgetl(file), '%d', 'Delimiter', ' '); % number of view repetition
    repetition = line2{1};
    i = 1; final = [];
    while feof(file) ~= 1
        line = textscan(fgetl(file), '%f %f', 'Delimiter', ' ');
        if volumeFlag && weightFlag % volume and weight
            final(i, 1) = line{1};
            final(i, 2) = line{2};
        elseif volumeFlag % volume only
            final(i, 1) = line{1};
            final(i, 2) = 0;
        else % weight only
            final(i, 1) = 0;
            final(i, 2) = line{1};
        end
        i = i + 1;
    end
    final_full = repelem(final, repetition, 1); % repeat each elements in a matrix
    fclose(file);
    
    % Matrix 'final':
    % 1st column -- Measured volume (in cm3)
    % 2nd column -- Measured weight (in g)
    % Matrix 'final_full': the replicated 'final' based on object
    % repetition

    % Locate input files
    segFolderName = strcat(inFolderName, 'Segmentation/');
%     S = load(fullfile(segFolderName, 'summary.mat'), '-mat');
%     info = S.summary;
    
    % Create output folder
    reconFolderName = strcat(inFolderName, 'Reconstruction/');
    if ~exist(reconFolderName, 'dir')
        mkdir(reconFolderName);
    end
    
    % Group reconstruction or single reconstruction based on user's option
    DEBUG = false; object = 2; % designate the object to debug
    if DEBUG
        % Create debug folder or clear existing folder
        debugFolderName = strcat(reconFolderName, 'Debug/');
        if ~exist(debugFolderName, 'dir')
            mkdir(debugFolderName);
        else
            % Clear existing files in debug folder
            delete(strcat(debugFolderName, '*'));
        end
        % Debug single object
        for view = 1 : 3
            rocks{view} = imread(fullfile(segFolderName, strcat('timg', sprintf('%04d', object), '_', num2str(view - 1), '_rock.png')));
            balls{view} = imread(fullfile(segFolderName, strcat('timg', sprintf('%04d', object), '_', num2str(view - 1), '_ball.png')));
            % D(view) = info(2 * object, view); 
            D(view) = min(size(balls{view})); % based on minimum dimension
            %R(view) = info(2 * object - 1, view);
        end
        rockVoxel = reconstruct3D(rocks, D, DEBUG);
        ballVoxel = reconstruct3D(balls, D, DEBUG);
        rockVolume = rockVoxel / ballVoxel * 8 * (2 - sqrt(2)) * 0.75^3 * 16.3871; % the orthogonal intersection volume of a sphere
        
        % Plot volume comparsion
        figure; hold on;
        range = 2000;
        xlim([0 range]), ylim([0 range]), pbaspect([1 1 1]);
        handle = zeros(4, 1);
        handle(1) = plot(final_full(object,1), rockVolume, '*r');  % data point
        refLine = linspace(0, range, 6); 
        percent10Error = refLine .* 0.1;
        percent20Error = refLine .* 0.2;
        handle(2) = plot(refLine, refLine, '-k', 'LineWidth', 1); % 45 deg reference line
        handle(3) = plot(refLine, refLine + percent10Error, '--g', 'LineWidth', 1); % 10% error range line
        plot(refLine, refLine - percent10Error, '--g', 'LineWidth', 1);
        handle(4) = plot(refLine, refLine + percent20Error, '--b', 'LineWidth', 1); % 20% error range line
        plot(refLine, refLine - percent20Error, '--b', 'LineWidth', 1);
        legend(handle, 'Reconstructed Volume', 'Reference Line', '10% Eror', '20% Error', 'Location', 'NorthWest');
        title('Volume Comparsion'), xlabel('Actual Volume (in cm3)'), ylabel('Reconstructed Volume (in cm3)');
        % saveas(gcf, fullfile(debugFolderName, 'comparison_volume.png'));
    else
        % Calculate the benchmarked dimensions (x,y,z) from the least squares 
        % solution of the linear system
        nums = size(final_full, 1); % size(info,1) / 2; % number of particles
        weights = [];
        volumes = [];
        for i = 1 : nums
            D = []; % diameters of calibration ball
            R = []; % hole ratios of rock
            for j = 1 : 3
                rocks{j} = imread(fullfile(segFolderName, strcat('timg', sprintf('%04d', i), '_', num2str(j - 1), '_rock.png')));
                balls{j} = imread(fullfile(segFolderName, strcat('timg', sprintf('%04d', i), '_', num2str(j - 1), '_ball.png')));
                % D(j) = info(2 * i, j); 
                D(j) = min(size(balls{j})); % options: use equivalent diameter, or the minimum diameter
                % R(j) = info(2 * i - 1, j);
            end
            rockVoxel = reconstruct3D(rocks, D, DEBUG);
            [ballVoxel, sphericity] = reconstruct3D(balls, D, DEBUG);
            % Options for the calibration ball volume: 
            % 1. Actual 1 in. ball volume is 4/3*PI*R^3 = 0.523599 in3
            % use the ball diameter in top view to compute volume from the
            % volume equation
            % 2. Assume the reconstructed body is exactly the 1 in. ball
            % volume            
            % 3. Theoretical 3D reconstructed ball is not a sphere, but a
            % intersected body with volume V = 8(2 - sqrt(2))R^3
            % use the reconstructed ball voxel to compute the rock volume
            % Note: an overall calibration factor is needed b/c the
            % reconstructed volume is consistently greater than the actual,
            % by default 0.8
            % Ref: http://xuxzmail.blog.163.com/blog/static/25131916200974113416209/
            
            % rockVolume = 0.8 * rockVoxel / (4 / 3 * 3.1415926 * (D(1)/2)^3) * 0.523599 * 16.3871;
            % rockVolume = 0.8 * rockVoxel / ballVoxel * 0.523599 * 16.3871; % calibration ball is V = 4/3 * PI * R3 = 0.523599 in3; 1 in3 = 16.3871 cm3
            rockVolume =  rockVoxel / ballVoxel * 8 * (2 - sqrt(2)) * 0.75^3 * 16.3871; % the orthogonal intersection volume of a sphere
            Gs = 2.65; % typical specific gravity of rock = 2.65g/cm3
            rockWeight = rockVolume * Gs; 
            volumes(i, 1) = rockVolume;
            weights(i, 1) = rockWeight;
            % sphere(i,1) = sphericity; % not used
            
            % Volume correction based on hole ratio (not used for now)
            % holeRatio = 1 - mean(R);
            % rockVoxel = rockVoxel * holeRatio;     
        end
        
        % Save the 3D voxel array to disk
        save(fullfile(reconFolderName, 'volume.mhat'), 'volumes'); 
            
        % For full results
        error_volume = (volumes - final_full(:, 1)) ./ final_full(:, 1) * 100;
        error_weight = (weights - final_full(:, 2)) ./ final_full(:, 2) * 100;
        final_full = [final_full(:, 1) volumes error_volume final_full(:, 2) weights error_weight];
        
        % For average results
        volumes = reshape(volumes, repetition, []); % reshape the matrix
        mean_volume = mean(volumes, 1)'; % take the mean every n repetitons
        error_volume = (mean_volume - final(:, 1)) ./ final(:, 1) * 100;
        weights = reshape(weights, repetition, []);
        mean_weight = mean(weights, 1)';
        error_weight = (mean_weight - final(:, 2)) ./ final(:, 2) * 100;
        final = [final(:, 1) mean_volume error_volume final(:, 2) mean_weight error_weight];
        
        % Plot volume comparsion
        figure(1); hold on;
        range = 2000;
        xlim([0 range]), ylim([0 range]), pbaspect([1 1 1]); 
        handle = zeros(5, 1);
        handle(1) = plot(final_full(:,1), final_full(:,2), '*r'); % data point
        handle(2) = plot(final(:, 1), final(:, 2), 'ob'); % average value
        refLine = linspace(0, range, 6); 
        percent10Error = refLine .* 0.1;
        percent20Error = refLine .* 0.2;
        handle(3) = plot(refLine, refLine, '-k', 'LineWidth', 1); % 45 deg reference line
        handle(4) = plot(refLine, refLine + percent10Error, '--g', 'LineWidth', 1); % 10% error range line
        plot(refLine, refLine - percent10Error, '--g', 'LineWidth', 1);
        handle(5) = plot(refLine, refLine + percent20Error, '--b', 'LineWidth', 1); % 20% error range line
        plot(refLine, refLine - percent20Error, '--b', 'LineWidth', 1);
        title('Volume Comparsion'), xlabel('Actual Volume (in cm3)'), ylabel('Reconstructed Volume (in cm3)');
        legend(handle, 'Reconstructed Volume', 'Average Volume', 'Reference Line', '10% Eror', '20% Error', 'Location', 'NorthWest');
        % saveas(gcf, fullfile(reconFolderName, 'comparison_volume.png'));
        
        % Plot weight comparison
%         figure(2); hold on;
%         range = 4000;
%         xlim([0 range]), ylim([0 range]), pbaspect([1 1 1]);
%         handle = zeros(5, 1);
%         handle(1) = plot(final_full(:,4), final_full(:,5), '*r'); % data point
%         handle(2) = plot(final(:, 4), final(:, 5), 'ob'); % average value
%         refLine = linspace(0, range, 6);
%         percent10Error = refLine .* 0.1;
%         percent20Error = refLine .* 0.2;
%         handle(3) = plot(refLine, refLine, '-k', 'LineWidth', 1); % 45 deg reference line
%         handle(4) = plot(refLine, refLine + percent10Error, '--g', 'LineWidth', 1); % 10% error range line
%         plot(refLine, refLine - percent10Error, '--g', 'LineWidth', 1);
%         handle(5) = plot(refLine, refLine + percent20Error, '--b', 'LineWidth', 1); % 20% error range line
%         plot(refLine, refLine - percent20Error, '--b', 'LineWidth', 1);
%         title('Weight Comparsion'), xlabel('Actual Weight (in g)'), ylabel('Reconstructed Weight (in g)');
%         legend(handle, 'Reconstructed Weight', 'Average Weight', 'Reference Line', '10% Eror', '20% Error', 'Location', 'NorthWest');
%         % saveas(gcf, fullfile(reconFolderName, 'comparison_weight.png'));
    end
    
end

%% Notes
% Volume visualization tutorial:
% https://blogs.mathworks.com/videos/2009/10/23/basics-volume-visualization-19-defining-scalar-and-vector-fields/
% https://stackoverflow.com/questions/2942251/matlab-3d-volume-visualization-and-3d-overlay
% https://stackoverflow.com/questions/13553108/how-i-can-display-3d-logical-volume-data-matlab
% https://stackoverflow.com/questions/6891154/creating-3d-volume-from-2d-slice-set-of-grayscale-images
% 
% Isosurface: isosurface(V, value)
% https://www.mathworks.com/help/matlab/ref/isosurface.html
% 
% 3D binary boundary of mask:
% https://www.mathworks.com/matlabcentral/answers/85180-multi-dimensional-version-of-bwboundaries
%
% 3D boundary of a set of points: boundary(x,y,z)
% https://www.mathworks.com/help/matlab/ref/boundary.html
%
% 3D Minkowski geometric measures: imMinkowski package
% https://www.mathworks.com/matlabcentral/fileexchange/33690-geometric-measures-in-2d-3d-images
%
% 3D voxel rendering: vol3d package
% https://www.mathworks.com/matlabcentral/fileexchange/22940-vol3d-v2
% https://blogs.mathworks.com/pick/2013/10/04/easy-visualization-of-volumetric-data/
% 
% 3D reconstruct:
% https://www.mathworks.com/matlabcentral/fileexchange/3280-voxel
% https://www.mathworks.com/matlabcentral/fileexchange/42876-surf2solid-make-a-solid-volume-from-a-surface-for-3d-printing?focused=3810976&tab=function
% https://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d?focused=8549167&tab=function
% https://www.mathworks.com/matlabcentral/fileexchange/37268-3d-volume-visualization
% https://www.mathworks.com/matlabcentral/fileexchange/59161-volumetric-3?s_tid=srchtitle
% https://www.mathworks.com/help/images/explore-3-d-volumetric-data-with-volume-viewer-app.html