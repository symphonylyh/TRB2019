
function [voxel, sphericity, cornerPoints, digRatio] = reconstruct3Dx(views, D, debug_mode, rock_mode)
close all;

PLOT = debug_mode;
PLOT = false;
ROCK = rock_mode;

% Silence warning
voxel = 1;
sphericity = 1;
cornerPoints = 1;
digRatio = 1;

% Normalize/Scale with respect to the *top* view based on the diameter ratio of calibration ball
views{2} = imresize(views{2}, D(1) / D(2));
views{3} = imresize(views{3}, D(1) / D(3));

% Solve linear system
for i = 1 : 3
    b(2 * i - 1, 1) = size(views{i}, 2); % width
    b(2 * i, 1) = size(views{i}, 1); % height
end
A = [0 0 1; 1 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 0];

if (size(views{1}, 1) < 100) % ball
    b(7:9, 1) = [0 0 0]';
    A(7:9, :) = [1 -1 0; 0 1 -1; 1 0 -1];
end

scale = ceil(A \ b); % [x y z]
top = imresize(views{1}, [scale(1) scale(3)]);
front = imresize(views{2}, [scale(2) scale(1)]);
side = imresize(views{3}, [scale(2) scale(3)]);

% Extrude and rearrange into [x y z] dimension
top_extrude = repmat(top, [1 1 scale(2)]); % [x z y]
top_extrude = permute(top_extrude, [1 3 2]);
front_extrude = repmat(front, [1 1 scale(3)]); % [y x z]
front_extrude = permute(front_extrude, [2 1 3]);
side_extrude = repmat(side, [1 1 scale(1)]); % [y z x]
side_extrude = permute(side_extrude, [3 1 2]);

% Intersect the three extruded body
volume = top_extrude & front_extrude & side_extrude;
voxel = sum(volume(:));
volume_raw = volume; % for plot
voxel_raw = voxel;
surfaceTrace = volume - imerode(volume, true(3));
surfaceArea_raw = sum(surfaceTrace(:));
dim = size(volume);

side_recon = logical(squeeze(sum(volume,1)));
top_recon = logical(squeeze(sum(volume,2)));
front_recon = logical(squeeze(sum(volume,3)));

% tic
% if ROCK
%     while true
%         disp('Calculating next layer...');
% 
%         % Surface pixels and surface area are defined a little different
%         surface = bwperim(volume);
%         [px, py, pz] = ind2sub(size(surface), find(surface == 1));
%         surfaceArea = sum(surface(:));
% 
%         for i = 1 : surfaceArea
%             surface_previous = bwperim(volume);
%             surfaceArea_previous = sum(surface_previous(:));
% 
%             % Trial remove the current pixel
%             volume(px(i), py(i), pz(i)) = 0;
%             
%             % Criteria#1: Check for profile views
%             side_remove = logical(squeeze(sum(volume, 1)));
%             top_remove = logical(squeeze(sum(volume, 2)));
%             front_remove = logical(squeeze(sum(volume,3)));
%             side_check = (side_remove ~= side_recon);
%             top_check = (top_remove ~= top_recon);
%             front_check = (front_remove ~= front_recon);
%             check = sum(side_check(:)) + sum(top_check(:)) + sum(front_check(:));
%             
%             % Criteria#2: Check for surface area
%             surface = bwperim(volume);
%             surfaceArea_new = sum(surface(:));
%             
%             % Remove the pixel if #1: profile views are not affected #2: surface area decreases 
% %             if check == 0 || surfaceArea_new < surfaceArea_previous
% %                 volume(px(i), py(i), pz(i)) = 1; % rollback
% %             end
%             
%             if check == 0 && surfaceArea_new < surfaceArea_previous
%                 volume(px(i), py(i), pz(i)) = 0; % remove
%             else
%                 volume(px(i), py(i), pz(i)) = 1; % keep
%             end
%             
%         end
%         
%         if surfaceArea - surfaceArea_new <= 1
%             break;
%         end
%     end
% end
% toc
% 
% voxel = sum(volume(:));
% disp(voxel_raw);
% disp(voxel);
%--------------------------------------------------------------------------
% Minimum surface area from local evaluation
% tic
% if ROCK
%     removeRatio = 1; 
%     while removeRatio > 0
%         % Surface pixels and surface area are defined a little different
%         if removeRatio == 1 % loop 1, assign default to surfacePixel_unvisited (none of any is visited)
%             surfacePixel_unvisited = true(dim);
%         end
%         surfacePixel = bwperim(volume) & surfacePixel_unvisited;
%         [px, py, pz] = ind2sub(size(surfacePixel), find(surfacePixel == 1));
%         N = sum(surfacePixel(:)); % total number of surface pixels
%         surfacePixel_unvisited = ~surfacePixel & surfacePixel_unvisited; % record the pixels visited, so we will never check them again
%         
%         removeCount = 0;
%         for i = 1 : N % loop thru each surface pixel to check if it can be deleted
%             % Extract a local box around the surface pixel (e.g., 5x5x5, radius = 2)
%             r = 2;
%             localBox = volume(max(px(i)-r,1):min(px(i)+r, dim(1)), max(py(i)-r,1):min(py(i)+r, dim(2)), max(pz(i)-r,1):min(pz(i)+r, dim(3))); % avoid touching the matrix boundary
%             side_raw = logical(squeeze(sum(localBox,1)));
%             top_raw = logical(squeeze(sum(localBox,2)));
%             front_raw = logical(squeeze(sum(localBox,3)));
%             temp = localBox - imerode(localBox, true(3)); % get all surface pixels (including turning point)
%             area_raw = sum(temp(:));
%             
%             % Trial remove the current pixel
%             localBox(3,3,3) = 0; 
%             
%             % Criteria#1: Check for profile views
%             side_remove = logical(squeeze(sum(localBox, 1)));
%             top_remove = logical(squeeze(sum(localBox, 2)));
%             front_remove = logical(squeeze(sum(localBox,3)));
%             side_check = (side_remove ~= side_raw);
%             top_check = (top_remove ~= top_raw);
%             front_check = (front_remove ~= front_raw);
%             view_check = sum(side_check(:)) + sum(top_check(:)) + sum(front_check(:));
%             
%             % Criteria#2: Check for surface area
%             temp = localBox - imerode(localBox, true(3)); % get all surface pixels (including turning point)
%             area_check = sum(temp(:));
%             
%             % Remove the pixel if #1: profile views are not affected #2: surface area decreases 
%             if view_check == 0 && area_check < area_raw 
%                 volume(max(px(i)-r,1):min(px(i)+r, dim(1)), max(py(i)-r,1):min(py(i)+r, dim(2)), max(pz(i)-r,1):min(pz(i)+r, dim(3))) = 0;
%                 removeCount = removeCount + 1;
%             end
%             
%         end
%         removeRatio = removeCount / N;
%         disp('Loop+1');
%     end
%     
%     % Calculate final minimal surface area and volume
%     surfaceTrace = volume - imerode(volume, true(3)); % get all surface pixels (including turning point)
%     surfaceArea = sum(surfaceTrace(:));
%     voxel = sum(volume(:));
%     
%     % Report volume and surface area before and after the scheme
% %     voxel_raw
% %     voxel
% %     surfaceArea_raw
% %     surfaceArea
%     disp(voxel/voxel_raw);
% 
%     % Plot
%     % volume_raw
%     % volume
% end
% toc

tic
if ROCK
    remove = true; % loop flag: true if there are pixels to be removed, false if no more pixel on the surface can be removed (minimum surface principle)
    while remove == true
        % Surface pixels and surface area are defined a little different
        surfacePixel = bwperim(volume);
        [px, py, pz] = ind2sub(size(surfacePixel), find(surfacePixel == 1));
        N = sum(surfacePixel(:)); % total number of surface pixels
        
        remove = false; % stop sign for while loop.
        for i = 1 : N % loop thru each surface pixel to check if it can be deleted
            if true
            % Extract a local box around the surface pixel (e.g., 5x5x5, radius = 2)
%             r = 2;
%             localBox = volume(max(px(i)-r,1):min(px(i)+r, dim(1)), max(py(i)-r,1):min(py(i)+r, dim(2)), max(pz(i)-r,1):min(pz(i)+r, dim(3))); % avoid touching the matrix boundary
%             side_bar = volume(:, max(py(i)-r,1):min(py(i)+r, dim(2)), max(pz(i)-r,1):min(pz(i)+r, dim(3)));
%             top_bar = volume(max(px(i)-r,1):min(px(i)+r, dim(1)), :, max(pz(i)-r,1):min(pz(i)+r, dim(3)));
%             front_bar = volume(max(px(i)-r,1):min(px(i)+r, dim(1)), max(py(i)-r,1):min(py(i)+r, dim(2)), :);
%             side_raw = logical(squeeze(sum(side_bar,1)));
%             top_raw = logical(squeeze(sum(top_bar,2)));
%             front_raw = logical(squeeze(sum(front_bar,3)));
%             temp = localBox - imerode(localBox, true(3)); % get all surface pixels (including turning point)
%             area_raw = sum(temp(:));
%             
%             % Trial remove the current pixel
%             localBox(3,3,3) = 0; 
%             side_bar(px(i),3,3) = 0;
%             top_bar(3,py(i),3) = 0;
%             front_bar(3,3,pz(i)) = 0;
%             
%             % Criteria#1: Check for profile views
%             side_remove = logical(squeeze(sum(side_bar, 1)));
%             top_remove = logical(squeeze(sum(top_bar, 2)));
%             front_remove = logical(squeeze(sum(front_bar,3)));
%             side_check = (side_remove ~= side_raw);
%             top_check = (top_remove ~= top_raw);
%             front_check = (front_remove ~= front_raw);
%             view_check = sum(side_check(:)) + sum(top_check(:)) + sum(front_check(:));
%             
%             % Criteria#2: Check for surface area
%             temp = localBox - imerode(localBox, true(3)); % get all surface pixels (including turning point)
%             area_check = sum(temp(:));
%             
%             % Remove the pixel if #1: profile views are not affected #2: surface area decreases 
%             if view_check == 0 && area_check < area_raw 
%                 volume(max(px(i)-r,1):min(px(i)+r, dim(1)), max(py(i)-r,1):min(py(i)+r, dim(2)), max(pz(i)-r,1):min(pz(i)+r, dim(3))) = 0;
%                 remove = true;
%             end
            end
            
            temp = volume - imerode(volume, true(3)); %bwperim(volume);
            area_raw = sum(temp(:));
            
            % Trial remove the current pixel
            volume(px(i), py(i), pz(i)) = 0;
            
            % Criteria#1: Check for profile views
            side_remove = logical(squeeze(sum(volume, 1)));
            top_remove = logical(squeeze(sum(volume, 2)));
            front_remove = logical(squeeze(sum(volume,3)));
            side_check = (side_remove ~= side_recon);
            top_check = (top_remove ~= top_recon);
            front_check = (front_remove ~= front_recon);
            view_check = sum(side_check(:)) + sum(top_check(:)) + sum(front_check(:));      
            
            % Criteria#2: Check for surface area
            temp = volume - imerode(volume, true(3)); % bwperim(volume);
            area_check = sum(temp(:));
            
            % Remove the pixel if #1: profile views are not affected #2: surface area decreases 
            if view_check == 0 && area_check <= area_raw
                remove = true;
            else
                volume(px(i), py(i), pz(i)) = 1;
            end
            
        end
        disp('Loop+1');
    end
    
    % Calculate final minimal surface area and volume
    surfaceTrace = volume - imerode(volume, true(3)); % get all surface pixels (including turning point)
    surfaceArea = sum(surfaceTrace(:));
    voxel = sum(volume(:));
    
    % Report volume and surface area before and after the scheme
%     voxel_raw
%     voxel
%     surfaceArea_raw
%     surfaceArea
    disp(voxel_raw);
    disp(voxel);
    % Plot
    % volume_raw
    % volume
end
toc

if PLOT
figure(4);
% top
surface([0 scale(3); 0 scale(3)], [0 0; scale(1) scale(1)], [scale(2) scale(2); scale(2) scale(2)], 'FaceColor', 'texturemap', 'CData', flip(double(top)));
% front
surface([0 scale(3); 0 scale(3)], [0 0; 0 0], [0 0; scale(2) scale(2)], 'FaceColor', 'texturemap', 'CData', flip(double(side)));
% left
flip_front = imrotate(double(front), 90);
flip_front = flip(flip_front, 1);
flip_front = flip(flip_front, 2);
surface([0 0; 0 0], [scale(1) scale(1); 0 0], [0 scale(2); 0 scale(2)], 'FaceColor', 'texturemap', 'CData', flip_front);
axis equal
view(3);
  
% Plot 3d view   
figure(5);
% adjust the coordinates
volume = flip(volume, 1); % flip x and y
volume = flip(volume, 2);
volume = permute(volume, [1 3 2]); % exchange y and z
v = double(volume);
v = smooth3(v);
p = patch( isosurface(v,0) );                 %# create isosurface patch
isonormals(v, p)                              %# compute and set normals
set(p, 'FaceColor','r', 'EdgeColor','none')   %# set surface props

daspect([1 1 1])                              %# axes aspect ratio
view(0, 90), axis vis3d tight, box on, grid on    %# set axes props
set(gca, 'CameraUpVector', [0 1 0])
set(gca, 'CameraUpVectorMode', 'manual')
rotate3d on
camproj perspective                           %# use perspective projection
camlight('left'), lighting phong, alpha(1)
% end plot 3d view
end

end % end of function