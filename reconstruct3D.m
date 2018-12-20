function [voxel, sphericity, cornerPoints, digRatio, voxels, X12] = reconstruct3D(views, D, debug_mode, rock_mode)
% 3D reconstruction based on three near-orthogonal views.
%
% Input:
% views: 1x3 cell array containing three views (binary images, in sequence top-front-side)
% D: 1x3 vector containing three equivalent diameters of calibration ball
% in each view
%
% Output:
% voxel: voxel counter of the reconstructed object

% Silence warning
voxel = 1;
sphericity = 1;
cornerPoints = 1;
digRatio = 1;
voxels = 1;
X12 = 1;
close all;

PLOT = debug_mode;
PLOT = true;
ROCK = rock_mode;
ROCK = false;

% Normalize/Scale with respect to the *top* view based on the diameter ratio of calibration ball
views{2} = imresize(views{2}, D(1) / D(2));
views{3} = imresize(views{3}, D(1) / D(3));

% Update the new image dimension in info table
% info(2 * i - 1, 2 * 2 -1 : 2 * 2) = [size(views{2}, 2) size(views{2}, 1)];
% info(2 * i - 1, 2 * 3 -1 : 2 * 3) = [size(views{3}, 2) size(views{3}, 1)];

% Following the sequence of photos top-front-side
% a right-hand coordinates system is used:
% -------- +x
% |
% |
% |
% |
% +y
% and positive z is pointing into the screen
% in the image sets taken April 25th, each row in summary matrix
% is [z x x y z y]

% The target is to find a benchmark dimension [x0 y0 z0] for three views to 
% be scaled with, thereby voxels can be calculated. The method is to solve
% a least squares problem of the linear system.
% Ax ~= b
% A       *     x      =  b    
% | 0  0  1 |  | x0 |     | z |  top_width
% | 1  0  0 |  | y0 |     | x |  top_height
% | 1  0  0 |  | z0 |     | x |  front_width
% | 0  1  0 |             | y |  front_height
% | 0  0  1 |             | z |  side_width
% | 0  1  0 |             | y |  side_height
%    +z________________
%     /               /|
%    /      p        / |
%   /      o        /  |
%  /      T        /   |
% /_______________/+x  /
% |               |   /  e
% |               |  /  d
% |    Front      | /  i
% |_______________|/  S
% +y
% The text direction shows the way the photo is took
% Restrict x = y = z for calibration ball
% | 1  -1  0 |             | 0 |  x ~= y
% | 0  1  -1 |             | 0 |  y ~= z
% | 1  0  -1 |             | 0 |  x ~= z

% Solve linear system
for i = 1 : 3
    b(2 * i - 1, 1) = size(views{i}, 2); % width
    b(2 * i, 1) = size(views{i}, 1); % height
end
% b = info(2 * i - 1, :)';
A = [0 0 1; 1 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 0];

if (size(views{1}, 1) < 100) % ball reconstruction
    b(7:9, 1) = [0 0 0]';
    A(7:9, :) = [1 -1 0; 0 1 -1; 1 0 -1];
end
scale = ceil(A \ b); % [x y z]
top = imresize(views{1}, [scale(1) scale(3)]);
front = imresize(views{2}, [scale(2) scale(1)]);
side = imresize(views{3}, [scale(2) scale(3)]);

distortion = @(xyz) ( abs(xyz(1) - xyz(2)) + abs(xyz(2) - xyz(3)) + abs(xyz(1) - xyz(3)) ) / mean(xyz); 
sphericity = 1 - distortion(scale); % calculate the sphericity of the captured calibration ball, 1 for perfect sphere

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

if rock_mode
    % New intersection trial
    cubeVoxel = size(volume,1) * size(volume, 2) * size(volume, 3);

    F1 = top_extrude;
    F2 = front_extrude;
    F3 = side_extrude;

    AAA = F1 & F2 & F3;
    BBB = ~F1 & ~F2 & ~F3;

    AAB = F1 & F2 & ~F3;
    ABA = F1 & ~F2 & F3;
    BAA = ~F1 & F2 & F3;

    ABB = F1 & ~F2 & ~F3;
    BBA = ~F1 & ~F2 & F3;
    BAB = ~F1 & F2 & ~F3;
    
    voxels = [cubeVoxel sum(AAA(:)) sum(BBB(:)) sum(AAB(:)) sum(ABA(:)) sum(BAA(:)) sum(ABB(:)) sum(BBA(:)) sum(BAB(:))];
    %X12 = (voxel - min([sum(ABB(:)) sum(BBA(:)) sum(BAB(:))]))/voxel;
    %X12 = 0.25 * (cubeVoxel - sum(AAA(:)) - sum(BBB(:)))/cubeVoxel;
    X12 = (cubeVoxel - sum(AAA(:)) - sum(BBB(:)))/cubeVoxel;
end

% ballratio = 8*(2-sqrt(2))/(4/3*3.1415926)
if ROCK
[Gx, Gy, Gz] = imgradientxyz(volume);
Gx = mat2gray(abs(Gx));
Gy = mat2gray(abs(Gy));
Gz = mat2gray(abs(Gz));
corner = Gx > 0.5 & Gy > 0.5 & Gz > 0.5 & volume == 1;
[x1 y1 z1] = ind2sub(size(corner), find(corner == 1));
cornerPoints = sum(corner(:));

% Only show one corner point
% corner = false(size(corner));
% corner(x1(1), y1(1), z1(1)) = true;

side_recon = logical(squeeze(sum(volume,1)));
top_recon = logical(squeeze(sum(volume,2)));
front_recon = logical(squeeze(sum(volume,3)));

%r = floor((voxel/50000)^ 1/3);
volume_curr = volume; % keep track of the current state of the volume
dim = size(volume); % dimension of the matrix (for checking if the dig volume exceed the boundary)
radius = [];
digVoxel = [];
tic
for i = 1 : cornerPoints
    % Determine the maximum diggable volume by an iteration scheme
    check = 0; r = 15; % reset check point and starting radius everytime
    isEmpty = false;
    volume_trial = volume_curr;
    %while check == 0
        % Dig in a cube of radius r
%         dig = volume_trial(max(x1(i)-r,1):min(x1(i)+r, dim(1)), max(y1(i)-r,1):min(y1(i)+r, dim(2)), max(z1(i)-r,1):min(z1(i)+r, dim(3)));
%         if sum(dig(:)) == 0 % if there is no solid voxel in the current volume, it means this area is already digged by previous corner points. Skip the loop and skip the convex dig further below
%             isEmpty = true;
%             r = 0;
%             dig_convex = 0;
%             break;
%         end
        volume_trial(max(x1(i)-r,1):min(x1(i)+r, dim(1)), max(y1(i)-r,1):min(y1(i)+r, dim(2)), max(z1(i)-r,1):min(z1(i)+r, dim(3))) = 0;
        % Check if the dig will affect the three profile views
        side_dig = logical(squeeze(sum(volume_trial, 1))); % profile views after dig
        top_dig = logical(squeeze(sum(volume_trial, 2)));
        front_dig = logical(squeeze(sum(volume_trial,3)));
        side_diff = (side_dig ~= side_recon); % different pixels between original profile and profile after dig
        top_diff = (top_dig ~= top_recon);
        front_diff = (front_dig ~= front_recon);
        check = sum(side_diff(:)) + sum(top_diff(:)) + sum(front_diff(:)); % check if any different pixel exists
        r = 2 * r; % double the radius for the next larger dig
    %end
    
    dig_convex = 0; %
    
    if check == 0
    % Dig the maximum convex volume and update the volume
    r = r / 2; % maximum diggable radius (in while loop an additonal *2 is multiplied)
    dig = volume_curr(max(x1(i)-r,1):min(x1(i)+r, dim(1)), max(y1(i)-r,1):min(y1(i)+r, dim(2)), max(z1(i)-r,1):min(z1(i)+r, dim(3))); % extract the neighborhood (cube) of the current corner point
    sz = size(dig);
    
    if sum(dig(:)) ~= 0 % no need to do convex dig is there is no voxel in the volume
    
    % Determine dig direction
    cube_centroid = sz / 2; % centroid of the dig cube
    dig_centroid = centerOfMass(single(dig)); % centroid of the dig volume (or dig mass)
    dig_direction = dig_centroid - cube_centroid; % indicate at which corner the dig volume is concentrated
    v = dig_direction; % for abbreviation
    
    % Switch between 8 different types of convex pyramid prototype
    t = zeros(sz); % the pyramid to be formed based on dig direction
    x0 = sz(1); y0 = sz(2); z0 = sz(3);
    if v(1) < 0 && v(2) < 0 && v(3) < 0 % pyramid 1 - - - quadrant
        %p1 = [x0 0 0]; p2 = [0 y0 0]; p3 = [0 0 z0];
        p1 = [x0 y0 0]; p2 = [x0 0 z0/2]; p3 = [0 0 z0];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x-1 y-1 z-1]) >= d % dot(normal,[x-1 y-1 z-1]) <= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) > 0 && v(2) < 0 && v(3) < 0 % pyramid 2 + - -
        %p1 = [0 0 0]; p2 = [x0 0 z0]; p3 = [x0 y0 0];
        p1 = [0 y0 0]; p2 = [x0 0 z0]; p3 = [0 0 z0/2];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x-1 y-1 z-1]) >= d % dot(normal,[x-1 y-1 z-1]) <= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) > 0 && v(2) < 0 && v(3) > 0 % pyramid 3 + - +
        %p1 = [x0 0 0]; p2 = [0 0 z0]; p3 = [x0 y0 z0];
        p1 = [x0 0 0]; p2 = [0 y0 z0]; p3 = [0 0 z0/2];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x y z]) >= d % dot(normal,[x y z]) <= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) < 0 && v(2) < 0 && v(3) > 0 % pyramid 4 - - +
        %p1 = [x0 0 z0]; p2 = [0 0 0]; p3 = [0 y0 z0];
        p1 = [0 0 0]; p2 = [x0 y0 z0]; p3 = [x0 0 z0/2];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x-1 y-1 z-1]) <= d % dot(normal,[x-1 y-1 z-1]) <= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) < 0 && v(2) > 0 && v(3) < 0 % pyramid 5 - + -
        % p1 = [0 0 0]; p2 = [x0 y0 0]; p3 = [0 y0 z0];
        p1 = [x0 0 0]; p2 = [0 y0 z0]; p3 = [0 0 z0/2];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x-1 y-1 z-1]) <= d % dot(normal,[x-1 y-1 z-1]) <= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) > 0 && v(2) > 0 && v(3) < 0 % pyramid 6 + + -
        % p1 = [x0 y0 z0]; p2 = [0 y0 0]; p3 = [x0 0 0];
        p1 = [x0 y0 z0]; p2 = [0 0 0]; p3 = [x0 0 z0/2];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x y z]) <= d % dot(normal,[x y z]) <= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) > 0 && v(2) > 0 && v(3) > 0 % pyramid 7 + + +
        % p1 = [x0 0 z0]; p2 = [x0 y0 0]; p3 = [0 y0 z0];
        p1 = [x0 y0 0]; p2 = [x0 0 z0/2]; p3 = [0 0 z0];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x y z]) <= d % dot(normal,[x y z]) >= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end

    elseif v(1) < 0 && v(2) > 0 && v(3) > 0 % pyramid 8 - + +
        %p1 = [0 0 z0]; p2 = [x0 y0 z0]; p3 = [0 y0 0];
        p1 = [0 y0 0]; p2 = [x0 0 z0]; p3 = [0 0 z0/2];
        normal = cross(p2 - p1, p3 - p1);
        d = dot(normal,p1);
        for x = 1:sz(1)
            for y = 1:sz(2)
                for z = 1:sz(3)
                    if dot(normal,[x y z]) <= d % dot(normal,[x y z]) >= d
                        t(x,y,z) = 1;
                    end
                end
            end
        end
    end
%         figure, VoxelPlotter(t, 1);
%         figure, VoxelPlotter(dig, 1);
%         figure, VoxelPlotter(logical(t) & dig, 1);

    % Dig and update the volume
    dig_convex = logical(t) & dig; % intersection convex volume
    volume_curr(max(x1(i)-r,1):min(x1(i)+r, dim(1)), max(y1(i)-r,1):min(y1(i)+r, dim(2)), max(z1(i)-r,1):min(z1(i)+r, dim(3))) = dig_convex;
    
    end
    end
    
    % Record the dig radius and dig voxel for output
    radius(i) = r; 
    digVoxel(i) = sum(dig_convex(:));   
end
toc
digVoxelSum = sum(digVoxel);
digRatio = digVoxelSum/voxel;

% unique_radius = unique(radius);
% unique_count = arrayfun(@(x)length(find(radius == x)), unique(radius), 'Uniform', false);
% unique_count = cell2mat(unique_count);
% removal = [unique_radius' unique_count'];

% [x y z] = ind2sub(size(volume), find(volume == 1));
% solidCoord = [x y z];
% scatter3(x, y, z, 'MarkerFaceColor',[0 .75 .75]);
% hold on;
% scatter3(x1, y1, z1, 'MarkerFaceColor',[0.75 0 0]);
end
%volume = volume_curr;

% rockVolume = voxel / D(1)^3 * 1^3; % in in^3
% rockWeight = rockVolume * 16.3871 * 2.65; % 1 in3 = 16.3871 cm3; typically rock density 2.65g/cm3
% volumes(i, 1) = rockVolume * 16.3871;
% weights(i, 1) = rockWeight;
% Save the 3D voxel array to disk
% save(fullfile(outFolderName, 'volume.mat'), 'volume');

if PLOT
% figure(1), imshow(top);
% figure(2), imshow(front);
% figure(3), imshow(side);
% 
% Show images on a 3D cube: https://www.mathworks.com/matlabcentral/answers/32070-rgb-images-on-a-3d-cube
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
%print('./Plot/orthogonal.png', '-r300', '-dpng');

%volume = permute(volume, [1 2 3]);
%         [Rx Ry Rz] = size(volume); % the reconstruct coordinates system used above
%         [Vx Vy Vz] = meshgrid(1:Rz, 1:Rx, 1:Ry); % rearrange the axis to Matlab plot's right-handed system
        
        % Plot 3d view   
        figure(5);
        % adjust the coordinates
        volume = flip(volume, 1); % flip x and y
        volume = flip(volume, 2);
        volume = permute(volume, [1 3 2]); % exchange y and z
        volume = padarray(volume, [2 2 2], 0, 'both'); % pad array to fill the holes when visualization
        v = double(volume);
        v = smooth3(v);
        p = patch( isosurface(v,0) );                 %# create isosurface patch
        isonormals(v, p)                              %# compute and set normals
        set(p, 'FaceColor','r', 'EdgeColor','none')   %# set surface props
        
%         hold on;
%         corner = flip(corner, 1);
%         corner = flip(corner, 2);
%         corner = permute(corner, [1 3 2]); % exchange y and z
%         v1 = double(corner);
%         v1 = smooth3(v1);
%         p1 = patch( isosurface(v1,0) );                 %# create isosurface patch
%         isonormals(v1, p1)                              %# compute and set normals
%         set(p1, 'FaceColor','y', 'EdgeColor','none')
        
        daspect([1 1 1])                              %# axes aspect ratio
        view(0, 90), axis vis3d tight, box on, grid on    %# set axes props
        set(gca, 'CameraUpVector', [0 1 0])
        set(gca, 'CameraUpVectorMode', 'manual')
        rotate3d on
        camproj perspective                           %# use perspective projection
        camlight('left'), lighting phong, alpha(1)
        % end plot 3d view
        %print('./Plot/3d.png', '-r300', '-dpng');
%         
%         surfaceVoxels = volume - imerode(volume, true(3));
%         
%         vol3d('cdata',volume,'texture','3D');
%         view(45,15);  axis tight;axis off;
%         camlight; camlight(-90,-10); camlight(180,-10);lighting phong;
%         alphamap('rampup');
%         alphamap(0.05 .* alphamap);
%         figure;
%         cmap = [0 0 1];
%         hpat = PATCH_3Darray(volume, cmap);

end

end % end of function