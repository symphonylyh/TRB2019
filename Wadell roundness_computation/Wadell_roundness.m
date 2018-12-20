%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was developed by
%       Junxing Zheng  (Sep. 2014)
%       University of Michigan, Ann Arbor
%       junxing@umich.edu 
%  
%  The detailed description of code is in the paper:
%  Zheng and Hryciw (2015)
%
%   If you use this code in your publication, please cite:
%   Zheng, J., and Hryciw, R.D. (2015). �Traditional Soil Particle Sphericity, 
%   Roundness and Surface Roughness by Computational Geometry�, 
%   G�otechnique, Vol. 65, No. 6, 494-506, DOI:10.1680/geot./14-P-192. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
addpath(genpath('sphericity_package'))
addpath(genpath('main_Funs'))


%% parameter settings
tol =0.3;     % forming straight lines to the bounndary
factor = 0.98; % fitting small circles
span = 0.07; % nonparametric fitting, if the boundary of particle is very rough, try a larger value;

%% main function
% img=imread('example.jpg');
% level = graythresh(img);
% im = im2bw(img,level);
% BW =~im;
img = imread('rock.png');
BW = img;
lbl = bwlabel(BW, 4);


dist_map=bwdist(~BW);
sz=size(BW);
cc = bwconncomp(BW, 8);
particles = discrete_boundary(cc);
particles = nonparametric_fitting(particles, span);


figure
imshow(img);
hold on
result = [];

for k = 1:cc.NumObjects
    obj = cc.PixelIdxList{k};
    [R, RInd]=max(dist_map(obj)); 
    [cy, cx]=ind2sub(sz, obj(RInd));
    boundary_points = particles.objects(k).cartesian;
    X = boundary_points(:, 1);
    Y = boundary_points(:, 2);

    % plot largest circle
    theta = [linspace(0,2*pi, 100)];
    plot(cos(theta)*R+cx,sin(theta)*R+cy,'color','r','LineWidth', 2);
    
    % segment the boundary of particels
    seglist = segment_boundary(X, Y, tol, 0);
    
    % concave and convex detection
    [concave, convex] = concave_convex(seglist, [cx, cy], 0);
    
    % fit small circles
    [z, r] = compute_corner_circles(sz,obj , convex, boundary_points, R, factor, 3);
    
    
    for ee = 1:length(r)
        plot(z(ee, 1),z(ee,2),...   % plot the center of circles
            z(ee, 1)  + r(ee)  * cos(theta),...
            z(ee,2)  + r(ee) * sin(theta), 'g','LineWidth', 2); 
    end
    
    Roundness = mean(r)/R;
    if Roundness > 1
        Roundness =1;
    end
    result = [result; Roundness];
      text(cx, cy, num2str(Roundness), 'Color', 'b' )
end
