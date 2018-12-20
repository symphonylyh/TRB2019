%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was developed by
%       Junxing Zheng  (Sep. 2014)
%       University of Michigan, Ann Arbor
%       junxing@umich.edu 
%  
%  The detailed description of code is in the paper:
%  Zheng and Hryciw (2015). Traditional Soil Particle Sphericity, 
%   Roundness and Surface Roughness by Computational Geometry�, 
%   G�otechnique, Vol. 65, No. 6, 494-506, DOI:10.1680/geot./14-P-192. 
%
%   If you use this code in your publication, please cite above paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  Updates:
%  5/30/2017: remove LOESS algorithm, using filtering method to smooth
%  particle boundary
%  7/17/2017: (1) add imerode and imdilate to better process raw images 
%  (2) add algorithm to remove small objects
%  
%

clear all
close all
addpath(genpath('main_Funs'))


%% parameter settings
tol =0.3;     % forming straight lines to the bounndary
factor = 0.98; % fitting small circles
toRemove = 200; % remove the small noise points having primeter smaller than 200 pixels
smoothParameter = 30; % smooth parameter in filter method


%% main function
% img=imread('example.jpg');
% level = graythresh(img);
% im = im2bw(img,level);
% BW =~im;

fnames = getAllFilesInFolder('./sample');
allRoundness = [];
for hhh = 1 : length(fnames)
    img = imread(strcat('./sample/', fnames{hhh}));
    BW = img;

for i = 1:5
    SE = strel('disk',10);
    BW = imerode (BW,SE); %imerode
    BW = imdilate(BW,SE);   %imdilate
end


lbl = bwlabel(BW, 4);

dist_map=bwdist(~BW);
sz=size(BW);
particles = discrete_boundary(BW, toRemove);
particles = boundary_smoothing(particles, smoothParameter);




% figure(1)
% imshow(img, []);
% hold on
result = [];

for k = 1:particles.NumObjects
    obj = particles.objects(k).PixelIdxList;
    [R, RInd]=max(dist_map(obj)); 
    [cy, cx]=ind2sub(sz, obj(RInd));
    boundary_points = particles.objects(k).cartesian;
    X = boundary_points(:, 1);
    Y = boundary_points(:, 2);
    %plot(X,Y,'k','LineWidth',1.5);
    % plot largest circle
    theta = [linspace(0,2*pi, 100)];
    %plot(cos(theta)*R+cx,sin(theta)*R+cy,'color','r','LineWidth', 1.5);
    
    % segment the boundary of particels
    seglist = segment_boundary(X, Y, tol, 0);
    
    % concave and convex detection
    [concave, convex] = concave_convex(seglist, [cx, cy], 0);
    
    % fit small circles
    [z, r] = compute_corner_circles(sz,obj , convex, boundary_points, R, factor, 3);
    
    
%     for ee = 1:length(r)
%         plot(z(ee, 1),z(ee,2),...   % plot the center of circles
%             z(ee, 1)  + r(ee)  * cos(theta),...
%             z(ee,2)  + r(ee) * sin(theta), 'g','LineWidth', 1.5); 
%     end
    
    Roundness = mean(r)/R;
    if Roundness > 1
        Roundness =1;
    end
 
    %text(cx, cy, num2str(Roundness), 'Color', 'b', 'FontSize', 15 )
      
      
      
 %% Sphericity computation
        [~,rcum] = min_circum_circle(X,Y);
        sphericity1 = particles.objects(k).area/(pi*rcum^2);  % area sphericity
        sphericity2 = sqrt( particles.objects(k).area/ pi)/rcum;   % diameter sphericity
        sphericity3 = R/rcum;   % circle ratio sphericity
        sphericity4 = 2*sqrt(pi*particles.objects(k).area)/particles.objects(k).perimeter; % perimeter sphericity
        sphericity5 = particles.objects(k).d1d2(2)/particles.objects(k).d1d2(1); % width to length ratio spehricity
        
 %% summary of results 
        result = [result; Roundness, sphericity1, sphericity2, sphericity3, sphericity4, sphericity5];
      
end

allRoundness(hhh) = Roundness;
csvwrite('roundess.csv', allRoundness');
end