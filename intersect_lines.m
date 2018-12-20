function [intersects, flag] = intersect_lines(thetas,rhos,h,w)
% Cluster Hough lines and compute intersection points by cross product
% Input:
%   thetas: Nx1 theta values
%   rhos:   Nx1 rho values
% Output:
%   intersects: intersections of corners (ideally 8)
%   flag: if less than 4 corners detected, return flag = 0, otherwise = 1

% Default parameters
reference = 90; % reference line is theta = 90 (horizontal line)
window = 20; % cluster the lines with theta = 90+-20
flag = 1; % less than 4 corners detected, flag = 0, discard; flag = 1, preserved
margin = 0.15; % ignore the lines close to image boundary (within 15% of the dimension)

% Cluster lines based on theta values. Use absolute theta values.
% 0: not horizontal; 1: nearly horizontal
lines = cat(2,thetas,rhos);
cluster_num = (reference - abs(thetas)) < window;
cluster0 = lines(find(cluster_num==0),:);
cluster1 = lines(find(cluster_num~=0),:);

% Sort cluster0 from theta small to large (negative to positive, i.e. right to left)
[B,order] = sort(cluster0(:,1));
cluster0 = cluster0(order,:);

% Sometimes the 'FillGap' of houghlines() fails, so should check if lines
% in cluster0 are unique, otherwise will have greater than 8 intersects
[~,unique_ind] = unique(cluster0(:,1));
cluster0 = cluster0(unique_ind,:);

% Remove the lines that are unlikely to be in the court region (close to
% the boundary of the scene, within 20% away from height boundary)
cluster1 = cluster1(find( (h/2 - abs(abs(cluster1(:,2)) -  h/2) ) > margin * h ), :);

M = size(cluster0,1);
N = size(cluster1,1);
if M >= 2 && M <=4 && N >= 2 % if either of cluster contains less than 2 lines (i.e., not able to have at least 4 intersects or beyond 8 intersects), discard this frame
    
% Sort cluster1 from abs(rho) small to large (i.e. top to bottom)
[B,order] = sort(abs(cluster1(:,2)));
cluster1 = cluster1(order,:);
cluster1 = cluster1([1 length(cluster1)],:);

% Compute intersection with Cluster 0&1 lines using cross product
M = size(cluster0,1);
N = size(cluster1,1);
intersects = zeros(M*N,2);

for j = 1: N
    % Express in homogeneous coordinates
    trans = cluster1(j,:);
    trans_3d = houghline_3d(trans(1), trans(2));
    
    for i = 1: M
        % Cluster 0 are Longitudinal lines
        long = cluster0(i,:);
        long_3d = houghline_3d(long(1),long(2));

        sect = cross(long_3d, trans_3d);
        intersects((j-1)*M+i,:) = [sect(1)/sect(3), sect(2)/sect(3)]; % (y,x)
    end
    
end
intersects = abs(intersects); % use absolute value!

% If not enough corners are computed (< 8), should either be wrong
% detection or the scene does not contain complete court (discard then)
if length(intersects) < 4
    flag = 0;
end

else  % if either of cluster contains less than 2 lines (i.e., not able to have at least 4 intersects), discard this frame
    intersects = [0 0];
    flag = 0;
end

end

function line_3d = houghline_3d(theta,rho)
% 3D homogeneous coordinates representation of Hough line (theta, rho)
% 2D line in homogeneous coordinates: y*sin+x*cos=rho-->y*sin+x*cos-rho=0
% Note in image, y,x are opposite, so write in above form
% the line can be represented by 3D vector (sin,cos,-rho)
% take cross product of two vectors we can get the intersection (y,x,w)
% Homogeneous divide to get the (y,x) coordinates (y/w,x/w)
PI = 3.1415926;
line_3d = [sin(theta*PI/180) cos(theta*PI/180) -rho];

end


