%% Setup VL_FEAT Toolbox 
% the only function used is the vl_colsubset in auto_homography.m, but is 
% merely a order-preserving random number, so replaced by sort(randperm())
% SETUP = false;
% if SETUP
% addpath vlfeat-0.9.19/
% addpath vlfeat-0.9.19/toolbox/
% vl_setup;
% end

%% Standard Court Dimension
% Standard tennis court is 78ft long, 36ft wide, we draw the court as a
% 780x360 binary image.
COURT_STD = false;
if COURT_STD
court = ones(500,1000); % Margin: 70 pixels in long, 110 pixels in wide
% 12 corner points
corners = ...
    [70,110; 
     115,110;
     385,110;
     430,110;
     
     115,290;
     385,290;
     
     115,710;
     385,710;
     
     70,890;
     115,890;
     385,890;
     430,890;
     ];
% 8 corners we typically want to detect, denoted as standard points
standards = corners([1:4 9:12],:);
% 9 boundary lines
court(70:430,110) = 0;
court(430,110:890) = 0;
court(70:430,890) = 0;
court(70,110:890) = 0;
court(115,110:890) = 0;
court(385,110:890) = 0;
court(115:385,290) = 0;
court(115:385,710) = 0;
court(250,290:710) = 0;
% ------------------------------------------------------------------------
% PLOT Court
% close all;
% imshow(court);
% hold on
% plot(corners([1:4 9:12],2), corners([1:4 9:12],1), 's', 'color', 'red');
% saveas(gcf, './court.png');
% ------------------------------------------------------------------------
% Previous dimension, too small to display the 2nd player
% court = ones(400,800); % Margin: 10 pixels in long, 20 pixels in wide
% % 12 corner points
% corners = ...
%     [20,10; 
%      65,10;
%      335,10;
%      380,10;
%      
%      65,190;
%      335,190;
%      
%      65,610;
%      335,610;
%      
%      20,790;
%      65,790;
%      335,790;
%      380,790;
%      ];
% % 8 corners we typically want to detect
% standards = corners([1:4 9:12],:);
% % 9 boundary lines
% court(20:380,10) = 0;
% court(380,10:790) = 0;
% court(20:380,790) = 0;
% court(20,10:790) = 0;
% court(65,10:790) = 0;
% court(335,10:790) = 0;
% court(65:335,190) = 0;
% court(65:335,610) = 0;
% court(200,190:610) = 0;
end

%% Court detection
% For a typical and professional tennis game broadcasting and recording,
% either indoor game or outdoor game, the camera is always placed along the
% center axis of the court. Moreover, the camera can shift and zoom yet
% slightly and tenderly. These facts made the court and player detection
% easier than other games.
% 
% Edge detector usually detect points but will lose continuous information
% on object features. Hough transform will generalize the feature by
% connecting the discrete points into lines/circles/eclipses.
tic
COURT_DETECT = false;
if COURT_DETECT
close all;
infolderName = 'original_frames'; % original sampled frames
outfolderName = 'court_frames'; % court corner detected frames
fnames = getAllFilesInFolder(infolderName);
N = length(fnames);
% Note for initialize cell, fnames{2,N} = 0 can only allocate one cell, and
% fnames{2,:} = 1 does not work, use the following way: Use () to index the
% location, and assign as {}
fnames(2,:) = {0}; % create space to store the flag of each frame: 0-discard;1-preserve
fnames(3,:) = {0}; % create space to store the quality of homography: 0-error exceeds threshold, poor; 1-error within threshold, good

court_info = struct('corners',cell(N,1),'homography',cell(N,1), 'error', cell(N,1)); % initialize the court_info structure (keep in mind, do not grow array or struct in matlab, very slow)
% structure that contains information in this step: 1. corner coordinates
% [y,x] 2. homography matrix H 3. error of the homography estimation
for i = 1:N
court0 = imread(fullfile(infolderName, fnames{1,i})); % court0 here: the original frame
im_gray = rgb2gray(court0);
im_bw = im2bw(im_gray); % convert to binary is an essential step for removing noises
im_c = edge(im_bw, 'canny');
[h,w] = size(im_c);
% Based on the fact that the court lines of a tennis court is white and
% distinguishable from the ground, converting to binary image tends to
% yield better performance before further edge detection (which can
% also be true for badminton, soccer, table tennis, etc.)
% ------------------------------------------------------------------------
% PLOT pre-processing
% figure(1)
% subplot(2,1,1), imshow(RGB), title('Original image');
% subplot(2,1,2), imshow(im_gray), title('Grayscale image');
% tightfig;
% print(gcf,'-r200','-dpng','./preprocessing_gray.png');
% figure(2)
% subplot(2,1,1), imshow(im_bw), title('Binary image');
% subplot(2,1,2), imshow(im_c), title('Canny edge detection');
% tightfig;
% print(gcf,'-r200','-dpng','./preprocessing_bw_canny.png');
% ------------------------------------------------------------------------
% Hough transform detect principle line orientations using a voting system.
% H: the voting bin.counting the apperance of each orientation
% T: theta interval. The smaller, the finer the search of orientations 
% R: rho interval. Does not matter too much
[H,T,R] = hough(im_c,'RhoResolution',2,'Theta',-90:0.5:89.5);
% Pick orientations with highest vote.
P  = houghpeaks(H,15,'threshold',ceil(0.5*max(H(:))));
% ------------------------------------------------------------------------
% PLOT Hough transform
% figure(1)
% imshow(imadjust(mat2gray(H)),'XData',T,'YData',R,...
%       'InitialMagnification','fit');
% xlabel('Angle, \theta', 'fontsize',20), ylabel('Distance, \rho', 'fontsize',20);
% axis on, axis normal;
% colormap(hot)
% hold on
% 
% x = T(P(:,2)); y = R(P(:,1));
% plot(x,y,'s','color','blue','Markersize', 10, 'linewidth',2);
% title('Hough transform statistics (Peaks labelled)','fontsize',20);
% print(gcf,'-r200','-dpng','./hough.png');
% ------------------------------------------------------------------------
% Given the identified orientations, search along that line for edge pixels
% (in binary image) and merge discrete pixels together. The following
% parameters need some adjustment based on different scene conditions.
% FillGap: if two segments are not too far away, merge them into one.
% MinLength: discard the lines which are still short even after the
% merging, so finally the lines preserved are most likely long, continuous
% edges.
% lines: return the start and end points, rho, theta of line segments.
lines = houghlines(im_c,T,R,P,'FillGap',20,'MinLength',100);
% ------------------------------------------------------------------------
% PLOT Hough lines
% figure(1)
% imshow(im_gray), axis on, hold on, title('Line segments detected (Fill gap = 20, Min Length = 100)','fontsize',10);
% set(get(gca,'title'),'Position',[640 -20 1])
% max_len = 0;
% for k = 1:length(lines)
%    % Plot line segments 
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%    % Determine the longest line segment
%    len = norm(lines(k).point1 - lines(k).point2);
%    lines(k).length = len; % add a new field 'length'
%    if ( len > max_len)
%       max_len = len;
%    end
% end
% print(gcf,'-r200','-dpng','./houghlines.png');
% ------------------------------------------------------------------------
% The idea is: from the perspective of the camera location in a ball game,
% the rectangular court will always display ~two clusters of straight
% lines, converging to vanishing point VPx and VPy. Based on these
% perspective cues, I cluster the detected lines by the distance of their
% vanishing points. If lines have vanishing points at finite distance (near
% convergence)-->Cluster 0; if lines have vanishing points at approximately
% infinity (far convergence)-->Cluster 1. The 'theta' value indicates the 
% orientaion of the line, thus lines can be clustered based on similar 
% theta values. Then:
% (1) Sort and pick the two boundary edges from Cluster 1 based on their 
% 'rho' values.
% (2) For each member in Cluster 0, compute the cross product with the
% above two boundary edges to get intersection points at corners.
% (3) For tennis court, typically 8 corners or more are expected from the
% algorithm. If much less than 8, the frame might not contain complete
% court information and can be discarded from the repository.
% Note:
% (1) In tennis game video, horizontal lines can usually be clustered
% together, therefore a threshold (5 deg in this case) is set to find those
% lines with theta = 90+-5 deg. For more general cases such as camera at
% the diagnoal directions, K-mean algorithm or others can be used.
% (2) Hough representation of line in image plane: 
%     y*sin(theta) + x*cos(theta) = rho
% Thus a line in homogeneous coordinates is:
%     [sin(theta), cos(theta), rho]
% Cross product of two lines yields [y,x,w], then the coordinates of
% intersection point in image plane is:
%     [y/w, x/w]

thetas = [lines(:).theta]'; % syntax of getting field from struct
rhos = [lines(:).rho]';
[intersects, flag] = intersect_lines(thetas, rhos, h, w); % also send in height and width of image to eliminate some artifact edges close to the image boundary
fnames{2,i} = flag;

% Add marker without plotting
if flag % for valid (preserved) frames only
    [H,quality,H_error] = auto_homography(intersects, standards); % compute homography from RANSAC
    fnames{3,i} = quality; % quality of homography: 0-poor; 1-good
    % ------------------------------------------------------------------------
    % OVERWRITE markers on image
    vertices = [intersects(:,2) intersects(:,1)];
    polygon = reshape(vertices.',1,[]); % to insert polygon shape, the order should be [x1 y1 x2 y2 ...] so first transpose and then reshape
    court0 = insertMarker(court0, vertices,'s','color','red','Size',5);
    imwrite(court0,sprintf('court_frames/%04d.png',i));
else % to prevent randomly allocated variable in court_info file
    intersects = [0 0];
    H = zeros(3);
end

court_info(i).corners = intersects;
court_info(i).homography = H;
court_info(i).error = H_error;

end
toc

disp(['Good Homography Estimation: ', num2str(sum(cell2mat(fnames(3,:)))), ' out of ', num2str(N)]); % count numbers of good homography estimation
save('court_info.mat','court_info'); % save 1xN non-scalar struct 'court_info' to matlab file. Read and Indexing: S = load('court_info.mat'); S.court_info(i).corners/homography;
save('frames_info.mat', 'fnames'); % save 3xN cell 'fnames' to matlab file: row1-framename 'xxxx.png'; row2-flag of frame discarded/preserved; row3-flag of homography quality poor/good

end

%% Pick out preserved frames
% Pick out the preserved frames to distinguish between discarded and
% preserved frames
PICK = false;
if PICK
F = load('frames_info.mat');
fnames_court = F.fnames;
quality = [fnames_court{3,:}];
infolderName = 'court_frames';
preserved = find(flag & quality);
for i = preserved
movefile(fullfile(infolderName, fnames{1,i}),'./preserved_frames');
end
end

%% Manually player detection
% Manually obtain player positions, as a comparison to the trained model
% detection results
MANUAL = false;
if MANUAL
close all;
infolderName = 'original_frames';

fnames = getAllFilesInFolder(infolderName);
N = length(fnames);

fnames(2,:) = {[]}; % initialize 
positions = zeros(2,2); % position in [y x]
Box = zeros(2,4); % initialize the draw box

for i = 1:N
    
    im = imread(fullfile(infolderName, fnames{1,i}));
    disp(fnames{1,i})
    imshow(im), axis on;
    disp('Select the first player:')
    [x1, y1, z1] = ginput(2);
    disp('Select the second player:')
    [x2, y2, z2] = ginput(2);
    
    positions(1,:) = [max(y1) mean(x1)];
    positions(2,:) = [max(y2) mean(x2)];
    fnames{2,i} = [positions ones(2,1)];

    Box(1,:) = [x1(1) y1(1) x1(2)-x1(1) y1(2)-y1(1)];
    Box(2,:) = [x2(1) y2(1) x2(2)-x2(1) y2(2)-y2(1)];
    
    % Show the box for user to decide if he did a good job
    hold on;
    rectangle('Position', Box(1,:),'EdgeColor','green', 'LineWidth', 5);
    rectangle('Position', Box(2,:),'EdgeColor','green', 'LineWidth', 5);
    
    flag = 1;
    prompt = 'Failed? [0-succeed/1-fail]:';
    flag = input(prompt);
    
    if flag 
        disp('Redo:')
        disp('Select the first player:')
        [x1, y1, z1] = ginput(2);
        disp('Select the second player:')
        [x2, y2, z2] = ginput(2);
        positions(1,:) = [max(y1) mean(x1)];
        positions(2,:) = [max(y2) mean(x2)];
        fnames{2,i} = [positions ones(2,1)];

        Box(1,:) = [x1(1) y1(1) x1(2)-x1(1) y1(2)-y1(1)];
        Box(2,:) = [x2(1) y2(1) x2(2)-x2(1) y2(2)-y2(1)];
        
        im = insertShape(im, 'Rectangle',Box,'color','green','LineWidth', 5);        
        
    else
        im = insertShape(im, 'Rectangle',Box,'color','green','LineWidth', 5);
    end
    imwrite(im,sprintf('manual_frames/%04d.png',i));
end

save('manual_player.mat', 'fnames'); % query by M = load('manual_player.mat'); M.fnames(2,:)

end

%% Manual detected player projection and visualization
MANUAL_RESULT = false;
if MANUAL_RESULT
C = load('court_info.mat'); % court detection: struct: .corners, corner [y x]; .homography, 3x3 matrix
F = load('frames_info.mat'); % court detection: 3xN cell: frame name; flag:0-discarded,1-preserved; flag: 0-poor quality,1-good quality
P = load('manual_player.mat'); % player detection: 2xN cell: frame name; positions of player [y1 x1 1; y2 x2 1]
fnames_court = F.fnames;
fnames_player = P.fnames;
court_info = C.court_info;
infolderName = 'manual_frames';
fnames = getAllFilesInFolder(infolderName);
N = length(fnames);

court0 = court;
distance1 = 0; % top player
distance2 = 0; % bottom player
heatmap = zeros(size(court));
pad1 = 70;
pad2 = 220;
heatmap = padarray(heatmap, [pad1 pad2]);

[h1,w1,d1] = size(imread(fullfile(infolderName, fnames{1,1})));
[h2,w2,d2] = size(court);

flag = [fnames_court{2,:}];
quality = [fnames_court{3,:}];

preserved = find(flag & quality);

distance1_list = zeros(2,length(preserved));
distance1_list(1,:) = preserved;
distance2_list = zeros(2,length(preserved));  % keep a track list of distance
distance2_list(1,:) = preserved;
count = 1;

for i = preserved
% ------------------------------------------------------------------------
% Image compositing
player = imread(fullfile(infolderName, fnames{1,i}));
player_n_court = insertShape(player, 'FilledCircle', [court_info(i).corners(:,2) court_info(i).corners(:,1) 7*ones(length(court_info(i).corners),1)], 'Color','red', 'Linewidth',2, 'Opacity', 0.5); % composite detected players & court corners; [y x] to [x y]
% player_n_court = insertMarker(player, [court_info(i).corners(:,2) court_info(i).corners(:,1)], 'square','color','red', 'size',5);
% imwrite(player_n_court,sprintf('combined_frames/%04d.png',i));
% ------------------------------------------------------------------------
% Homography projection
H = court_info(i).homography;
pos = fnames_player{2,i};
projected = H * pos';
projected = [projected(2,:)./projected(3,:);projected(1,:)./projected(3,:)]'; % to [x y]
court0 = insertShape(court0, 'FilledCircle', [projected 5*ones(size(projected,1),1)],'color','blue', 'LineWidth', 2,'Opacity',0.8);
% ------------------------------------------------------------------------
% DRAW trajectory
if i > 1
    court0 = insertShape(court0,'Line',[pos_prev projected],'Color','red'); % draw trajectory from previous position to current
    distance1 = distance1 + pdist([pos_prev(1,:);projected(1,:)], 'Euclidean');
    distance2 = distance2 + pdist([pos_prev(2,:);projected(2,:)], 'Euclidean');
    
    distance1_list(2,count) = distance1/10;
    distance2_list(2,count) = distance2/10;
end
count = count + 1;
pos_prev = projected;
court_text = court0; % insert text will always overlap, so make a copy
% ------------------------------------------------------------------------
% DISP distance
text_string = cell(2,1);
distance = [distance1 distance2];
for k = 1:2
    text_string{k} = ['Player ', num2str(k), ' trajectory distance: ', num2str(distance(k)/10,'%0.2f'), ' ft']; % 1ft = 10 pixel, so report the distance in ft
end
court_text = insertText(court_text, [110 40; 580 40], text_string, 'FontSize', 18,'BoxColor', {'red', 'blue'}, 'BoxOpacity', 0.4);
%imwrite(court_text,sprintf('projected_frames/%04d.png',i));
% ------------------------------------------------------------------------
% PLOT heat map
% search if there is better tool for draw heat map
heatmap(round(projected(1,2))+pad1-10:round(projected(1,2))+pad1+10, round(projected(1,1))+pad2-10:round(projected(1,1))+pad2+10)  = heatmap(round(projected(1,2))+pad1, round(projected(1,1))+pad2) + 1;
heatmap(round(projected(2,2))+pad1-10:round(projected(2,2))+pad1+10, round(projected(2,1))+pad2-10:round(projected(2,1))+pad2+10)  = heatmap(round(projected(2,2))+pad1, round(projected(2,1))+pad2) + 1;
% ------------------------------------------------------------------------
% Stack two images together
court_text = padarray(court_text,[(h1-h2)/2 (w1-w2)/2]); % padarray to make the same size
composite = cat(1, im2double(player_n_court), court_text); % use cat to stack together
%imwrite(composite,sprintf('stacked_frames/%04d.png',i));
% ------------------------------------------------------------------------
end
heatmap = mat2gray(heatmap);
% 9 boundary lines
heatmap(70+pad1:430+pad1,110+pad2) = 1;
heatmap(430+pad1,110+pad2:890+pad2) = 1;
heatmap(70+pad1:430+pad1,890+pad2) = 1;
heatmap(70+pad1,110+pad2:890+pad2) = 1;
heatmap(115+pad1,110+pad2:890+pad2) = 1;
heatmap(385+pad1,110+pad2:890+pad2) = 1;
heatmap(115+pad1:385+pad1,290+pad2) = 1;
heatmap(115+pad1:385+pad1,710+pad2) = 1;
heatmap(250+pad1,290+pad2:710+pad2) = 1;
%imshow(heatmap); axis normal;axis on; colormap(hot); colorbar;title('Heat map of the player motion');
%saveas(gcf, './heatmap.png');
end

%% Verify 2D player projection via homography matrix
PROJECT = false;
if PROJECT
close all;
im = imread('./combined_frames/0001.png');
S = load('court_info.mat');
court_info = S.court_info;
H = court_info(1).homography;
im = imrotate(im,90);
T = maketform('projective',H');
imt = imtransform(im, T, 'XData',[-100 500],'YData',[-200 1000]);
imt = imrotate(imt,90);
imshow(imt); title('Projected 2D frame', 'fontsize',15);
set(get(gca,'title'),'Position',[600 -10 1]);
print(gcf, '-r200', '-dpng','./results/projected2d.png');
end

%% Read and parse detected player positions file by Wil
PARSE = false;
if PARSE
fileID = fopen('player_coordinates.txt');
positions = textscan(fileID, '%4d.%s %d %d');
frame_index = positions{1};
y = positions{3};
x = positions{4};

M = 700;
fnames = cell(4,M); % follow the same format as previous convention
fnames(2,:) = {ones(2,3)}; % 2rd row: 2x3 matrix of two player coordinates [y x 1], [1 1 1] by default to prevent singularity issue in following steps
fnames(3,:) = {0}; % 3th row: flag:0-target not detected;1-first player (upper court) detected
fnames(4,:) = {0}; % 4th row: flag:0-target not detected;1-second player (lower court) detected

previous = 0;
N = length(frame_index);

for i = 1:N
    index = frame_index(i);
    if index == previous % for several detections, in Wil's file, it just duplicates the filename
        if y(i) < 280 
            pos = [y(i) x(i) 1];
            fnames{2,index}(1,:) = double(pos); 
            fnames{3,index} = 1; 
        else
            if y(i) > 380
            pos = [y(i) x(i) 1];
            fnames{2,index}(2,:) = double(pos);
            fnames{4,index} = 1;
            end
        end  
    else 
        if y(i) < 280 % upper court player 330-50
            fnames{1,index} = index;
            pos = [y(i) x(i) 1];
            fnames{2,index}(1,:) = double(pos);
            fnames{3,index} = 1; % change flag to 1
        else
            if y(i) > 380 % lower part player 330+50
            fnames{1,index} = index;
            pos = [y(i) x(i) 1];
            fnames{2,index}(2,:) = double(pos);
            fnames{4,index} = 1;
            end
        end
    end
    previous = index;
end
fclose(fileID);
save('detected_player.mat', 'fnames');
end

%% Trained player detection, projection, and visualization
TRAIN_RESULT = true;
if TRAIN_RESULT
C = load('court_info.mat'); % court detection: struct: .corners, corner [y x]; .homography, 3x3 matrix
F = load('frames_info.mat'); % court detection: 3xN cell: frame name; flag:0-discarded,1-preserved; flag: 0-poor quality,1-good quality
P = load('detected_player.mat'); % player detection: 2xN cell: frame name; positions of player [y1 x1 1; y2 x2 1]
fnames_court = F.fnames;
fnames_player = P.fnames;
court_info = C.court_info;
infolderName = 'player_frames';
fnames = getAllFilesInFolder(infolderName);
N = length(fnames);

first1 = 0;
first2 = 0;

court1 = court;
court2 = court;

court_text = court;
court1_copy = court1;
court2_copy = court2;
text_string1 = ['Player 1 trajectory distance: ', num2str(0,'%0.2f'), ' ft'];
text_string2 = ['Player 2 trajectory distance: ', num2str(0,'%0.2f'), ' ft'];
distance1 = 0; % top player
distance2 = 0; % bottom player
heatmap = zeros(size(court));
pad1 = 70;
pad2 = 220;
heatmap = padarray(heatmap, [pad1 pad2]);

[h1,w1,d1] = size(imread(fullfile(infolderName, fnames{1,1})));
[h2,w2,d2] = size(court);

flag = [fnames_court{2,:}];
quality = [fnames_court{3,:}];
player1 = [fnames_player{3,:}];
player2 = [fnames_player{4,:}];

preserved = find(flag & quality & (player1 | player2));

distance1_list = zeros(2,length(find(flag & quality & player1)));
distance2_list = zeros(2,length(find(flag & quality & player2)));  % keep a track list of distance in each frame
count1 = 1;
count2 = 1;

for i = preserved
% ------------------------------------------------------------------------
% Image compositing
player = imread(fullfile(infolderName, fnames{1,i}));
player_n_court = insertShape(player, 'FilledCircle', [court_info(i).corners(:,2) court_info(i).corners(:,1) 7*ones(length(court_info(i).corners),1)], 'Color','red', 'Linewidth',2, 'Opacity', 0.5); % composite detected players & court corners; [y x] to [x y]
% imwrite(player_n_court,sprintf('combined_frames_train/%04d.png',i));
% ------------------------------------------------------------------------
% Homography projection
H = court_info(i).homography;
pos = fnames_player{2,i};
projected = H * pos';
projected = [projected(2,:)./projected(3,:);projected(1,:)./projected(3,:)]'; % to [x y]
if player1(i)
    if first1 == 0 % record the index when player 1 was initially tracked
        first1 = i;
    end
    if player2(i) == 0 % if no player2 detected, use the previous player2 background at a starting point and plot on that
        court1 = court2_copy;
    end
    court1 = insertShape(court1, 'FilledCircle', [projected(1,:) 5],'color','blue', 'LineWidth', 2,'Opacity',0.8);
    % ------------------------------------------------------------------------
    % DRAW trajectory for player 1
    if i > first1
        court1 = insertShape(court1,'Line',[pos_prev1 projected(1,:)],'Color','red'); % draw trajectory from previous position to current
        distance1 = distance1 + pdist([pos_prev1;projected(1,:)], 'Euclidean');
        
        distance1_list(1,count1) = i;
        distance1_list(2,count1) = distance1/10;
        count1 = count1 + 1;
    end
    pos_prev1 = projected(1,:);
    % ------------------------------------------------------------------------
    % DISP distance for player 1
    text_string1 = ['Player 1 trajectory distance: ', num2str(distance1/10,'%0.2f'), ' ft']; % 1ft = 10 pixel, so report the distance in ft
    % ------------------------------------------------------------------------
    % ADD distance tracking
    court1_copy = court1;
    court_text1 = court1_copy;
    court_text1 = insertText(court_text1, [110 40], text_string1, 'FontSize', 18,'BoxColor', 'red', 'BoxOpacity', 0.4);
    if player2(i) == 0 % preserve the info from last step if the player is not detected in this frame
        court_text1 = insertText(court_text1, [580 40], text_string2, 'FontSize', 18,'BoxColor', 'blue', 'BoxOpacity', 0.4);
        court_text = court_text1;
        court2_copy = court1_copy;
        %imwrite(court_text1,sprintf('projected_frames_train/%04d.png',i));
    else
        court2 = court1_copy; % if player2 is also detected, then draw on top of this player 1 drawings
    end
    % ------------------------------------------------------------------------
    % PLOT heat map
    heatmap(round(projected(1,2))+pad1-10:round(projected(1,2))+pad1+10, round(projected(1,1))+pad2-10:round(projected(1,1))+pad2+10)  = heatmap(round(projected(1,2))+pad1, round(projected(1,1))+pad2) + 1; % [x y]
    % ------------------------------------------------------------------------
end
if player2(i)
    if first2 == 0 % record the index when player 2 was first tracked
        first2 = i;
    end
    if player1(i) == 0
        court2 = court1_copy;
    end
    court2 = insertShape(court2, 'FilledCircle', [projected(2,:) 5],'color','blue', 'LineWidth', 2,'Opacity',0.8);
    % ------------------------------------------------------------------------
    % DRAW trajectory
    if i > first2
        court2 = insertShape(court2,'Line',[pos_prev2 projected(2,:)],'Color','red'); % draw trajectory from previous position to current
        distance2 = distance2 + pdist([pos_prev2;projected(2,:)], 'Euclidean');
        
        distance2_list(1,count2) = i;
        distance2_list(2,count2) = distance2/10;
        count2 = count2 + 1;
    end
    pos_prev2 = projected(2,:);
    % ------------------------------------------------------------------------
    % DISP distance for player 2
    text_string2 = ['Player 2 trajectory distance: ', num2str(distance2/10,'%0.2f'), ' ft']; % 1ft = 10 pixel, so report the distance in ft
    % ------------------------------------------------------------------------
    court2_copy = court2;
    court_text2 = court2_copy;
    court_text2 = insertText(court_text2, [580 40], text_string2, 'FontSize', 18,'BoxColor', 'blue', 'BoxOpacity', 0.4);
    
    court_text2 = insertText(court_text2, [110 40], text_string1, 'FontSize', 18,'BoxColor', 'red', 'BoxOpacity', 0.4);
    court_text = court_text2;
    court1 = court2_copy;
    court1_copy = court2_copy;

    %imwrite(court_text2,sprintf('projected_frames_train/%04d.png',i));
    % ------------------------------------------------------------------------
    % PLOT heat map
    heatmap(round(projected(2,2))+pad1-10:round(projected(2,2))+pad1+10, round(projected(2,1))+pad2-10:round(projected(2,1))+pad2+10)  = heatmap(round(projected(2,2))+pad1, round(projected(2,1))+pad2) + 1;
    % ------------------------------------------------------------------------
end
% imwrite(court_text,sprintf('projected_frames_train/%04d.png',i));
% ------------------------------------------------------------------------
% Stack two images together
court_composite = padarray(court_text,[(h1-h2)/2 (w1-w2)/2]); % padarray to make the same size
composite = cat(1, im2double(player_n_court), court_composite); % use cat to stack together
% imwrite(composite,sprintf('stack_frames_train/%04d.png',i));
end
heatmap = mat2gray(heatmap);
% 9 boundary lines
heatmap(70+pad1:430+pad1,110+pad2) = 1;
heatmap(430+pad1,110+pad2:890+pad2) = 1;
heatmap(70+pad1:430+pad1,890+pad2) = 1;
heatmap(70+pad1,110+pad2:890+pad2) = 1;
heatmap(115+pad1,110+pad2:890+pad2) = 1;
heatmap(385+pad1,110+pad2:890+pad2) = 1;
heatmap(115+pad1:385+pad1,290+pad2) = 1;
heatmap(115+pad1:385+pad1,710+pad2) = 1;
heatmap(250+pad1,290+pad2:710+pad2) = 1;
% imshow(heatmap); axis normal;axis on; colormap(hot); colorbar;title('Heat map of the player motion');
% saveas(gcf, './heatmap_train.png');
end

%% Create videos
VIDEO = false;
if VIDEO
    imageFolder2mpeg('court_frames','frameRate', 5, 'movieFname', 'court_detected.mp4');
    
    imageFolder2mpeg('projected_frames','frameRate', 5, 'movieFname', 'trajectory.mp4');
    imageFolder2mpeg('combined_frames','frameRate', 5, 'movieFname', 'composite.mp4');
    imageFolder2mpeg('stacked_frames','frameRate', 5, 'movieFname', 'stacked.mp4');
    
    imageFolder2mpeg('projected_frames_train','frameRate', 5, 'movieFname', 'trajectory_train.mp4');
    imageFolder2mpeg('combined_frames_train','frameRate', 5, 'movieFname', 'composite_train.mp4');
    imageFolder2mpeg('stack_frames_train','frameRate', 5, 'movieFname', 'stacked_train.mp4');
end
