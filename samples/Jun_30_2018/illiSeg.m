function results = illiSeg(filename, debug_mode)

%% Control panel
close all;
BOUNDARY_ENHANCE = true; % enhance boundary information
PLOT = debug_mode; % show procedural figures
PRINT = false; % save figures
HOLE_DETECTION = false; % detect holes on rock surface

%% Filter the image to remove noises and convert to CIE Lab color space
img = imread(filename);
[h,w,d] = size(img);
sigma = floor(max(h,w) / 500); % estimate filter size based on image size
% rgb = imgaussfilt(img, sigma); % @note: not as good as guided filter
% Use guided filter to do edge-preserving smoothing
% Note: the behavior of guided filter is excellent! It enhances the
% prominent edges when suppressing noises. Later in the gradient map, the
% extraction of boundary becomes eaiser.
rgb = imguidedfilter(img, 'NeighborhoodSize', 2 * sigma + 1); 

% Convert to CIE Lab space
[L,a,b] = rgb2lab(img);

% Scale to [0,1]
L = mat2gray(L);
a = mat2gray(a);
b = mat2gray(b);

%% Object Boundary Detection
if BOUNDARY_ENHANCE
% Idea:
% Color-based segmentation is prone to lose information on weak object
% boundaries, thus the segmented objects are not intact. To overcome this
% effect, an extra boundary detection step is added.

% A sharp boundary of object is required for the segmentation, so we
% should have boundary enhancement step. However, this is a color
% segmentation task rather than a mere boundary sharpening one, in which we
% also rely on the face information (i.e., the facet of the object
% because we are using the color information of it). If we just enhance
% the boundary by adding it to the original image, accordingly the facet
% information will be suppressed. There is certainly a trade-off between 
% the sharpening of object boundary and the brightening of the object face.
% In our segmentation task, unluckily, we need both because neither is 
% 100% reliable when we are in a complex natural lighting condition. 
% The face may be incomplete due to the highlights and shadows on the 
% irregular shape, while the boundary may not always be intact/closed
% even after the sharpening.

% The approach is: 
% 1) Threshold the boundary gradient image into a binary mask
% 2) Only enhance the boudnary mask area in the target image
% 3) Threshold the enhanced image
% ---------------------------------------------------------------------

% Extract object boundary based on gradient information of a & b channels
[LMag, ~] = imgradient(L);
[aMag, ~] = imgradient(a);
[bMag, ~] = imgradient(b);

% Accumulate boundary information from multiple channels
boundary = max(LMag, max(aMag,bMag)); % gather boundary information from all channels. old: max(aMag,bMag); 
boundary = imguidedfilter(boundary, 'NeighborhoodSize', 2 * sigma + 1); % this step is important! Guided filter is powerful...it's like double-enhancing the boundary
boundary = imadjust(boundary, [0 1], [0 1], 1.5); % gamma correction of gamma = 1.5 > 1 can suppress the gray noises and enhance weak boundaries. alpha > 1 curve down, alpha < 1 curve up.

% Merge boundary clues into object face information
boundaryMask = imbinarize(boundary);
boundaryMask = imdilate(boundaryMask, strel('disk', 1)); % connect discontinuous boundary
boundaryMask = imfill(boundaryMask, 4, 'holes'); % from line to face

% Idea:
% The boundary of shadow should be excluded from the above boundary map. So
% at each pixel location in the boundary mask, we check if its correspondence
% in L and b channel So first extract the non-shadow mask as a 
% restriction for the detected boundary mask.
objectMask = L > 0.3 & b > 0.3; % observation: the object (or non-shadow) area is brighter in L and b channel
boundaryMask = boundaryMask & objectMask;

% Clean the mask
boundaryMask = imerode(boundaryMask, strel('disk', 1)); % imdilate correspondence
boundaryMask = imopen(boundaryMask, strel('disk', 1)); % erode-dilate, denoise
boundaryMask = bwareaopen(boundaryMask, ceil(h/50) * ceil(w/50), 4); % remove small objects, by default 8-connectivity thus a small diamond won't be removed. change to 4
boundaryMask = imclearborder(boundaryMask, 8);

% Visualization of mask area on an image
% shadowMask = ~objectMask;
% maskColor = cat(3, ones(size(shadowMask)), ones(size(shadowMask)), zeros(size(shadowMask))); % yellow [255 255 0]
% shadowAlpha = zeros(size(shadowMask)); % alpha matrix
% shadowAlpha(shadowMask) = 0.5; % set the mask area to semi-transparent, and other area invisible
% figure(1), imshow(rgb), hold on; % base image 
% h = imshow(maskColor); % overlay the colored mask 
% set(h, 'AlphaData', shadowAlpha); % modify the transparency of the colored mask
end % end of boundary enhancement step, deliverable: boundary mask

%% Color Segmentation in HSV color space (for calibration ball)
% Note: 
% Works fine for June 29th images when the ball is facing direct, strong
% sunlight so it is very close to pure white. However, for indoor lighting
% and complex scenes this does not work perfectly. 
% Default param: S < 0.5 & V > 0.5
% For indoor: S < 0.2 & V > 0.8 or ~imbinarize(S) & imbinarize(V)
% bwareaopen() should use 4-connector!

% CIE Lab color space does not perfectly handle shadow & highlight area
% (b/c the color difference will converge to the two Lab poles for the
% brightest and darkest region.
% In HSV space, the white color is ~0 in Saturation channel & ~1 in Value
% (Illuminance) channel. With this information and the prior that the
% calibration ball is white, we can segment the calibration ball from the
% regional properties with ~0 eccentricity (circular shape).
NEW = false;
if NEW
% Convert to HSV color space
hsv = rgb2hsv(rgb);
H = hsv(:,:,1);
S = hsv(:,:,2);
V = hsv(:,:,3);

figure(1);
[ha, pos] = tight_subplot(1,4,[.01 .01],[.01 .01],[.01 .01]);
axes(ha(1)), imshow(img), title('Original Image');
axes(ha(2)), imshow(H), title('H Channel');
axes(ha(3)), imshow(S), title('S Channel');
axes(ha(4)), imshow(V), title('V Channel');

% Select white region
mask = S < 0.5 & V > 0.5;
mask = imopen(mask, strel('disk', sigma)); % remove noises
mask = bwareaopen(mask, ceil(h/50) * ceil(w/50), 4); % remove small objects 
figure(2), imshow(mask);

% Segment ball region based on eccentricity
[Label,N] = bwlabel(mask, 4); 
stats = regionprops(Label, 'Eccentricity', 'BoundingBox'); 
eccen = [stats.Eccentricity];
box = [stats.BoundingBox];
[eccen, ind] = sort(eccen, 'ascend');
ballIdx = ind(1);
ballMask = ismember(Label, ballIdx);
ballCrop = imcrop(ballMask,box(4*(ballIdx-1)+1:4*(ballIdx-1)+4));
figure(3),imshow(img), hold on, visboundaries(ballMask, 'Color', 'r');

% Get the perimeter of each object and burn it onto the raw image
ballBoundary = bwperim(ballMask, 4);
mark = imoverlay(img, ballBoundary, 'yellow');
[path, name, extension] = fileparts(filename); % path is to "Compressed folder"
path = erase(path, '/Compressed'); % direct to the upper folder

if debug_mode
%     imwrite(mark, fullfile(path, 'Segmentation/Debug/', strcat(name, '.png')));
%     imwrite(ballCrop, fullfile(path, 'Segmentation/Debug', strcat('t', name, '_ball', '.png')));
else
    imwrite(mark, fullfile(path, 'Segmentation/', strcat(name, '.png')));
    imwrite(ballCrop, fullfile(path, 'Segmentation/', strcat('t', name, '_ball', '.png')));
end

results = [0;0];
return;

end

%% Color segmentation in CIE L*a*b* color space
% Advantages: be able to work independently of luminance. In HSV space, it
% is found that a color under different illuminances cannot be 
% consistently recognized. CIE Lab space is chosen to handle the object
% recognition task by eliminating the effect of luminance on color.
% But it should be noted that the Lab space will suppress the highlight and
% shadow in the scene so we should supplement these information from the
% luminance channel.
% Range:
%   L: [0, 100] --> [Dark, Bright]
%   a: (-500, 500)-->[Green, Red]
%   b: (-200, 200)-->[Blue, Yellow]
%
% The main idea of this color segmentation is:
% The assumption is that the color of the object in the scene is distinct 
% from the background color. Meanwhile, the segmentation is expected to be 
% user-independent so that the program does not need to know what exactly the 
% background color is. This is achieved by statistics analysis on the pixel
% information.
% The 2D a-b space is similar to a color (or "tone") space. Therefore, the
% values of a and b channels are used as the metric. Suppose that the color
% with highest occurences are background color, then the more deviation a
% pixel has from the background color, the higher probability it is of 
% foreground objects. The steps are:
% 1) Find the statistics of a and b space respectively (by mean/median/mode
% which represents the overall/dominant "tone" of the image)
% 2) Measure the pixel distance from the "tone" statistics. The measurement
% can either be sum of absolute distance or squared Euclidean distance, etc.
% This distance will make the foreground object prominent.
% -------------------------------------------------------------------------

% Calculate Euclidean distance from the average "tone", this step segments
% the object from the background
[Acounts, Avalues] = imhist(a);
Adistribution = cumsum(Acounts);
Apeaks = findchangepts(Adistribution, 'MaxNumChanges', 20); 
A = Avalues(Apeaks(20));
    
[Bcounts, Bvalues] = imhist(b);
Bdistribution = cumsum(Bcounts);
Bpeaks = findchangepts(Bdistribution, 'MaxNumChanges', 40);
B = Bvalues(Bpeaks(40)); % Option: 1 if we want to distinguish from background

% figure(1);
% subplot(2,3,1), imshow(a_plot), title('a Channel');
% subplot(2,3,2), bar(Avalues, Acounts), title('Pixel histogram of a channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Pixel Count', 'FontSize', 10);
% subplot(2,3,3), plot(Avalues, Adistribution, '-r'), y1 = get(gca, 'ylim'); hold on, plot([A A], y1, '--b'), title('Pixel cdf of a channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Cumulative sum', 'FontSize', 10);
% 
% subplot(2,3,4), imshow(b_plot), title('b Channel');
% subplot(2,3,5), bar(Bvalues, Bcounts), title('Pixel histogram of b channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Pixel Count', 'FontSize', 10);  
% subplot(2,3,6), plot(Bvalues, Bdistribution, '-r'), y2 = get(gca, 'ylim'); hold on, plot([B B], y2, '--b'), title('Pixel cdf of b channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Cumulative sum', 'FontSize', 10);
% tightfig;
    
% Under outdoor lighting the objects are often lightened up, so no need for this
% correction. The following corrections are mainly for indoor condition
% when the ball & object are not sufficiently proeminent in a & b channels
% -------------------------------------------------------------------------
% Extract the highlight portion (due to the specularity of the calibration
% ball, i.e., specular reflection) and supplement it to a and b
% channels (because the most illuminant part is suppressed in a and b)
% highlight = imbinarize(L, 0.95); % choose the top 5% brightness as highlight
% a(highlight) = A; % to avoid introduce artifact, set to the average object value. 
% b(highlight) = B;
% -------------------------------------------------------------------------

% Calculate distance map
% dist = mat2gray(max(b - B, 0).^2); % zero-pass filter if we choose the background value as the reference above, i.e. B = Bvalues(Bpeaks(1));
dist = mat2gray(abs(b - B).^1.5); % distance map, don't forget to scale to [0 1]
% Previous attempts:
% dist = mat2gray(abs(b - B).^2);
% dist = mat2gray(abs((a - A)/5).^2 + abs((b - B)/2).^2); % normalize w.r.t. the different ranges in a & b space % this approach is still problematic
% dist = mat2gray(1 - abs(a - A).^2) + mat2gray(abs(b - B).^2);

% Reconstructed distance map (suppress texture & enhance contrast)
dist_erode = imerode(dist, strel('disk', 2 * sigma)); % erosion is essentially a local-minima convolution kernal, it assign the minimum pixel in the window to the center pixel. Dilation is local-maxima opeartor. This is true for both grayscale and binary image. 
dist_reconstruct = imreconstruct(dist_erode, dist); % image reconstruction is like given a seed location, and dilate many times until it is similar to the target, imreconstruct(seed, target). usually seed is got by erosion (by focusing on the highlight part), and target is usually just the original image. The result is a smoothed/denoised shape-preserving image.
dist_dilate = imdilate(dist_reconstruct, strel('disk', 2 * sigma));
dist_reconstruct = imreconstruct(imcomplement(dist_dilate), imcomplement(dist_reconstruct)); % imreconstruct works on light pixels, so should use complement image
dist_reconstruct = imcomplement(dist_reconstruct);

%figure(1), imshowpair(dist, dist_reconstruct, 'montage');

dist = dist_reconstruct;
boundaryAdd = boundaryMask & dist < 0.25;
bd = bwperim(boundaryAdd);
dist(bd) = 0; % add the enhance boundary info. Try to think a way that can select information from both boundaryMask and dist

%figure(2), imshowpair(boundaryMask, dist, 'montage');

% Adaptive thresholding the distance map
T = adaptthresh(dist, 0.3, 'ForegroundPolarity', 'dark'); % Adaptive thresholding is awesome!! 0.1 is sensitivity to distinguish background & foreground old: bw = ~imbinarize(dist, 0.1);
bw = ~imbinarize(dist, T); % adaptive threshold
% bw = ~imbinarize(dist, 0.3); % manually set threshold for individual images, usually 0.25 or 0.3
% bw = bw & objectMask; % remove over-segmented shadow area

if PLOT
    bw_plot = bw;
end
%figure(3), imshowpair(dist, bw, 'montage');

% Morphological operations on binary mask
bw = imdilate(bw, strel('disk', 2 * sigma)); % obtain a closed boundary
bw = imfill(bw, 4, 'holes'); % fill holes inside a connected region, 8-connected is more strict and fill fewer holes
bw = imerode(bw, strel('disk', 2 * sigma)); % imdilate's correspondence
bw = imopen(bw, strel('disk', 2 * sigma)); % open: open holes (or remove objects), erode + dilate
bw = bwareaopen(bw, ceil(h/50) * ceil(w/50), 4); % remove small object
bw = imclearborder(bw, 8); % clear meaningless regions that are connected to image border

% Visualize result
if PLOT
    % Figure 1, Original image and Lab channels
    fig = 1; % start figure No.
    figure(fig); fig = fig + 1;
    [ha, pos] = tight_subplot(1,4,[.01 .01],[.01 .01],[.01 .01]);
    axes(ha(1)), imshow(img), title('Original Image');
    axes(ha(2)), imshow(L), title('L Channel');
    axes(ha(3)), imshow(a), title('a Channel');
    axes(ha(4)), imshow(b), title('b Channel');
    if PRINT
        print('color space.png', '-r300', '-dpng');
    end
    
    % Figure 2, boundary map and mask
    if BOUNDARY_ENHANCE
        figure(fig); fig = fig + 1;
        [ha, pos] = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
        axes(ha(1)), imshow(boundary), title('Object Boundary');
        axes(ha(2)), imshow(boundaryMask), title('Boundary Mask');
        if PRINT
            print('boundary mask.png', '-r300', '-dpng');
        end
    end
    
    % Figure 3, pixel distribution in a and b channels
    figure(fig); fig = fig + 1;
    subplot(2,3,1), imshow(a), title('a Channel');
    subplot(2,3,2), bar(Avalues, Acounts), title('Pixel histogram of a channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Pixel Count', 'FontSize', 10);
    subplot(2,3,3), plot(Avalues, Adistribution, '-r'), y1 = get(gca, 'ylim'); hold on, plot([A A], y1, '--b'), title('Pixel cdf of a channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Cumulative sum', 'FontSize', 10);

    subplot(2,3,4), imshow(b), title('b Channel');
    subplot(2,3,5), bar(Bvalues, Bcounts), title('Pixel histogram of b channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Pixel Count', 'FontSize', 10);  
    subplot(2,3,6), plot(Bvalues, Bdistribution, '-r'), y2 = get(gca, 'ylim'); hold on, plot([B B], y2, '--b'), title('Pixel cdf of b channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Cumulative sum', 'FontSize', 10);
    tightfig;
    if PRINT
        print('histogram.png', '-r300', '-dpng');
    end
    
    % Figure 4, distance map
    figure(fig); fig = fig + 1;
    [ha, pos] = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
    axes(ha(1)), imshow(dist), title('Distance Map, Original');
    axes(ha(2)), imshow(dist), title('Distance Map, Boundary Enhanced');
    if PRINT
        print('distance map.png', '-r300', '-dpng');
    end
    
    % Figure 5, object mask
    figure(fig); fig = fig + 1;
    [ha, pos] = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
    axes(ha(1)), imshow(bw_plot), title('Thresholded Image, Original');
    axes(ha(2)), imshow(bw), title('Thresholded Image, Clean');
    if PRINT
        print('threshold image.png', '-r300', '-dpng');
    end
    
    figure(fig); fig = fig + 1;
    imshow(img);
    bd = bwperim(bw);
    hold on;
    visboundaries(bd, 'LineWidth', 1);
end

%% Obtain region properties and geometric features

% From distance map
[Label,N] = bwlabel(bw, 4); % N is the number of regions
stats = regionprops(Label, 'all'); 
allArea = [stats.Area];
allBoundingBox = [stats.BoundingBox];
allEccen = [stats.Eccentricity];
allMinAxis = [stats.MinorAxisLength];
allMaxAxis = [stats.MajorAxisLength];
allDiameter = [stats.EquivDiameter];

% From boundary mask (mainly to find the ball)
[Label_b, N_b] = bwlabel(boundaryMask, 4);
stats_b = regionprops(Label_b, 'all');
allArea_b = [stats_b.Area];
allBoundingBox_b = [stats_b.BoundingBox];
allMinAxis_b = [stats_b.MinorAxisLength];
allMaxAxis_b = [stats_b.MajorAxisLength];
allDiameter_b = [stats_b.EquivDiameter];

if N < 2
    error('Segmentation failed...');
end

% Recognize rock by area
[~, index] = sort(allArea, 'descend'); % sort by the region area in descending order to distinguish ball and rock. can also use circular Hough transform to recognize ball 
rockIdx = index(1);
rockArea = allArea(rockIdx);
rockMask = ismember(Label, rockIdx); % Get the mask of each object, and crop the object
rockCrop = imcrop(rockMask,allBoundingBox(4*(rockIdx-1)+1:4*(rockIdx-1)+4));

% Recognize ball by eccentricity (choose from distance map/boundary mask, whichever is better)
% [~, index] = sort(allEccen, 'ascend'); % for circle, eccentricity = 0
sphericity = (allMaxAxis - allMinAxis) ./ allDiameter; % (max - min) / avg, should be close to 0 for a circle
[~, ind] = sort(sphericity, 'ascend');
sphericity_b = (allMaxAxis_b - allMinAxis_b) ./ allDiameter_b; 
[~, ind_b] = sort(sphericity_b, 'ascend');

if allArea_b(ind_b(1)) < allArea(ind(1)) % if boundary mask result is better
    ballIdx = ind_b(1); 
    ballArea = allArea_b(ballIdx);
    ballDiameter = allDiameter_b(ballIdx);
    ballMask = ismember(Label_b, ballIdx);
    ballCrop = imcrop(ballMask,allBoundingBox_b(4*(ballIdx-1)+1:4*(ballIdx-1)+4));
else
    ballIdx = ind(1); 
    ballArea = allArea(ballIdx);
    ballDiameter = allDiameter(ballIdx);
    ballMask = ismember(Label, ballIdx);
    ballCrop = imcrop(ballMask,allBoundingBox(4*(ballIdx-1)+1:4*(ballIdx-1)+4));
end

% Get the perimeter of each object and burn it onto the raw image
mark = img;
rockBoundary = bwperim(rockMask, 4); 
ballBoundary = bwperim(ballMask, 4);
mark = imoverlay(mark, rockBoundary, 'red');
mark = imoverlay(mark, ballBoundary, 'yellow');
[path, name, extension] = fileparts(filename); % path is to "Compressed folder"
path = erase(path, '/Compressed'); % direct to the upper folder

if debug_mode
    imwrite(mark, fullfile(path, 'Segmentation/Debug/', strcat(name, '.png')));
    imwrite(rockCrop, fullfile(path, 'Segmentation/Debug/', strcat('t', name, '_rock', '.png')));
    imwrite(ballCrop, fullfile(path, 'Segmentation/Debug', strcat('t', name, '_ball', '.png')));
else
    imwrite(mark, fullfile(path, 'Segmentation/', strcat(name, '.png')));
    imwrite(rockCrop, fullfile(path, 'Segmentation/', strcat('t', name, '_rock', '.png')));
    imwrite(ballCrop, fullfile(path, 'Segmentation/', strcat('t', name, '_ball', '.png')));
end

% Visualize result
if PLOT
    % Figure 6, final result
    figure(fig); fig = fig + 1;
    imshow(img);
    hold on;
    visboundaries(rockMask, 'Color', 'red', 'LineStyle', '-', 'LineWidth', 1, 'EnhanceVisibility', false); 
    visboundaries(ballMask, 'Color', 'yellow', 'LineStyle', '-', 'LineWidth', 1, 'EnhanceVisibility', false);
    if PRINT
        print('final result.png', '-r300', '-dpng');
    end
end

if HOLE_DETECTION
    % Detect surface holes for volume correction
    % Ref: https://www.mathworks.com/help/images/examples/marker-controlled-watershed-segmentation.html
    [h, w] = size(rockCrop);
    radius = min(ceil(0.01 * h), ceil(0.01 * w));
    se = strel('disk', radius);

    rockMask = imerode(rockMask, strel('disk', 5 * sigma)); % avoid touching the boundary
    rock = L;
    rock(~rockMask) = 1; % Or: rock = bsxfun(@times, L, cast(rockMask, 'like', L)); % extract the rock object from L channel image
    rock = imcomplement(rock); % negative the image to highlight the area of hole shadows
    rock = histeq(rock); % contrast enhancement via histogram equalization
    surface_erode = imerode(rock, se); % erosion is essentially a local-minima convolution kernal, it assign the minimum pixel in the window to the center pixel. Dilation is local-maxima opeartor. This is true for both grayscale and binary image. 
    surface_reconstruct = imreconstruct(surface_erode, rock); % image reconstruction is like given a seed location, and dilate many times until it is similar to the target, imreconstruct(seed, target). usually seed is got by erosion (by focusing on the highlight part), and target is usually just the original image. The result is a smoothed/denoised shape-preserving image.
    surface_dilate = imdilate(surface_reconstruct, se);
    surface_reconstruct = imreconstruct(imcomplement(surface_dilate), imcomplement(surface_reconstruct)); % imreconstruct works on light pixels, so should use complement image
    surface_reconstruct = imcomplement(surface_reconstruct);
    holes = surface_reconstruct == 1; % Or: holes = imregionalmax(surface_reconstruct);
    holes = bwareaopen(holes, ceil(h/100) * ceil(w/100));
    holeArea = sum(holes(:));
%     rgb(holes) = 1;
%     imshowpair(rgb, rock, 'montage');
    holeRatio = holeArea / rockArea;
end
holeRatio = 1;
results = [holeRatio; ballDiameter]; % hole ratio of the particle & equivalent diameter of calibration ball

end % end of function

%% Learning notes:
% Good references on morphological operations:
% 1. https://blog.csdn.net/langb2014/article/details/45620249
% 2. https://blog.csdn.net/u011587361/article/details/45024087
% 3. Cell: https://blogs.mathworks.com/steve/2006/06/02/cell-segmentation/
% 4. Watershed: https://blog.csdn.net/wenyusuran/article/details/26255447
% http://www.ilovematlab.cn/thread-222540-1-1.html
% 5. Guided filter: https://blog.csdn.net/baimafujinji/article/details/74750283
% 6. Edge filter: https://blog.csdn.net/memray/article/details/51531998
%
% Binary image operations:
% bw = bwareaopen(bw, N) can remove smaller objects in a binary image that has
% fewer than N pixels
%
% [L,N] = bwlabel(bw, conn) can label each region in a binary image with 
% n=conn(4 or 8) connection. Return L the label matrix, N the number of
% regions
%
% stats = regionprops(L,'args') can take a labelled matrix and compute the
% properties of each region. stats{n} is a cell array with n = max(L(:)).
% 'args' can be 'all', 'basic' or user-defined.
% To extract a specific region under certain condition, we should first
% find the index and use ismember(L, index) function. e.g.,
% index = find([stats.Area] > 10); bw = ismember(L, index);
% 
% 
% bw = bwperim(bw, conn) can return the boundary map of a binary image. The
% boundary is designated as N=conn connection
% 
% B = imoverlay(im, bw, color) can burn the binary image into an image. Usually
% we can first get the perimeter of the binary image by bw = bwperim(bw)
% and only burn the boundary into the image
%
% bd = bwboundaries(bw) can detect the closed region in a binary image and
% return the boundary pixels of each region in a cell array bd. You can
% access the boundary by bd{i}
% 
% Multi-level thresholding: multithresh(im, N) can thresholds image into
% N+1 different parts
%
% A list of MATLAB image segmentation methods: https://www.mathworks.com/help/images/image-segmentation.html
%
% Find the inflection point in a cumulative sum distribution:
% https://www.mathworks.com/help/signal/examples/detecting-outbreaks-and-significant-changes-in-signals.html
% https://www.mathworks.com/help/signal/ref/findchangepts.html#mw_748d5529-e05a-4b55-a3fe-2a12a5772d22
