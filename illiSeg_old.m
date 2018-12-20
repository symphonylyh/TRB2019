function results = illiSeg_old(filename, debug_mode)

%% Control panel
close all;
PLOT = debug_mode; % show procedural figures
PRINT = false; % save figures
BOUNDARY_ENHANCE = true; % enhance boundary information. For strong shadow, disable this; for object in shadow (dark lighting condition), enable this
HOLE_DETECTION = false; % detect holes on rock surface

%% Filter the image to remove noises
img = imread(filename);
[h,w,d] = size(img);
sigma = floor(max(h,w) / 500); % estimate filter size based on image size
% rgb = imgaussfilt(img, sigma); % @note: not as good as guided filter
% Use guided filter to do edge-preserving smoothing
% Note: the behavior of guided filter is excellent! It enhances the
% prominent edges when suppressing noises. Later in the gradient map, the
% extraction of boundary becomes eaiser.
rgb = imguidedfilter(img, 'NeighborhoodSize', 2 * sigma + 1); 

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

% Convert to CIE Lab space
[L,a,b] = rgb2lab(rgb);
 
% Scale to [0,1]
L = mat2gray(L);
a = mat2gray(a);
b = mat2gray(b);

if PLOT
    a_plot = a;
    b_plot = b;
end

% Calculate Euclidean distance from the average "tone", this step segments
% the object from the background
[Acounts, Avalues] = imhist(a);
Adistribution = cumsum(Acounts);
Apeaks = findchangepts(Adistribution, 'MaxNumChanges', 10); % old: 10
A = Avalues(Apeaks(10));
    
[Bcounts, Bvalues] = imhist(b);
Bdistribution = cumsum(Bcounts);
Bpeaks = findchangepts(Bdistribution, 'MaxNumChanges', 10);
B = Bvalues(Bpeaks(10));

dist = abs((a - A)/5).^2 + abs((b - B)/2).^2;
dist = mat2gray(dist); % scale to [0,1]

if PLOT
    dist_plot = dist;
end

% Extract the highlight portion (due to the specularity of the calibration
% ball, i.e., specular reflection) and enhance that portion in a and b
% channels (because the most illuminant part is suppressed in a and b)
highlight = imbinarize(L, 0.95); % choose the top 5% brightness as highlight
shadow = L < 0.3 & b < 0.3;%~imbinarize(L, 0.3); % choose the bottom 30% brightness as shadow
shadow_new = imguidedfilter(shadow, 'NeighborhoodSize', 2 * sigma + 1);
% figure(1), imshowpair(L, b, 'montage');
% figure(2), imshowpair(shadow, shadow_new, 'montage');
mask = b .* shadow_new;
% figure(3),imshowpair(b, mask, 'montage');
new_shadow = mask > 0 & (b .* shadow_new) < 0.3;
new_shadow = imdilate(new_shadow,strel('disk',sigma));
% figure(4),imshowpair(shadow_new,new_shadow,'montage'); 


a_plot = a;
b_plot = b;

% Extract the highlight portion and supplement it to a and b channels
% before doing boundary enhancement
a(highlight) = A; % to avoid introduce artifact, set to the average object value. 
b(highlight) = B;

% figure(1), imshowpair(a_plot, a, 'montage');
% figure(2), imshowpair(b_plot, imbinarize(b), 'montage');
% 
% dist = abs((a - A)/5).^2 + abs((b - B)/2).^2;
% dist = mat2gray(dist); % scale to [0,1]
% figure(3), imshow(dist);

if PLOT
    dist_plot = dist;
end

if BOUNDARY_ENHANCE
    % Idea: 
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
    % [LMag, LDir] = imgradient(L); % @note: shouldn't include info from L channel since it will recognize shadow area as boundary
    [aMag, aDir] = imgradient(a);
    [bMag, bDir] = imgradient(b);
    
    % Overlap the boundary information from multiple channels
    % boundary = aMag+bMag; % @note: previous version by adding. Not reasonable because a and b could cancel instead of complementing each other
    boundary = max(aMag,bMag); 
    boundary = imguidedfilter(boundary, 'NeighborhoodSize', 2 * sigma + 1); % this step is important! Guided filter is powerful...it's like double-enhancing the boundary
    boundaryMask = imbinarize(boundary);
    boundaryMask(new_shadow) = 0;
    
    % Enhance the boundary pixels in the distance map 
    dist(boundaryMask) = 0; 
end

% Binarize the distance map
bw = ~imbinarize(dist, 0.2); % old: bw = ~imbinarize(dist); the threshold can be adjusted 0.15, 0.2 or other number
bw(shadow) = 0; % can be omitted, this is used to ensure shadow edge is not recognized as objecct boundary

if PLOT
    bw_plot = bw;
end

% Morphological operations on binary image
bw = imdilate(bw, strel('disk', 2 *sigma)); % obtain a closed boundary, old: 2 * sigma
bw = imfill(bw, 4, 'holes'); % fill holes inside a connected region, 8-connected is more strict and fill fewer holes
bw = imerode(bw, strel('disk', 2 * sigma)); % imdilate's correspondence
bw = imopen(bw, strel('disk', 2 * sigma)); % open: open holes (or remove objects), erode + dilate, old: 2 * sigma
bw = bwareaopen(bw, ceil(h/100) * ceil(w/100)); % remove small object, old: ceil(h/100) * ceil(w/100)
bw = imclearborder(bw, 8); % clear meaningless regions that are connected to image border

% only for now
% figure(4);
% imshow(img);
% bd = bwperim(bw);
% hold on;
% visboundaries(bd, 'LineWidth', 1);

if PLOT
    fig = 1; % start figure No.
    figure(fig); fig = fig + 1;
    [ha, pos] = tight_subplot(1,4,[.01 .01],[.01 .01],[.01 .01]);
    axes(ha(1)), imshow(img), title('Original Image');
    axes(ha(2)), imshow(L), title('L Channel');
    axes(ha(3)), imshow(a_plot), title('a Channel');
    axes(ha(4)), imshow(b_plot), title('b Channel');
    if PRINT
        print('color space.png', '-r300', '-dpng');
    end
    
    figure(fig); fig = fig + 1;
    subplot(2,3,1), imshow(a_plot), title('a Channel');
    subplot(2,3,2), bar(Avalues, Acounts), title('Pixel histogram of a channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Pixel Count', 'FontSize', 10);
    subplot(2,3,3), plot(Avalues, Adistribution, '-r'), y1 = get(gca, 'ylim'); hold on, plot([A A], y1, '--b'), title('Pixel cdf of a channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Cumulative sum', 'FontSize', 10);

    subplot(2,3,4), imshow(b_plot), title('b Channel');
    subplot(2,3,5), bar(Bvalues, Bcounts), title('Pixel histogram of b channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Pixel Count', 'FontSize', 10);  
    subplot(2,3,6), plot(Bvalues, Bdistribution, '-r'), y2 = get(gca, 'ylim'); hold on, plot([B B], y2, '--b'), title('Pixel cdf of b channel', 'FontSize', 10), grid on, xlabel('Value', 'FontSize', 10), ylabel('Cumulative sum', 'FontSize', 10);
    tightfig;
    if PRINT
        print('histogram.png', '-r300', '-dpng');
    end
 
    if BOUNDARY_ENHANCE
        figure(fig); fig = fig + 1;
        [ha, pos] = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
        axes(ha(1)), imshow(boundary), title('Object Boundary');
        axes(ha(2)), imshow(boundaryMask), title('Boundary Mask');
        if PRINT
            print('boundary mask.png', '-r300', '-dpng');
        end
    end
    
    figure(fig); fig = fig + 1;
    [ha, pos] = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
    axes(ha(1)), imshow(dist_plot), title('Distance Map, Original');
    axes(ha(2)), imshow(dist), title('Distance Map, Boundary Enhanced');
    if PRINT
        print('distance map.png', '-r300', '-dpng');
    end
    
    figure(fig); fig = fig + 1;
    [ha, pos] = tight_subplot(1,2,[.01 .01],[.01 .01],[.01 .01]);
    axes(ha(1)), imshow(bw_plot), title('Thresholded Image, Original');
    axes(ha(2)), imshow(bw), title('Thresholded Image, Clean');
    if PRINT
        print('threshold image.png', '-r300', '-dpng');
    end
end

%% Obtain region properties and geometric features
[Label,N] = bwlabel(bw, 4); % N is the number of regions
stats = regionprops(Label, 'all'); 
allArea = [stats.Area];
allBoundingBox = [stats.BoundingBox];
allDiameter = [stats.EquivDiameter];

if N < 2
    error('Segmentation failed...');
end

% sort by the region area in descending order to distinguish ball and rock. can also use circular Hough transform to recognize ball 
[~, index] = sort(allArea, 'descend'); 
rockIdx = index(1);
ballIdx = index(2);
rockArea = allArea(rockIdx);
ballArea = allArea(ballIdx);
ballDiameter = allDiameter(ballIdx);

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
    figure(fig); fig = fig + 1;
    imshow(rockCrop), title('Cropped Object');
    if PRINT
        imwrite(rockCrop, 'timg100001.png'); % format of E-UIAIA input: timgx000N.png. x: front(0)/top(1)/side(2), N: image No.
    end
    
    figure(fig); fig = fig + 1;
    imshow(img);% , title('Boundary Detection');
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
