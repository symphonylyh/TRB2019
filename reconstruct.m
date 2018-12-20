% Ref:
% Watershed: https://www.mathworks.com/help/images/examples/marker-controlled-watershed-segmentation.html
% test.m
% Color thresholding: https://www.mathworks.com/matlabcentral/fileexchange/25682-color-threshold
% regionprops: https://wenku.baidu.com/view/3af583c108a1284ac850432d.html
% opencv thresholding: https://blog.csdn.net/guduruyu/article/details/68059450
% http://www.cnblogs.com/polly333/p/7284223.html
% 3D reconstruct: http://blog.sciencenet.cn/blog-4099-1059118.html
% https://blog.csdn.net/bizer_csdn/article/details/52712965
% Delaunay triangulation: https://blog.csdn.net/berguiliu/article/details/25003055
% Centroid of polyhedron: http://wwwf.imperial.ac.uk/~rn/centroid.pdf
% https://www.mathworks.com/matlabcentral/fileexchange/8514-centroid-of-a-convex-n-dimensional-polyhedron
% see Blobsdemo.m for other implementation
% Extract from green backdrop: 
% https://blogs.mathworks.com/steve/2014/08/12/it-aint-easy-seeing-green-unless-you-have-matlab/

%% Image Segmentation
global ZJU; ZJU = false;
global UIUC; UIUC = true;

% ZJU type image (white background, dark particle), simple binarization is
% sufficient
if ZJU
    close all;
    front = imread('front.bmp');
    side = imread('side.bmp');
    top = imread('top.bmp');
    front_bi = toBinary(rgb2gray(front)); % use grayscale
    side_bi = toBinary(rgb2gray(side));
    top_bi = toBinary(rgb2gray(top));
end

% UIUC type image (blue background, bright particle), color segmentation is
% required
if UIUC
    close all;
    front = imread('front_1.jpeg');
    side = imread('side_1.jpeg');
    top = imread('top_1.jpeg');
    front_bi = toBinary(rgb2hsv(front)); % use HSV space
    side_bi = toBinary(rgb2hsv(side));
    top_bi = toBinary(rgb2hsv(top));
end

figure
[ha, pos] = tight_subplot(2,3,[.01 .01],[.01 .01],[.01 .01]);
axes(ha(1)), imshow(front), title('front'); 
axes(ha(2)), imshow(side), title('side');
axes(ha(3)), imshow(top), title('top');
axes(ha(4)), imshow(front_bi), title('front');
axes(ha(5)), imshow(side_bi), title('side');
axes(ha(6)), imshow(top_bi), title('top');

textContent = {'Original images', 'Binary images'};
textPos = {[0 0.9 1 0.1], [0 0.4 1 0.1]}; % relative location w.r.t to the figure
for i = 1:2
    annotation('textbox', textPos{i}, ... 
               'String', textContent{i}, ...
               'EdgeColor', 'none', ...
               'HorizontalAlignment', 'center',...
               'FitBoxToText','on',...
               'FontWeight','bold') 
end

print('UIUC_image.png', '-r300', '-dpng');

%% toBinary()
% @param im The input image
% @return The desired binary image
function im = toBinary(im)

global ZJU
global UIUC

if ZJU
    im = imgaussfilt(im, 2); % Gaussian filtering with sigma = 2
    im = imbinarize(im,0.7); % Binarize with threshold = 0.7. Illuminance > threshod, true; otherwise, false
    im = imclearborder(~im, 8); % Only take the object that is in center of the image
    im = imfill(im, 8, 'holes'); % Fill holes in the mask. Note the object (particle) is always darker than background, use ~ (NOT) operator
end

if UIUC
    im = imgaussfilt(im, 2);
    hueThres = graythresh(im(:,:,1)); % To obtain the best threshold value. https://www.mathworks.com/help/images/ref/graythresh.html
    satThres = graythresh(im(:,:,2));
    % im = (abs(im(:,:,1) - 0.67) > 0.3) + (im(:,:,2) < satThres); % hue and saturation
    im = im(:,:,2) <= satThres;
    im = imfill(im, 8, 'holes');
    
    se = strel('disk',10);
    im = imopen(im,se); %remove noise
    se = strel('disk',10);
    im = imclose(im,se); %fill small holes
    im = imclearborder(im, 8);
    se = strel('disk',20);
    im = imclose(im,se); %fill big holes
end
      
end

% Notes:
% im = imbinarize(im,'adaptive','ForegroundPolarity','dark','Sensitivity',0.6);
% imreconstruct()

%cc = imconncomp(im, 8);

% [h, w] = size(im);
% seed = false(h,w);
% seed(floor(h/2), floor(w/2)) = true;
% W = gradientweight(im, 1.0);
% thresh = 0.1;
% [im, D] = imsegfmm(W, seed, thresh);
% imshow(im)

% bm = boundarymask(front);
% imshow(bm);

