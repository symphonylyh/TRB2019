% https://blog.csdn.net/q664111/article/details/51116378
% https://blog.csdn.net/langb2014/article/details/45620249
close all;
figure;

rgb = imread('top_1.png');

HSV = false;
LAB = true;
% HSV color space
% Hue and Saturation space can help with segmentation, but the output may
% not be satisfactory with different light condition, shadow, reflection,
% etc
if HSV
hsv = rgb2hsv(rgb);
h = hsv(:,:,1);
s = hsv(:,:,2);
% subplot(3,3,1), imshow(h), title('Hue');
% subplot(3,3,2), imshow(s), title('Sat');
% subplot(3,3,2), imhist(h(:)), title('Hue');
% Find threshold using Otsu's method (imbinarize() uses this by default)
hMask = h <= graythresh(h);
sMask = s <= graythresh(s);

subplot(3,3,4), imshow(hMask), title('Hue');
subplot(3,3,5), imshow(sMask), title('Sat');

subplot(3,3,7), imshow(hMask+sMask), title('');
subplot(3,3,8), imshow(hMask+sMask), title('');
end

if LAB
% CIE Lab color space
% Advantages: be able to work independently of luminance
% a: [-500, 500]-->[Green, Red]
% b: [-200, 200]-->[Blue, Yellow]
% The 2D a-b plane is similar to a color (or "hue") space, so the key steps
% of color segmentation are:
% 1. Find the statistics of a and b space respectively (by mean/median/mode
% which represents the "tone" of the image)
% 2. Measure the pixel distance from the "tone" and segment out the target
% object (either sum of absolute distance or squared Euclidean distance)
[L,a,b] = rgb2lab(rgb);

a = mat2gray(a);
b = mat2gray(b);
% Statistics of a & b channels (mean/median/mode)
A = mode(a(:));
B = mode(b(:));

% Calculate distance from the representative A,B values
% dist = abs(a - A) + abs(b - B); % sum of absolute distance
dist = abs(a - A).^2 + abs(b - B).^2; % squared Euclidean distance
dist_scaled = mat2gray(dist);
a = dist_scaled;

% subplot(3,2,5), imshow(dist_scaled), title('a-b Space Distance (Scaled)');
% subplot(3,2,6), hist(dist(:)), title('Histogram, distance');
% figure(2)
% subplot(3,1,1), imshow(a >= graythresh(a));
% subplot(3,1,2), imshow((a >= graythresh(a)) | (b <= graythresh(b)));
% subplot(3,1,3), imshow(dist_scaled >= graythresh(dist_scaled));
% figure(1)
%subplot(3,2,1), imshow(a), title('a Channel (Scaled)');
%subplot(3,2,2), hist(a(:), 0:1/1000:1), title('Histogram, a Channel');
%subplot(3,2,3), imshow(b), title('b Channel (Scaled)');
%subplot(3,2,4), hist(b(:), 0:1/400:1), title('Histogram, b Channel');



filt = imgaussfilt(L,5);
lap = L - filt;
lap = mat2gray(lap);

canny = edge(lap, 'canny', 0.2, 5);


figure(2)
dilate = imdilate(canny, strel('disk', 10));
%dilate = imclearborder(dilate, 8);
imshow(dilate);

figure(3)
erode = imerode(dilate, strel('disk', 10));
imshow(erode);

filt = imgaussfilt(a,5);
lap = a - filt;
lap = mat2gray(lap);
canny = edge(lap, 'canny', 0.2, 5);
dilate = imdilate(canny, strel('disk', 10));
erode1 = imerode(dilate, strel('disk', 10));

im = imfill(erode+erode1, 'holes');
%im = imclearborder(im, 8);
imshow(im);

%figure(4), imshow(a);


end






% im = imgaussfilt(im, 2);
% hueThres = graythresh(im(:,:,1)); 
% satThres = graythresh(im(:,:,2));
% % im = (abs(im(:,:,1) - 0.67) > 0.3) + (im(:,:,2) < satThres); % hue and saturation
% im = im(:,:,2) <= satThres;
% im = imfill(im, 8, 'holes');
% 
% se = strel('disk',5);
% im = imopen(im,se); %remove noise
% se = strel('disk',5);
% im = imclose(im,se); %fill small holes
% im = imclearborder(im, 8);
% se = strel('disk',5);
% im = imclose(im,se); %fill big holes
% imshow(im);