close all;
figure;

rgb = imread('_side.png');
rgb = imgaussfilt(rgb, 2);

hsv = rgb2hsv(rgb);
h = hsv(:,:,1);
s = hsv(:,:,2);
v = hsv(:,:,3);

s = histeq(s);
[sMag, sDir] = imgradient(s);
sMag = mat2gray(sMag);
%imshow(sMag);
[L,a,b] = rgb2lab(rgb);
L = mat2gray(L);
a = mat2gray(a);
b = mat2gray(b);

A = mean(a(:));
B = mean(b(:));
%dist = abs(a - A) + abs(b - B); % sum of absolute distance
%dist = (abs(a - A) + abs(b - B)) .* (1 + 2 * abs(L-0.5)); % sum of absolute distance
dist = (abs(a - A).^0.3 + abs(b - B).^0.3); % squared Euclidean distance
dist = mat2gray(dist);
% dist = histeq(dist);
% imshow(dist);
bw = imbinarize(dist) | (L < 0.5 * mean(L(:)));
imshowpair(bw, L < 0.5 * mean(L(:)), 'montage');
bw = imerode(bw, strel('disk', 3));
bw = imfill(bw, 'holes');
bw = imdilate(bw, strel('disk', 3));
%bw = imclearborder(bw, 8);
bw = imdilate(bw, strel('disk', 10));
bw = imerode(bw, strel('disk', 10));

imshowpair(dist, bw, 'montage');
%dist = imclearborder(dist, 8);

% dist = imgaussfilt(dist,2);
[Mag, Dir] = imgradient(dist);
Mag = mat2gray(Mag);

%imshow((Mag.^2 > 0.1);
imshow(Mag);
a = histeq(a);
b = histeq(b);
%L = histeq(L);
[LMag, LDir] = imgradient(L);

Lab = mat2gray(1.5*LMag + Mag);
mask = imbinarize(Lab);
dist = mat2gray(dist);
temp = dist;
temp(mask) = 1;
imshow(mat2gray(temp));
%im = imgaussfilt(im, 2);

imshowpair(dist, temp, 'montage');
%Lab = im > 0.1;
imshow(Lab);
%imshowpair(im, im > 0.1, 'montage');
[aMag, aDir] = imgradient(a);
[bMag, bDir] = imgradient(b);
% figure(1), imshowpair(L, LMag, 'montage');
% figure(2), imshowpair(a, aMag, 'montage');
% figure(3), imshowpair(b, bMag, 'montage');
% LMag = binarize(LMag);
% aMag = binarize(aMag);
% bMag = binarize(bMag);
% Mag = binarize(Mag);
% Lab = LMag + Mag;

Lab = imerode(Lab, strel('disk', 1));
Lab = imdilate(Lab, strel('disk', 1));
Lab = imdilate(Lab, strel('disk', 3));
Lab = imfill(Lab, 'holes');
Lab = imerode(Lab, strel('disk', 3));
imshow(Lab);
% Lab = imclearborder(Lab, 8);
Lab = imerode(Lab, strel('disk', 10));
Lab = imdilate(Lab, strel('disk', 10));
figure(6), imshow(Lab);
a = histeq(a);
b = histeq(b);
laplacian = @(img) mat2gray(img - imgaussfilt(img,5));
L_lap = laplacian(L);
a_lap = laplacian(a);
b_lap = laplacian(b);
% figure(4), imshow(L_lap);
% figure(5), imshow(a_lap);
% figure(6), imshow(b_lap);





bw = imbinarize(dist_scaled);
figure(5), imshow(bw);
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

function bw = binarize(im)
    bw = imbinarize(im);
    %bw = imopen(bw, strel('disk', 1)); % remove noises (erode + dilate)
    %bw = imdilate(bw, strel('disk', 3)); % make boundary closed
    bw = imfill(bw, 8, 'holes'); % fill holes
    %bw = imerode(bw, strel('disk', 3)); % dilate-erode pair
end

