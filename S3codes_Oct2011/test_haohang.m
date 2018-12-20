close all;

img = imread('img0001_2.png');
img = imresize(img, 0.5);
img = rgb2gray(img);
img = double(img);

img = imguidedfilter(img, 'NeighborhoodSize', 10);
[s1, s2, s3] = s3_map(img, 1);