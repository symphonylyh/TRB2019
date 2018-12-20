addpath('meanshift');
addpath('used_desc');
addpath('svmlight_mex');
addpath('utils');
addpath(genpath('UGM'));

unix('rm data/binary/* data/mask/* data/cache/* data/matting/* data/removal/* data/unary/*');
opt = {};
opt.dir = 'data/';
fnlist1 = {'DSC01536.jpg', 'DSC01611.jpg', 'DSC01508.jpg', 'DSC01603.jpg'};
fnlist2 = {'p21_1.jpg', 'p14_1.jpg', 'siebel.bmp'};
for i=1:length(fnlist1)
    opt.fn = fnlist1{i};
    opt.save = 1;
    opt.binaryClassifier = 'models/model_pair.mat';
    opt.unaryClassifier = 'models/unary_model.mat';
    opt.resize = 1;
    opt.adjecent = 0;
    opt.pairwiseonly = 0;
    opt.linearize = 0;
    h = findshadowcut_cross(opt);
end

for i=1:length(fnlist2)
    opt.fn = fnlist2{i};
    opt.save = 1;
    opt.binaryClassifier = 'models/model_pair_our.mat';
    opt.unaryClassifier = 'models/unary_model_our.mat';
    opt.resize = 1;
    opt.adjecent = 0;
    opt.pairwiseonly = 0;
    opt.linearize = 0;
    h = findshadowcut_cross(opt);
end

testfnlist = {}; cnt = 0;
for i=1:length(fnlist1)
    cnt = cnt+1;
    testfnlist{cnt} = fnlist1{i}(1:end-4);
end
for i=1:length(fnlist2)
    cnt = cnt+1;
    testfnlist{cnt} = fnlist2{i}(1:end-4);
end

runmatting;