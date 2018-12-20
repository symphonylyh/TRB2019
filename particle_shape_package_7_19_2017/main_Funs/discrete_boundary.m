function obj = discrete_boundary(im, toRemove)

[contour,~] = bwboundaries(im,'noholes');
s  = regionprops(im, 'centroid', 'Area','Perimeter','MajorAxisLength',...
    'MinorAxisLength', 'PixelIdxList');

indroRemove = [s.Perimeter]' < toRemove;
s(indroRemove) = [];
contour(indroRemove) =[];


for k = 1:length(s)
        
    X=contour{k}(:,2);
    Y=contour{k}(:,1);
    obj.objects(k).centroid = s(k).Centroid;
    obj.objects(k).area = s(k).Area;
    obj.objects(k).perimeter = s(k).Perimeter;
    obj.objects(k).rawXY = [X, Y];
    obj.objects(k).d1d2 = [s(k).MajorAxisLength, s(k).MinorAxisLength];
    obj.objects(k).PixelIdxList = s(k).PixelIdxList;
end
    obj.NumObjects = length(s);
    obj.ImageSize = size(im);
end
