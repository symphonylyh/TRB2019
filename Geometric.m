function varout = Geometric(views)
% Mirror reconstruction paper 2006
% Source code: https://github.com/keith2000/VisualHullFromMirrors

imCell = cell(1,1);
DimCell = cell(1,1);
C = cell(1,1);
PixelA = cell(1,1);
PixelCvxA = cell(1,1);
PixelCvxity = cell(1,1);
EquiDim = cell(1,1);
Boundary = cell(1,1);
CvxHull = cell(1,1);
FeretDimension = cell(1,1);
imCell{1} = views{2};
imCell{2} = views{3};
imCell{3} = views{1};
for i = 1:1:3
    %-1
    BW = bwareaopen(imCell{i},4000);
    DimCell{i} = im2double(BW);
    %-2-����͹�ȡ����ġ�͹������Чֱ��
    C{i} = regionprops(BW,'Centroid');
    CvxHull{i} = regionprops(BW,'ConvexHull');
    EquiDim{i} = regionprops(BW,'EquivDiameter');
    PixelA{i} = regionprops(BW,'Area');
    PixelCvxA{i} = regionprops(BW,'ConvexArea');
    PixelCvxity{i} = [PixelA{i}.Area]./[PixelCvxA{i}.ConvexArea];
    %-3-����Feret����
    x = CvxHull{i}.ConvexHull(:,1)' - C{i}.Centroid(1);
    y = CvxHull{i}.ConvexHull(:,2)' - C{i}.Centroid(2);
    deta_theta = 0.5/180*pi;
    count0 = 1;
    for angle = 0:deta_theta:pi
        xx = x.*cos(angle) - y.*sin(angle);
        yy = y.*cos(angle) + x.*sin(angle);
        FeretDimension{i}(count0,1) = angle;
        FeretDimension{i}(count0,2) = min(xx);
        FeretDimension{i}(count0,3) = max(xx);
        FeretDimension{i}(count0,4) = min(yy);
        FeretDimension{i}(count0,5) = max(yy);
        FeretDimension{i}(count0,6) = max(xx) - min(xx);
        FeretDimension{i}(count0,7) = max(yy) - min(yy);
        count0 = count0 + 1;
    end
    %-3-����߽�
    Boundary{i} = GetBoundaryMex(SelectObjectMex(DimCell{i},1))';
end
Convexity = (PixelCvxity{1}*PixelA{1}.Area + PixelCvxity{2}*PixelA{2}.Area + ...
             PixelCvxity{3}*PixelA{3}.Area)/(PixelA{1}.Area + PixelA{2}.Area + ...
             PixelA{3}.Area);
%--���ݼ���õ���Feret�������������������
resolution_factor = varin{3};
maxSize = zeros(1,3);
minSize = zeros(1,3);
for i = 1:1:3
    maxSize(i) = max(FeretDimension{i}(:,7))*resolution_factor(i);
    minSize(i) = min(FeretDimension{i}(:,7))*resolution_factor(i);
end
maxDiameter = max(maxSize);
minDiameter = min(minSize);
midDiameter = max(minSize);
sieveDiameter = min(maxSize);
FERatio = maxDiameter/minDiameter;
Sphericity = (midDiameter*minDiameter/maxDiameter/maxDiameter)^(1/3);
%--����߽缫����
radius = cell(1,1);
theta = cell(1,1);
for i = 1:1:3
    vector = vertcat(Boundary{i}(1,:) - C{i}.Centroid(1),Boundary{i}(2,:) - C{i}.Centroid(2));
    radius{i} = sqrt((vector(1,:)).^2 + (vector(2,:)).^2);
    for k = 1:1:size(vector,2)
        if vector(1,k)>=0 && vector(2,k)>=0
            theta{i}(1,k) = atan(vector(2,k)/vector(1,k));
        elseif vector(1,k)<=0 && vector(2,k)>=0
            theta{i}(1,k) = atan(vector(2,k)/vector(1,k)) + pi;
        elseif vector(1,k)<=0 && vector(2,k)<=0
            theta{i}(1,k) = atan(vector(2,k)/vector(1,k)) + pi;
        elseif vector(1,k)>=0 && vector(2,k)<=0
            theta{i}(1,k) = atan(vector(2,k)/vector(1,k)) + 2*pi;
        end
    end
end
%--ȥ���������桰roughness�����򻯿�������
r_smooth = cell(1,1);
r_simplify = cell(1,1);
t_smooth = cell(1,1);
t_simplify = cell(1,1);
boundary_smooth = cell(1,1);
boundary_simplify = cell(1,1);
for i = 1:1:3
    %-1-ȥ���������桰roughness��
    [angle,ix] = sort(theta{i});
    r_smooth{i} = radius{i}(ix)';
    r_smooth{i} = smooth(r_smooth{i},0.02,'rloess');
    r_smooth{i} = r_smooth{i}';
    t_smooth{i} = angle;
    n = size(r_smooth{i},2);
    x = C{i}.Centroid(1) + r_smooth{i}.*cos(angle);
    y = C{i}.Centroid(2) + r_smooth{i}.*sin(angle);
    boundary_smooth{i} = [x;y];
    %-2-�򻯿�������
    col1 = 1;
    col2 = n;
    count = 1;
    num = zeros(1,1);
    while col1 < n
        if col2 == col1 + 1
            col1 = col2;
            col2 = n;
            num(1,count) = col1;
            continue;
        else
            num(1,count) = col1;
        end
        x1 = x(col1);
        x2 = x(col2);
        y1 = y(col1);
        y2 = y(col2);
        mid_p = [mean([x1,x2]),mean([y1,y2])];
        v_n = [y2 - y1,x1 - x2];
        v_n = v_n/normest(v_n);
        vetx = vertcat(x(col1+1:col2-1)-mid_p(1),y(col1+1:col2-1)-mid_p(2));
        D = vetx'*v_n';
        deta = max(abs(D));
        col3 = find(abs(D) == deta);
        if deta < EquiDim{i}.EquivDiameter*1.50/100
            count = count + 1;
            col1 = col2;
            col2 = n;
        else
            col2 = col3(1) + col1;
        end
    end
    r_simplify{i} = r_smooth{i}(num);
    t_simplify{i} = t_smooth{i}(num);
    boundary_simplify{i} = [C{i}.Centroid(1) + r_simplify{i}.*cos(t_simplify{i});...
            C{i}.Centroid(2) + r_simplify{i}.*sin(t_simplify{i})];
end
%--����򻯺�����ȥ����Non-corner points��
boundary_corner = cell(1,1);
for i = 1:1:3
    x = boundary_simplify{i}(1,:)';
    y = boundary_simplify{i}(2,:)';
    K = convhull(x,y);
    boundary_corner{i} = [x(K),y(K)]';
end
%--����Բ��roundness
Perimeter = zeros(1,1);
Area_smooth = zeros(1,1);
for i = 1:1:3
    x = boundary_smooth{i}(1,:);
    y = boundary_smooth{i}(2,:);
    x1 = [x(end),x(1:1:end-1)];
    y1 = [y(end),y(1:1:end-1)];
    Area_smooth(i) = polyarea(x',y');
    Perimeter(i) = sum(sqrt((x-x1).^2+(y-y1).^2));
end
Roundness = sum(Perimeter.^2)/sum(Area_smooth)/4/pi;
%--�������ָ��
Angularity = zeros(1,1);
Area_simplify = zeros(1,1);
for i = 1:1:3
    x = boundary_simplify{i}(1,:);
    y = boundary_simplify{i}(2,:);
    Area_simplify(i) = polyarea(x',y');
    angles = polyangles(x(end:-1:1),y(end:-1:1))';
    angles0 = [angles(2:1:end),angles(1)];
    [f,~] = hist(abs(angles-angles0),5:10:175);
    Angularity(i) = sum((0:10:170).*(f));
end
AI = sum(Angularity.*Area_simplify)/sum(Area_simplify);
%--����ͶӰ���
Area = zeros(1,1);
for i = 1:1:3
    x = boundary_smooth{i}(1,:).*resolution_factor(i);
    y = boundary_smooth{i}(2,:).*resolution_factor(i);
    Area(i) = polyarea(x',y');    
end
varout.C = C;
varout.Boundary = Boundary;
varout.boundary_smooth = boundary_smooth;
varout.boundary_simplify = boundary_simplify;
varout.boundary_corner = boundary_corner;
varout.Convexity = Convexity;
varout.FeretDimension = FeretDimension;
varout.Roundness = Roundness;
varout.AI = AI;
varout.maxDiameter = maxDiameter;
varout.minDiameter = minDiameter;
varout.midDiameter = midDiameter;
varout.sieveDiameter = sieveDiameter;
varout.FERatio = FERatio;
varout.Sphericity = Sphericity;
varout.Area = Area;
end

