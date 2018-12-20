function obj = boundary_smoothing(obj, toRemove)

for k = 1:obj.NumObjects

obj.objects(k).cartesian = smoothPolygon(obj.objects(k).rawXY, toRemove);


end

end


