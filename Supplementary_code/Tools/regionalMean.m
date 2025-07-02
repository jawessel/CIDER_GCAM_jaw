function av = regionalMean(M,latbounds,lonbounds)
% latbounds = p.latbounds;
% lonbounds = p.lonbounds;
sizey = size(M);
lon = linspace(0,358.75,sizey(1))';
lat = linspace(-90,90,sizey(2))';
M_size = size(M);
if length(M_size) ==2
    M_size = [M_size, 1];
end
if length(M_size) ==3
    M_size = [M_size, 1];
end
ww = cos(lat);

av = zeros(M_size(3:4));

for i = 1:M_size(3)
    for j = 1:M_size(4)
        av(i,j) = (mean((sum(M(inside(lon,lonbounds),inside(lat,latbounds),i,j)'.*ww(inside(lat,latbounds)))/sum(ww(inside(lat,latbounds))))'));
    end
end
end