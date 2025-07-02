function T1 = calculateT1(T_matrix)

size_of_T_matrix = size(T_matrix);
if length(size_of_T_matrix) == 2
    size_of_T_matrix = [size_of_T_matrix 2];
    T_matrix = cat(3,T_matrix,T_matrix);
end

T_zonal = squeeze(mean(T_matrix,1));

T1 = zeros(size_of_T_matrix(3),1);
load get_lat_and_lon.mat


A = 2*pi*earthRadius^2;

fun = @(j) T_zonal(j,i)*cosd(lat(j))*sind(lat(j));
j = 1:192';
for i = 1:size_of_T_matrix(3)
   T1(i) = sum(T_zonal(j,i).*cosd(lat(j)).*sind(lat(j)))/sum(cosd(lat(j)));
end

end