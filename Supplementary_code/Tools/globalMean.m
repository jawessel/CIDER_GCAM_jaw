function av = globalMean(M,varargin)




M_size = size(M);
if length(M_size) ==2
    M_size = [M_size, 1];
end
if length(M_size) ==3
    M_size = [M_size, 1];
end
% ww = p.ww;

lat = linspace(-90,90,M_size(2))';
ww = cos(lat/180*pi);
if nargin>1
    p = varargin{1};
    ww = p.ww;
end

av = zeros(M_size(3:4));
% area = getArea;

for i = 1:M_size(3)
    for j = 1:M_size(4)
        av(i,j) = (mean((sum(M(:,:,i,j)'.*ww)/sum(ww))'));
%         av(i,j) = M(:,:,i,j).*area;
    end
end
end