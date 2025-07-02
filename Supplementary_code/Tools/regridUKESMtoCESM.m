function arrayOut = regridUKESMtoCESM(arrayIn)
sizey = size(arrayIn);
arrayOut = zeros(288,192,sizey(3));
% for i = 1:192
%     for j = 1:144
%         newJ = j-1;
%         arrayOut(2*newJ+1,i,:) = arrayIn(i,j,:);
%         arrayOut(2*newJ+2,i,:) = arrayIn(i,j,:);
%     end
% end

% Assume data is your original 192x144 matrix
% Create the original grid (x and y coordinates for each point)
[x_orig, y_orig] = meshgrid(1:144, 1:192);
% Create the target grid (288x192)
[x_new, y_new] = meshgrid(linspace(1, 144, 192), linspace(1, 192, 288));
% Interpolate the data onto the new grid
for i = 1:sizey(3)
    arrayOut(:,:,i) = interp2(x_orig, y_orig, squeeze(arrayIn(:,:,i)), x_new, y_new, 'linear');
end
% 'data_new' is now the regridded matrix with dimensions 288x192

end