function array = repeatSeries(input_array,N)
size_of_input_array = size(input_array);
flip_later = 0;
if size_of_input_array(1) == 1
    flip_later = 1;

    input_array = input_array';
end
size_of_input_array = size(input_array);
array = zeros(N*size_of_input_array(1),1);

for i = 1:N
    array((i-1)*size_of_input_array+1:(i)*size_of_input_array) = input_array;
end

if flip_later
    array = array';
end

end