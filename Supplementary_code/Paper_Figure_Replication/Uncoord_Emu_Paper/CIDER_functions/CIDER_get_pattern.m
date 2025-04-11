function pattern = CIDER_get_pattern(data,steady_length)
    data_to_average = data(:,:,((end-steady_length)+1):end);
    pattern = mean(data_to_average,3)/globalMean(mean(data_to_average,3));
end