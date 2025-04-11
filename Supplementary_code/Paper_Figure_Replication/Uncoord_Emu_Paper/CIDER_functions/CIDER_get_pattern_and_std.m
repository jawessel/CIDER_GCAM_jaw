function [pattern,S] = CIDER_get_pattern_and_std(data,steady_length)
    data_to_average = data(:,:,((end-steady_length)+1):end);
    pattern = mean(data_to_average,3)/globalMean(mean(data_to_average,3));
    S = std(data_to_average,0,3);

end