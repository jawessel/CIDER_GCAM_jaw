function pattern = CIDER_pattern_scale(mean_response,pattern_to_scale)
    pattern = pattern_to_scale.*reshape(mean_response,1,1,[]);
end