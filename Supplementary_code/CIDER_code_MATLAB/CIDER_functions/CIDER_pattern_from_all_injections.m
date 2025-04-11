function total_pattern = CIDER_pattern_from_all_injections(all_injection,all_param_AOD,all_param_climate,all_pattern_to_scale)
injection_array_size = size(all_injection);
injection_length = injection_array_size(1);
injection_count = injection_array_size(2);
pattern_size = size(all_pattern_to_scale());
total_pattern = zeros(pattern_size(1),pattern_size(2),injection_length);

for i = 1:injection_count
    injection = all_injection(:,i);
    if mean(injection) ~= 0
        param_AOD = all_param_AOD(i,:);
        param_climate = all_param_climate(i,:);
        pattern_to_scale = all_pattern_to_scale(:,:,i);
        response_pattern = CIDER_pattern_from_1_injection(injection,param_AOD,param_climate,pattern_to_scale);
        total_pattern = total_pattern+response_pattern;
    end
end

end