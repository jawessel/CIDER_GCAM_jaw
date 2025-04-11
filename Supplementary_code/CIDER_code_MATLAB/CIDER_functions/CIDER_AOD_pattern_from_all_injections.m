function total_AOD_pattern = CIDER_AOD_pattern_from_all_injections(param_AOD_all,injections,patterns)
size_injections = size(injections);
injection_count = size_injections(2);
injection_length = size_injections(1);
size_patterns = size(patterns);
total_AOD_pattern = zeros(size_patterns(1),size_patterns(2),injection_length);
for i = 1:injection_count
    injection = injections(:,i);
    param_AOD = param_AOD_all(i,:);
    pattern = patterns(:,:,i);
    AOD_emulated = CIDER_AOD_from_injection(param_AOD,injection);
    scaled_pattern = CIDER_pattern_scale(AOD_emulated,pattern);
    total_AOD_pattern = total_AOD_pattern+scaled_pattern;
end

end

