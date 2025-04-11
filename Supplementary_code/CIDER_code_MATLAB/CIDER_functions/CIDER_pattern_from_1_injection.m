function response_pattern = CIDER_pattern_from_1_injection(injection,param_AOD,param_climate,pattern_to_scale)

AOD_emulated = CIDER_AOD_from_injection(param_AOD,injection);
emulated_response = CIDER_response_from_1_forcing(param_climate,AOD_emulated);
response_pattern = CIDER_pattern_scale(emulated_response,pattern_to_scale);

end