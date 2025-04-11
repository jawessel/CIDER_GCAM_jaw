function total_pattern = CIDER_pattern_from_all_injections_and_CO2(all_injection_and_CO2,all_param_AOD,all_param_climate,all_pattern_to_scale)
total_pattern_inj = CIDER_pattern_from_all_injections(all_injection_and_CO2(:,1:end-1),all_param_AOD,all_param_climate(1:end-1,:),all_pattern_to_scale);
pattern_CO2 = CIDER_pattern_scale(CIDER_response_from_1_forcing(all_param_climate(end,:),all_injection_and_CO2(:,end)),all_pattern_to_scale(:,:,end));
total_pattern = total_pattern_inj+pattern_CO2;
end