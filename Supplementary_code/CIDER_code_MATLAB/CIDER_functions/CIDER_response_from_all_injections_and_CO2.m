function total_response = CIDER_response_from_all_injections_and_CO2(all_injection_and_CO2,all_param_AOD,all_param_climate)
size_inputs = size(all_injection_and_CO2);
injection_count = size_inputs(2)-1;
total_response = CIDER_response_from_1_forcing(all_param_climate(end,:),all_injection_and_CO2(:,end));
for i = 1:injection_count
    injection = all_injection_and_CO2(:,i);
    param_AOD = all_param_AOD(i,:);
    param_climate = all_param_climate(i,:);
    AOD = CIDER_AOD_from_injection(param_AOD,injection);
    total_response = total_response+CIDER_response_from_1_forcing(param_climate,AOD);
end


end


