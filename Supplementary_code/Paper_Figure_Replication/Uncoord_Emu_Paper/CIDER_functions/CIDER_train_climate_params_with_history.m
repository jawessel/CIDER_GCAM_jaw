function all_params = CIDER_train_climate_params_with_history(all_forcings,all_responses,varargin)
forcings_size = size(all_forcings);
forcings_count = forcings_size(2);
forcings_length = forcings_size(1);
all_params = zeros(forcings_count,2);
x0 = [240;0];
% x0 = [300;0];
lb = [0;-1];
% lb = [100;-1];
ub = [5000;1];
% ub = [35;1];
if nargin>2
    x0 = varargin{1};
    lb = varargin{2};
    ub = varargin{3};
end

for i = 1:forcings_count
    forcing = all_forcings(:,i);
    simulated_response = all_responses(:,i);
    emulator_error = @(params) sum((simulated_response-CIDER_response_from_1_forcing_with_history(params,forcing,12*20)).^2) / length(simulated_response);
    
    optimal_params = fmincon(@(x) emulator_error(x),x0,[],[],[],[],lb,ub);
    all_params(i,:) = optimal_params;
end

end