function all_params = CIDER_train_climate_params(all_forcings,all_responses,varargin)
forcings_size = size(all_forcings);
forcings_count = forcings_size(2);
forcings_length = forcings_size(1);
all_params = zeros(forcings_count,2);
x0 = [240;0];
x0 = [300;0];
x0 = [120;0];
x0 = [];
for i = 1:forcings_count
    x0 = [x0, [120;0]];

end
lb = [0;-1];
% lb = [100;-1];
ub = [5000;1];
% ub = [100;1];
if nargin>2
    x0 = varargin{1}';
    if length(x0)==2
        for i = 1:forcings_count-1
            x0 = [x0, varargin{1}'];

        end
    end
    lb = varargin{2};
    ub = varargin{3};
end

for i = 1:forcings_count
    forcing = all_forcings(:,i);
    simulated_response = all_responses(:,i);
    emulator_error = @(params) sum((simulated_response-CIDER_response_from_1_forcing(params,forcing)).^2) / length(simulated_response);
    
    optimal_params = fmincon(@(x) emulator_error(x),x0(:,i),[],[],[],[],lb,ub);
    all_params(i,:) = optimal_params;
end

end