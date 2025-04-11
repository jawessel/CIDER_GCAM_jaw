function all_params = CIDER_train_AOD_params(all_step_injections,all_step_responses,all_feedback_injections,all_feedback_responses,weightings,varargin)
step_injections_size = size(all_step_injections);
step_injections_count = step_injections_size(2);
step_injections_length = step_injections_size(1);

all_params = zeros(step_injections_count,3);
if nargin>5
    all_short_step_injections = varargin{1};
    all_simulated_short_step_response = varargin{2};
    for i = 1:step_injections_count
        step_injection = all_step_injections(:,i);
        feedback_injection = all_feedback_injections(:,i);
        simulated_step_response = all_step_responses(:,i);
        simulated_feedback_response = all_feedback_responses(:,i);
        short_step_injection = all_short_step_injections(:,i);
        simulated_short_step_response = all_simulated_short_step_response(:,i);
        short_step_error = @(params) sum((simulated_short_step_response-CIDER_AOD_from_injection(params,short_step_injection)).^2) / length(simulated_short_step_response);
        step_error = @(params) sum((simulated_step_response-CIDER_AOD_from_injection(params,step_injection)).^2) / length(simulated_step_response);
        feedback_error = @(params) sum((simulated_feedback_response-CIDER_AOD_from_injection(params,feedback_injection)).^2) / length(simulated_feedback_response);
        emulator_error = @(params) weightings(1)*step_error(params)+weightings(1)*short_step_error(params)+weightings(2)*feedback_error(params);
        x0 = [0.02; 0.01; 0.0002];
        % optimal_params = fmincon(@(x) emulator_error(x),x0,[],[],[],[],[0;0;0],[.1;.2;.1]);
        optimal_params = fmincon(@(x) emulator_error(x),x0,[],[],[],[],[0;0.08;0],[.1;.2;1]);
        all_params(i,:) = optimal_params;
    end
else
for i = 1:step_injections_count
    step_injection = all_step_injections(:,i);
    feedback_injection = all_feedback_injections(:,i);
    simulated_step_response = all_step_responses(:,i);
    simulated_feedback_response = all_feedback_responses(:,i);
    step_error = @(params) sum((simulated_step_response-CIDER_AOD_from_injection(params,step_injection)).^2) / length(simulated_step_response);
    feedback_error = @(params) sum((simulated_feedback_response-CIDER_AOD_from_injection(params,feedback_injection)).^2) / length(simulated_feedback_response);
    emulator_error = @(params) weightings(1)*step_error(params)+weightings(2)*feedback_error(params);
    x0 = [0.02; 0.1; 0.0002];
    % optimal_params = fmincon(@(x) emulator_error(x),x0,[],[],[],[],[0;0;0],[.1;.2;.1]);
    optimal_params = fmincon(@(x) emulator_error(x),x0,[],[],[],[],[0;0.08;0],[1;.3;.1]);
    all_params(i,:) = optimal_params;
end
end

end