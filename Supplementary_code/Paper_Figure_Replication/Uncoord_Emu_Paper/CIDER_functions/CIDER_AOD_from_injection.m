function AOD_emulated = CIDER_AOD_from_injection(param_AOD,injection)
% Load in parameters for AOD
beta = param_AOD(1);
alpha = param_AOD(2);
gamma = param_AOD(3);

AOD_emulated = zeros(length(injection),1);
if mean(injection)==0
    return
end
for k = 1:length(AOD_emulated)
    for j = 1:k
        AOD_emulated(k) = AOD_emulated(k) + impulse_firstOdiff_wGammaInj(beta, alpha,gamma,injection(k-j+1), j)*injection(k-j+1);
    end
    
    % for j = 1:k
    %     if k == 1
    %         AOD_emulated(k) = AOD_emulated(k) + impulse_firstOdiff_wGammaAOD(beta, alpha,gamma,0, j)*injection(k-j+1);
    %     else
    %         AOD_emulated(k) = AOD_emulated(k) + impulse_firstOdiff_wGammaAOD(beta, alpha,gamma,AOD_emulated(k-1), j)*injection(k-j+1);
    %     end
    % end
    % if ~isempty(find(injection(1:k)'~=0,1))
    %     for z = find(injection(1:k)'~=0)
    %         j = k-z+1;
    %         AOD_emulated(k) = AOD_emulated(k) + impulse_firstOdiff_wGammaInj(beta, alpha,gamma,injection(z), j)*injection(z);
    %     end
    % end
end
end
