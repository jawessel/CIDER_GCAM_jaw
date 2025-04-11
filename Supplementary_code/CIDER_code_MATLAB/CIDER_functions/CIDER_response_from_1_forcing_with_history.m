function emulated_response = CIDER_response_from_1_forcing_with_history(params,forcing,history_length)
impulse_p_SAI = [];
impulse_p_SAI.tau = params(1);
impulse_p_SAI.mu = params(2);
emulated_response = zeros(length(forcing),1);

history_tseries = CIDER_response_from_1_forcing(params,[forcing(1:history_length);forcing(history_length)*ones(length(emulated_response),1)]);
history_tseries = history_tseries(history_length+1:end)-history_tseries(history_length);
% Convolve the impulse response
for k = 1:length(emulated_response)
    
    emulated_response(k) = history_tseries(k);
    
    for j = 1:k
        emulated_response(k) = emulated_response(k) + impulse_semiInfDiff(j,impulse_p_SAI)*forcing(k-j+1);
    end
end
end