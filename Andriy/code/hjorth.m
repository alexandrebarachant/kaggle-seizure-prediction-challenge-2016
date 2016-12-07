% Function to calculate Hjorth parameters
function [activity, mobility, complexity] = hjorth(y_step)                              

activity  = var(y_step,1);        % Variance of each EEG epoch. 1st Hjorth parameter

eeg_diff1 = diff(y_step,1,1);      % 1st derivative of EEG
mobility = std(eeg_diff1,1)./(std(y_step,1)+eps);     % EEG Mobility. 2nd Hjorth parameter

eeg_diff2 = diff(eeg_diff1,1,1);      % 2nd derivative of EEG
complexity = (std(eeg_diff2,1)./std(eeg_diff1,1)+eps)./mobility;        % EEG Complexity. 3rd Hjorth parameter
