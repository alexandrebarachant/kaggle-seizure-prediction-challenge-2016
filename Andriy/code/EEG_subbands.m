% Function to calculate power in sub-bands from freq_range
function [PSD_band] = EEG_subbands(PSD,freq,freq_range)

i1 = find(freq==freq_range(1));     % f1 hertz index
i2 = find(freq==freq_range(2));     % f2 Hz index
PSD_band = trapz(PSD(i1+1:i2,:));   % Range of PSD

%**********************************************