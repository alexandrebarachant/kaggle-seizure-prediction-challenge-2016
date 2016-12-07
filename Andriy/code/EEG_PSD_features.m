%Function to extract frequency and power features for EEG classifier method
function [peak,peak_freq] = EEG_PSD_features(spectrum,freq,f1,f2)

%*************** Find dominant peak in range F1-F2 *****************************
i1 = find(freq==f1);     % f1 hertz index
i2 = find(freq==f2);    % f2 Hz index

spec = spectrum(i1:i2,:);       %of 2-20HZ       %power
[peak,w] = max(spec,[],1);       %Find the peak and index of each peak for each psd for each epoch. Treats each column in spectrum as a vector and reurning max in each
peak = peak+eps;                % Add LSB to avoid errors

peak_freq = freq(w+i1);         %Find freqency of dominant peak
