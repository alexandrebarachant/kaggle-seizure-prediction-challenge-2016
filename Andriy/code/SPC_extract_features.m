function [feat_vector] = SPC_extract_features(data)

no_range = 0;
if(range(data)<0.01)
    no_range = 1;
    data = randn(size(data));
end

fs = 256;
zc = zero_crossing(data);
[N,L] = nonlin(data,1);
mmax = minmax(data);
[activity, mobility, complexity] = hjorth(data);
H_sh = entropy( data');
wavelet_energy = wavelet_coeff( data,fs);
RMS_amp = RMS_amplitude( data);
kurt = kurtosis( data);
skew = skewness( data);
[svd_ent, fisher] = inf_theory( data);
[ZC1d, ZC2d, V1d, V2d] = raw_analysis( data');

win_size = length(data);
Y = fft(data,win_size);                        %Winsize (N) point fft
spectrum1 = (Y.* conj(Y))/win_size;             %psd of each epoch
spectrum2 = spectrum1(1:(win_size/2+1));       %Ignore upper half of fft
clear Y spectrum1
freq = fs*(0:win_size/2)/win_size;

total_power = EEG_subbands(spectrum2,freq,[0 128]);

[band0_2] = EEG_subbands(spectrum2,freq,[0 2]);
[band1_3] = EEG_subbands(spectrum2,freq,[1 3]); 
[band2_4] = EEG_subbands(spectrum2,freq,[2 4]); 
[band3_5] = EEG_subbands(spectrum2,freq,[3 5]); 
[band4_6] = EEG_subbands(spectrum2,freq,[4 6]); 
[band5_7] = EEG_subbands(spectrum2,freq,[5 7]); 
[band6_8] = EEG_subbands(spectrum2,freq,[6 8]); 
[band7_9] = EEG_subbands(spectrum2,freq,[7 9]); 
[band8_10] = EEG_subbands(spectrum2,freq,[8 10]); 
[band9_11] = EEG_subbands(spectrum2,freq,[9 11]); 
[band10_12] = EEG_subbands(spectrum2,freq,[10 12]); 
[band11_13] = EEG_subbands(spectrum2,freq,[11 13]); 
[band12_14] = EEG_subbands(spectrum2,freq,[12 14]); 
[band13_15] = EEG_subbands(spectrum2,freq,[13 15]); 
[band14_16] = EEG_subbands(spectrum2,freq,[14 16]); 
[band15_17] = EEG_subbands(spectrum2,freq,[15 17]); 
[band16_18] = EEG_subbands(spectrum2,freq,[16 18]); 
[band17_19] = EEG_subbands(spectrum2,freq,[17 19]); 
[band18_20] = EEG_subbands(spectrum2,freq,[18 20]); 
[band19_21] = EEG_subbands(spectrum2,freq,[19 21]); 
[band20_22] = EEG_subbands(spectrum2,freq,[20 22]); 
[band21_23] = EEG_subbands(spectrum2,freq,[21 23]); 
[band22_24] = EEG_subbands(spectrum2,freq,[22 24]); 
[band23_25] = EEG_subbands(spectrum2,freq,[23 25]); 
[band24_26] = EEG_subbands(spectrum2,freq,[24 26]); 
[band25_27] = EEG_subbands(spectrum2,freq,[25 27]); 
[band26_28] = EEG_subbands(spectrum2,freq,[26 28]); 
[band27_29] = EEG_subbands(spectrum2,freq,[27 29]); 
[band28_30] = EEG_subbands(spectrum2,freq,[28 30]); 
[band29_31] = EEG_subbands(spectrum2,freq,[29 31]); 
[band30_32] = EEG_subbands(spectrum2,freq,[30 32]); 

[band3_15] = EEG_subbands(spectrum2,freq,[3 15]);%LeVan
[band15_30] = EEG_subbands(spectrum2,freq,[15 30]);
[band30_55] = EEG_subbands(spectrum2,freq,[30 55]);
[band59_61] = EEG_subbands(spectrum2,freq,[59 61]);

[band51_69] = EEG_subbands(spectrum2,freq,[51 69]); %Gasser
[band20_30] = EEG_subbands(spectrum2,freq,[20 30]);% EEG beta ryhthm...Goncharova
[band35_60] = EEG_subbands(spectrum2,freq,[35 60]); %piper rythm

[band25_128] = EEG_subbands(spectrum2,freq,[25 128]); %absolute high beta2 power ... van de velde



[band0_2norm] = band0_2./total_power; 
[band1_3norm] = band1_3./total_power; 
[band2_4norm] = band2_4./total_power; 
[band3_5norm] = band3_5./total_power; 
[band4_6norm] = band4_6./total_power; 
[band5_7norm] = band5_7./total_power; 
[band6_8norm] = band6_8./total_power; 
[band7_9norm] = band7_9./total_power; 
[band8_10norm] = band8_10./total_power; 
[band9_11norm] = band9_11./total_power; 
[band10_12norm] = band10_12./total_power; 
[band11_13norm] = band11_13./total_power; 
[band12_14norm] = band12_14./total_power; 
[band13_15norm] = band13_15./total_power; 
[band14_16norm] = band14_16./total_power; 
[band15_17norm] = band15_17./total_power; 
[band16_18norm] = band16_18./total_power; 
[band17_19norm] = band17_19./total_power; 
[band18_20norm] = band18_20./total_power; 
[band19_21norm] = band19_21./total_power; 
[band20_22norm] = band20_22./total_power; 
[band21_23norm] = band21_23./total_power; 
[band22_24norm] = band22_24./total_power; 
[band23_25norm] = band23_25./total_power; 
[band24_26norm] = band24_26./total_power; 
[band25_27norm] = band25_27./total_power; 
[band26_28norm] = band26_28./total_power; 
[band27_29norm] = band27_29./total_power; 
[band28_30norm] = band28_30./total_power; 
[band29_31norm] = band29_31./total_power; 
[band30_32norm] = band30_32./total_power; 

[band3_15norm] = band3_15 ./total_power;
[band15_30norm] = band15_30./total_power;
[band59_61norm] = band59_61 ./total_power;

[band51_69norm] = band51_69 ./total_power;
[band20_30norm] = band20_30./total_power;
[band59_61norm] = band59_61 ./total_power;


[band25_128norm] = band25_128 ./total_power;


[SEF90, TP] = spectral_edge(spectrum2,freq,1,32,.9);

[SEF95, TP] = spectral_edge(spectrum2,freq,1,32,.95); 
[SEF80, TP] = spectral_edge(spectrum2,freq,1,32,.8); 

[peak,peak_freq] = EEG_PSD_features(spectrum2,freq,1,32);
H_spec = spectral_entropy_g(spectrum2,(length(spectrum2)));

[mean_freq,bw]=iwmf_and_bw_low(data,fs);

feat_vector = [L; H_sh; mmax; RMS_amp; peak_freq; activity; mobility; complexity; N; H_spec; 
zc; wavelet_energy; kurt;skew;svd_ent;fisher;ZC1d;ZC2d;V1d;V2d;
total_power; band0_2; band1_3; band2_4; band3_5; band4_6; band5_7; band6_8; band7_9; band8_10; band9_11; band10_12;
band11_13;band12_14;band13_15;band14_16;band15_17;band16_18;band17_19;band18_20;band19_21;band20_22;band21_23;band22_24;band23_25;band24_26;
band25_27;band26_28;band27_29;band28_30;band29_31;band30_32;
band11_13norm;band12_14norm;band13_15norm;band14_16norm;band15_17norm;band16_18norm;band17_19norm;band18_20norm;band19_21norm;
band20_22norm;band21_23norm;band22_24norm;band23_25norm;band24_26norm;band25_27norm;band26_28norm;band27_29norm;band28_30norm;band29_31norm;band30_32norm;
band0_2norm; band1_3norm; band2_4norm; band3_5norm; band4_6norm; band5_7norm; band6_8norm; band7_9norm; band8_10norm; band9_11norm; band10_12norm;
SEF90;SEF95;SEF80; mean_freq;bw; 
band3_15; band15_30; band59_61; band51_69; band20_30; band59_61; band25_128;
band3_15norm; band15_30norm; band59_61norm; band51_69norm; band20_30norm; band59_61norm; band25_128norm];

if(no_range == 1)
    feat_vector = NaN(size(feat_vector));
end