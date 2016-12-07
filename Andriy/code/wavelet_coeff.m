function val=wavelet_coeff(data,fs)
num_layers=7;
num_coeffs=8;
[h,g,rh,rg]=daub(num_coeffs);

wav_res=wt(data,h,g,num_layers);
pos=sub_pos(length(data),num_layers);
bands=extract_subbands(wav_res,pos);

%MEAN OF COEFFICIENTS AND ABSOLUTE ENERGY
for j=1:length(bands)
    mean_coeffs(j)=mean(abs(bands{j}));
end
val=mean_coeffs(5);