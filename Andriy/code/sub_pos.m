%
% Stephen Faul
% 21st July 2004
%
% SUB_POS: returns the starting positions of each subband of the
%          resultant vector from wt
%
%          input: data_len -- the length of the original signal
%                 num_coeffs -- the number of filter coefficients (2*order for Daubechie
%
%          output: pos -- vector of the starting positions of each subband
%

function pos = sub_pos(data_len,num_coeffs)
for i=1:num_coeffs
    pos(i)=data_len/(2^i)+1;
end
pos(i+1)=1;