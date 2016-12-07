% EEG Spectral entropy per epoch
function H = spectral_entropy_g(spectrum,w)

flag=0;    
psd = spectrum;
pdf = zeros(size(spectrum));
if(flag==1)
    for i = 1:1:w
        pdf(:,i) = psd(:,i)./(sum(psd(:,i))+eps); 
    end
elseif(flag==0)
    pdf = psd./(repmat(sum(psd,1),size(psd,1),1)+eps); 
end

H = -(sum(pdf.*log2(pdf+eps)));    