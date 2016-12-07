function [SEF, TP] = spectral_edge(PSD,freq,f1,f2,percent)
i1 = find(freq==f1);    % f1 hertz index
i2 = find(freq==f2);    % f2 Hz index

P = PSD(i1:i2,:);       % Power in range f1-f2

edgeTP = zeros(1,size(P,2));
SEF = zeros(1,size(P,2));
TP = sum(P,1);

for j=1:size(P,2)
    i=1;
    edgeTP(j) = percent*sum(P(:,j));   
    
    while((sum(P([1:i],j)) < edgeTP(j)) & (i <= (i2-i1)))
        i=i+1;
    end
    
    SEF(j) = freq(i1+i);       %spectral edge freq
end