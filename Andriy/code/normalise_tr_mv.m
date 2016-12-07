function [data,m,v] = normalise(data)

[N]=size(data,2);
m = mean(data(:,:),2);
v = std(data(:,:),[],2);
data=(data-m*ones(1,N)) ./ (v*ones(1,N));
data=data';
