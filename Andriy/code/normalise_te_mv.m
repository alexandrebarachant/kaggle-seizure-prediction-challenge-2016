function [data] = normalise(data,u,std)

[N]=size(data,2);

data=(data-u*ones(1,N)) ./ (std*ones(1,N));
data=data';
