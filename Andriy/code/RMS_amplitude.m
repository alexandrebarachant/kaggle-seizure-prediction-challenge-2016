function result = RMS_amplitude(data)

if size(data,1) < size(data,2)
    data = data';
end
result = sqrt((data'*data)/(length(data)));


