function zc = zero_crossing(data)
zc = sum( (data(1:end-1,:).*data(2:end,:)) < 0, 1);