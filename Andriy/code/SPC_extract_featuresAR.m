function [feat_vector] = SPC_extract_featuresAR(data)
no_range = 0;
if(range(data)<0.01)
    no_range = 1;
    data = randn(size(data));
end


dat1=data(1:floor(end/2));
dat2=data(ceil(end/2)+1:end);
fit = ar_prediction_error(dat1,dat2,9);
AR1 = fit(1);
AR2 = fit(2);
AR3 = fit(3);
AR4 = fit(4);
AR5 = fit(5);
AR6 = fit(6);
AR7 = fit(7);
AR8 = fit(8);
AR9 = fit(9);

feat_vector = [AR1;AR2;AR3;AR4;AR5;AR6;AR7;AR8;AR9];

if(no_range == 1)
    feat_vector = NaN(size(feat_vector));
end