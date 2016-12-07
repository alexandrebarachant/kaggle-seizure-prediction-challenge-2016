function fit_error=ar_prediction_error(model_data,test_data,ar_order)
model_data = remove_infsnans(model_data);
test_data = remove_infsnans(test_data);

if(range(model_data)<1e-5)
    fit_error (1:ar_order) = 50;
else
    if (range(test_data)<1e-5)
        fit_error(1:ar_order) = 100;
    else
        if size(model_data,2)>size(model_data,1)    %These functions need column matrices
            model_data=model_data';
        end
        if size(test_data,2)>size(test_data,1)
            test_data=test_data';
        end
        m = ar(model_data,ar_order,'YW');
        for zzz=1:size(m,2)
           fit_error(zzz) = compare_new(test_data,m{zzz});
        end
    end
end

function data = remove_infsnans(data)
NANs = find(isnan(data));
for i=1:length(NANs)
    if(NANs(i)==1)
        data(1) = data(2);
    else if(NANs(i)==length(data))
            data(end) = data(end-1);
        else
            data(NANs(i)) = mean([data(NANs(i)-1);data(NANs(i)+1)]);
        end
    end
end
NANs = find(isinf(data));
for i=1:length(NANs)
    if(NANs(i)==1)
        data(1) = data(2);
    else if(NANs(i)==length(data))
            data(end) = data(end-1);
        else
            data(NANs(i)) = mean([data(NANs(i)-1);data(NANs(i)+1)]);
        end
    end
end