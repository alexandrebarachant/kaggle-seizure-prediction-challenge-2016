function bands = extract_subbands(coeffs,pos)
pos=round(pos);
for i=1:length(pos)
    if i>1
        bands{i} = coeffs(pos(i):pos(i-1)-1);
    else
        bands{1} = coeffs(pos(1):end);
    end
end