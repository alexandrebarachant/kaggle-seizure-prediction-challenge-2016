function [fit] = compare_new(varargin)

% Determine list of inputs.
inpn = cell(1, length(varargin));
for kn = 1:length(varargin);
    inpn{kn} = inputname(kn);
end

v = {varargin{:} inpn};
th = idss(v{2});
th = th('y1', cell(0));

z = v{1};
z = iddata(z(:, 1), z(:, 2:end), 1);
y = pvget(z, 'OutputData');
z1 = z(:, 'y1', cell(0));

[yh, x01] = predict(th, z1, 1, 'e');
yhh = pvget(yh, 'OutputData');

%Compute fit.
err = norm(yhh{1} - y{1});
meanerr = norm(y{1} - mean(y{1}));
fit = 100*(1-err/meanerr);
