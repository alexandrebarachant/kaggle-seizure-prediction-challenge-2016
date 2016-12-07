function [N,L] = nonlin(y_step,v)

N = zeros(1,v);
L = zeros(1,v);
for i=1:1:v
    N(i) = nonlinear_energy(y_step(:,i));    % Nonlinear energy per epoch
    L(i) = sum(abs(diff(y_step(:,i))));      % curve length;
end