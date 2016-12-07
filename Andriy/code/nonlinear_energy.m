% Function to calculate nonlinear energy
function [N,x] = nonlinear_energy(epoch)

a = epoch.*epoch;
x = a(2:end-1) - epoch(1:end-2).*epoch(3:end);
N = mean(x);
