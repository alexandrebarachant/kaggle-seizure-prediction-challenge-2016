function polinomio=trigpol(N)

%  TRIGPOL generate trigonometric polynomial.
%
%          TRIGPOL(N) generates the following polynomial in e^jw:
%          
%          P(e^jw)=sum(k=0,...,N-1){ (N-1+k) ( e^(-jw) - 2 + e^jw )
%                                       k 
%          
%          The output of the function is a vector of size 2*N-1 holding
%          the coefficients corresponding to the different powers of e^jw. 
%
%          See also: NUMCOMB, DAUB, SPLINE


%--------------------------------------------------------
% Copyright (C) 1994, 1995, 1996, by Universidad de Vigo 
%                                                      
%                                                      
% Uvi_Wave is free software; you can redistribute it and/or modify it      
% under the terms of the GNU General Public License as published by the    
% Free Software Foundation; either version 2, or (at your option) any      
% later version.                                                           
%                                                                          
% Uvi_Wave is distributed in the hope that it will be useful, but WITHOUT  
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or    
% FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License    
% for more details.                                                        
%                                                                          
% You should have received a copy of the GNU General Public License        
% along with Uvi_Wave; see the file COPYING.  If not, write to the Free    
% Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.             
%                                                                          
%       Author: Nuria Gonzalez Prelcic
%       e-mail: Uvi_Wave@tsc.uvigo.es
%--------------------------------------------------------


%The polynomial is constructed from a sum. 
%coefs holds the coefficients of each term in the sum.

coefs=zeros(N,2*N-1);
coefs(1,N)=1;

 
for i=1:N-1
	fila=[1 -2 1];
	for j=2:i
		fila=conv(fila,[1 -2 1]);
	end;
	fila=numcomb(N-1+i,i)*(-0.25)^i*fila;
	fila=[ zeros(1,(N-i-1))  fila zeros(1,(N-i-1))];
	coefs(i+1,:)=fila;
end

for i=0:(2*(N-1))
	polinomio(i+1)=0;
	for j=1:N
		polinomio(i+1)=polinomio(i+1)+coefs(j,i+1);
	end
end;
