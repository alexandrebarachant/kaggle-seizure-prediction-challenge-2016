function [h,g,rh,rg]=daub(num_coefs)

%DAUB    Generates Daubechies filters. 
%
%        [H,G,RH,RG]=DAUB(NUM_COEFS) returns the coefficients of
%        the orthonormal Daubechies wavelets with maximum number
%        of vanishing moments and minimum phase. NUM_COEFS specifies
%        the number of coefficients. The number of vanishing moments,
%        that coincides with the number of zeros at z=-1, is equal 
%        to NUM_COEFS/2. 
%
%        H is the analysis lowpass filter, RH the synthesis 
%        lowpass filter, G the analysis highpass filter and
%        RG the synthesis highpass filter.
%
%        The choice of minimum phase leads to the most asymmetric scale
%        function.
%
%        NUM_COEFS must be even.
%
%        See also: TRIGPOL, SYMLETS.

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
%      Authors: Carlos Mosquera Nartallo
%               Nuria Gonzalez Prelcic
%       e-mail: Uvi_Wave@tsc.uvigo.es
%--------------------------------------------------------

if rem(num_coefs,2)
   error( 'Error: NUM_COEFS must be even!!!')
end

N=num_coefs/2;


poly=trigpol(N);    %Calculate trigonometric polynomial 


zeros=roots(poly);  %Calculate roots

% To construct rh for the minimum phase choice, we choose all the zeros 
% inside the unit circle. 

modulus=abs(zeros);

j=1;
for i=1:(2*(N-1))
	if (modulus(i)<1)
		zerosinside(j)=zeros(i);
		j=j+1;
	end;
end;

if j ~= N
	error('Error!!!'); % The number of zeros inside the unit circle
                             % must be equal to the number of zeros outside
                             % the unit circle.
end

An=poly(1);

realzeros=[];
imagzeros=[];
numrealzeros=0;
numimagzeros=0;

for i=1:(N-1)
	if (imag(zerosinside(i))==0)
		numrealzeros=numrealzeros+1;
		realzeros(numrealzeros)=zerosinside(i);
	else
		numimagzeros=numimagzeros+1;
		imagzeros(numimagzeros)=zerosinside(i);	
		
	end;

end;


% Once ho is factorized in its zeros, it must be normalized multiplying by "cte".

cte=1;

for i=1:numrealzeros
	cte=cte*abs(realzeros(i));
end

for i=1:numimagzeros
	cte=cte*abs(imagzeros(i));
end

cte=sqrt(abs(An)/cte);

cte=0.5^N*sqrt(2)*cte;

% Construction of rh from its zeros

rh=[ 1 1];

for i=2:N
	rh=conv(rh,[1 1]);
end

for i=1:numrealzeros
	rh=conv(rh,[1 -realzeros(i)]);
end

for i=1:2:numimagzeros
	rh=conv(rh,[1 -2*real(imagzeros(i)) abs(imagzeros(i))^2]);
end

rh=cte*rh;
[rh,rg,h,g]=rh2rg(rh);
