function d=center(x,op);

%  CENTER  Delay calculation for Wavelet transform alignment.
%
%          CENTER (X, OP) calculates the delay for filter in X 
%          according to the alignment method indicated in OP. 
%          This delay is used by Wavelet transform functions.
%          The value of OP can be:
%              0 : First Absolute Maxima Location
%              1 : Zero delay in analysis (Full for synthesis).
%              2 : Mass center (sum(m*d)/sum(m))
%              3 : Energy center (sum(m^2 *d)/sum(m^2))
%
%          If no output argument is given, then the vector X will
%          be plotted in the current figure, and a color line will be 
%          marking the result.(red: OP=0; green: OP=1; cyan: OP=2; 
%          blue: OP=4)
%
%          See also: WTCENTER, WTMETHOD


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
%       Author: Sergio J. Garcia Galan 
%       e-mail: Uvi_Wave@tsc.uvigo.es
%--------------------------------------------------------


lx=length(x);
l=1:lx;

if op==1
	d=0;
else
	if op==2
		xx=abs(x(:)');
		L=l;
	end
	if op==3
		xx=x(:)'.^2;
		L=l;
	end
	if op==0
		[mx,d]=max(abs(x));
	else 
		
		d=sum(xx.*L)/sum(xx);
	end
end

if nargout==0,
  cad='rgcbk';
  plot(x)
  l=line([d,d],[min(x),max(x)]);
  set(l,'Color',cad(op+1));
end
