function [rh,rg,h,g]=rh2rg(rh)

%RH2RG    Calculates all the filters from the synthesis lowpass
%	  in the orthogonal case.
%
%	  [RH,RG,H,G]=RH2RG(RH) begins with the synthesis lowpass
%	  filter (RH) and returns the synthesis highpass filter (RG),
%	  the analysis lowpass filter (H) and the analysis highpass
%	  filter (G).
%	
%	  It is an auxiliary function for orthogonal filters design.

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

% Calculate rg from rh.

for i=1:length(rh)        
	rg(i) = -(-1)^i*rh(length(rh)-i+1);
end  

% Calculate h and g

h=rh(length(rh):-1:1);
g=rg(length(rg):-1:1);
