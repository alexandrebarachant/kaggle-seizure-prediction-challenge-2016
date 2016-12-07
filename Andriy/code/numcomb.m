function y=numcomb(n,k)

% NUMCOMB   combinatorial number.
%
%           NUMCOMB(n,k) calculates the combinatorial number defined
%           as n!/(k!.(n-k)!). 

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



if n==k,
   y=1;
elseif k==0,
   y=1;
elseif k==1,
   y=n;
else 
   y=fact(n)/(fact(k)*fact(n-k));
end
