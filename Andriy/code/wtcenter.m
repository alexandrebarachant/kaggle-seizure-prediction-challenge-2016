function d=wtcenter(x);

%  WTCENTER Calculates the delay of filters for alignment.
%
%           WTCENTER (X) calculates the integer aproximation
%           of delay for filter X using the method set with
%           the WTMETHOD function, for alignment operations
%           in Wavelet transforms.
%
%           For a non integer value, use the CENTER function.
%
%           See also: WTMETHOD, CENTER, WT
%

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

global WTCENTERMETHOD

if size(WTCENTERMETHOD)==[0,0]
	WTCENTERMETHOD=0;
end

if WTCENTERMETHOD>3 | WTCENTERMETHOD<0
	WTCENTERMETHOD=0
end

d=floor(center(x,WTCENTERMETHOD));

% (Another long function !!!)

