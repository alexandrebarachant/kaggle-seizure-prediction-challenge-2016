function [y]=mean_nan(x,DIM,opt)
% MEAN calculates the mean of data elements. 
% 
%  y = mean(x [,DIM] [,opt])
%
% DIM	dimension
%	1 MEAN of columns
%	2 MEAN of rows
% 	N MEAN of  N-th dimension 
%	default or []: first DIMENSION, with more than 1 element
%
% opt	options 
%	'A' arithmetic mean
%	'G' geometric mean
%	'H' harmonic mean
%
% Any combination between opt and DIM is possible. e.g. 
%   y = mean(x,2,'G')
%   calculates the geometric mean of each row in x. 
%
% features:
% - can deal with NaN's (missing values)
% - dimension argument also in Octave
% - compatible to Matlab and Octave
%
% see also: SUMSKIPNAN, MEAN, GEOMEAN, HARMMEAN
%

 
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


% original Copyright by:  KH <Kurt.Hornik@ci.tuwien.ac.at>
% Copyright (c) 2001-2002 by Alois Schloegl <a.schloegl@ieee.org>	

%	$Revision: 1.1 $
%	$Id: mean_nan.m,v 1.1 2005/09/14 17:15:47 bgreene Exp $
%    Copyright (C) 2000-2004 by Alois Schloegl <a.schloegl@ieee.org>	
%    This is part of the NaN-toolbox. For more details see
%    http://www.dpmi.tu-graz.ac.at/~schloegl/matlab/NaN/



if nargin<2
        DIM=[]; 
        opt='a';
end
if (nargin<3)
        if ~isnumeric(DIM), %>=65;%abs('a'), 
                opt=DIM;
                DIM=[]; 
        else
                opt='a';
        end;	
else 
        if ~isnumeric(DIM), %>=65;%abs('a'), 
                tmp=opt;
                opt=DIM;
                DIM=opt;
        end;
end;	
if isempty(DIM), 
        DIM=min(find(size(x)>1));
        if isempty(DIM), DIM=1; end;
end;

%warning('off') % no more division-by-zero warnings

opt = upper(opt); % eliminate old version 

if  (opt == 'A')
	[y, n] = sumskipnan(x,DIM);
        y = y./n;
elseif (opt == 'G')
	[y, n] = sumskipnan(log(x),DIM);
    	y = exp (y./n);
elseif (opt == 'H')
	[y, n] = sumskipnan(1./x,DIM);
    	y = n./y;
else
    	fprintf (2,'mean:  option `%s` not recognized', opt);
end 

