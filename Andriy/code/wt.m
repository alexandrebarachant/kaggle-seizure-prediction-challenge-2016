function y=wt(x,h,g,k,del1,del2)

% WT   Discrete Wavelet Transform.
%
%      WT(X,H,G,K) calculates the wavelet transform of vector X.
%      If X is a matrix (2D), WT will calculate the one dimensional
%      wavelet transform of each row vector. The second argument H
%      is the lowpass filter and the third argument G the highpass
%      filter.
%
%      The output vector contains the coefficients of the DWT ordered
%      from the low pass residue at scale K to the coefficients
%      at the lowest scale, as the following example ilustrates:
%
%      Output vector (k=3):
%
%      [------|------|------------|------------------------]
%	  |	  |	   |		    |
%	  |	  |	   |		    `-> 1st scale coefficients
% 	  |	  |	   `-----------> 2nd scale coefficients
%	  |	  `--------------------> 3rd scale coefficients
%	  `----------------> Low pass residue  at 3rd scale
%
%
%      If X is a matrix, the result will be another matrix with
%      the same number of rows, holding each one its respective
%      transformation.
%
%      WT (X,H,G,K,DEL1,DEL2) calculates the wavelet transform of
%      vector X, but also allows the users to change the alignment
%      of the outputs with respect to the input signal. This effect
%      is achieved by setting to DEL1 and DEL2 the delays of H and
%      G respectively. The default values of DEL1 and DEL2 are
%      calculated using the function WTCENTER.
%
%      See also:  IWT, WT2D, IWT2D, WTCENTER, ISPLIT.


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
%      Authors: Sergio J. Garcia Galan
%               Cristina Sanchez Cabanelas
%       e-mail: Uvi_Wave@tsc.uvigo.es
%--------------------------------------------------------


% -----------------------------------
%    CHECK PARAMETERS AND OPTIONS
% -----------------------------------

h=h(:)';	% Arrange the filters so that they are row vectors.
g=g(:)';

if length(x)<2^k
	disp('The scale is too high. The maximum for the signal is:')
	floor(log2(length(x)))
	return
end

[liy,lix]=size(x);

trasp=0;
if lix==1        	% And arrange the input vector to a row if
	x=x';           % it's not a matrix.
	trasp=1;	% (and take note of it)
	[liy,lix]=size(x);
end


%--------------------------
%    DELAY CALCULATION
%--------------------------

% Calculate delays as the C.O.E. of the filters
dlp=wtcenter(h);
dhp=wtcenter(g);

if rem(dhp-dlp,2)~=0		% difference between them.
	dhp=dhp+1;		% must be even
end;

if nargin==6,			% Other experimental filter delays
	dlp=del1;		% can be forced from the arguments
	dhp=del2;
end;

%------------------------------
%    WRAPPAROUND CALCULATION
%------------------------------
llp=length(h);                	% Length of the lowpass filter
lhp=length(g);                	% Length of the highpass filter.

L=max([lhp,llp,dlp,dhp]);	% The number of samples for the
				% wrapparound. Thus, we should need to
				% move along any L samples to get the
				% output wavelet vector phase equal to
				% original input phase.


%------------------------------
%     START THE ALGORITHM
%------------------------------

for it=1:liy,		% For every row of the input matrix...
			% (this makes one wavelet transform
			% for each of the rows of the input matrix)
	tm=[];
	t=x(it,:);			% Copy the vector to transform.

	for i=1:k			% For every scale (iteration)...
		lx=length(t);
		if rem(lx,2)~=0    	% Check that the number of samples
			t=[t,0];       	% will be even (because of decimation).
			lx=lx+1;
		end
		tp=t;		       	% Build wrapparound. The input signal
		pl=length(tp);	       	% can be smaller than L, so it can
		while L>pl		% be necessary to repeat it several
			tp=[tp,t];	% times
			pl=length(tp);
		end

		t=[tp(pl-L+1:pl),t,tp(1:L)];	% Add the wrapparound.

		yl=conv(t,h);	       	% Then do lowpass filtering ...
		yh=conv(t,g);         	% ... and highpass filtering.

		yl=yl((dlp+1+L):2:(dlp+L+lx));    % Decimate the outputs
		yh=yh((dhp+1+L):2:(dhp+L+lx));    % and leave out wrapparound

		tm=[yh,tm];            	% Put the resulting wavelet step
					% on its place into the wavelet
					% vector...
		t=yl;                  	% ... and set the next iteration.
	end

	y(it,:)=[t,tm];		       	% Wavelet vector (1 row vector)


end				% End of the "rows" loop.

%------------------------------
%    END OF THE ALGORITHM
%------------------------------

if trasp==1		       	% If the input data was a column vector
	y=y';		       	% then transpose it.
end
