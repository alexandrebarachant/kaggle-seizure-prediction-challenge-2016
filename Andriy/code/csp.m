
%   Coded by James Ethridge and William Weaver

function [result] = CSP(varargin)
    

    if (nargin ~= 2)
        disp('Must have 2 classes for CSP!')
    end
    
    Rsum=0;
    %finding the covariance of each class and composite covariance
    for i = 1:nargin 
        %mean here?
        R{i} = ((varargin{i}*varargin{i}')/trace(varargin{i}*varargin{i}'));%instantiate me before the loop!
        %Ramoser equation (2)
        Rsum=Rsum+R{i};
    end
    
    
   
    %   Find Eigenvalues and Eigenvectors of RC
    %   Sort eigenvalues in descending order
    [EVecsum,EValsum] = eig(Rsum);
    [EValsum,ind] = sort(diag(EValsum),'descend');
    EVecsum = EVecsum(:,ind);
    
    %   Find Whitening Transformation Matrix - Ramoser Equation (3)
        W = sqrt(inv(diag(EValsum))) * EVecsum';
    
    
    for k = 1:nargin
        S{k} = W * R{k} * W'; %       Whiten Data Using Whiting Transform - Ramoser Equation (4)
    end
    
    
    
    % Ramoser equation (5)
    %[U{1},Psi{1}] = eig(S{1});
    %[U{2},Psi{2}] = eig(S{2});
    
    %generalized eigenvectors/values
    [B,D] = eig(S{1},S{2});
    % Simultanous diagonalization
			% Should be equivalent to [B,D]=eig(S{1});
    
    %verify algorithim
    %disp('test1:Psi{1}+Psi{2}=I')
    %Psi{1}+Psi{2}
    
    %sort ascending by default
    %[Psi{1},ind] = sort(diag(Psi{1})); U{1} = U{1}(:,ind);
    %[Psi{2},ind] = sort(diag(Psi{2})); U{2} = U{2}(:,ind);
    [D,ind]=sort(diag(D)); B=B(:,ind);
    
    %Resulting Projection Matrix-these are the spatial filter coefficients
    result = B'*W;
end
