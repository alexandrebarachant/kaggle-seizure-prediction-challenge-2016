function featx=connectivity_features(x,Fs,freq_bands,PSD_window,PSD_overlap)

myidx = [1 4 5 16 18 19 20 30 32 33 34 45 46 55 58 59 66 68 69 70 76 78 79 80 87 88 93 94 96 97 102 103 104 106 108 109 110 113 114 115 118 120];

N_freq_bands=size(freq_bands,1);
ipairs = combnk(1:16,2)';
ipairs = ipairs(:,myidx);
N_pairs=size(ipairs,2);

featLAG=nan(N_freq_bands,N_pairs);
featCOR=nan(N_freq_bands,N_pairs);
featASY=nan(N_freq_bands,N_pairs);
featBSI=nan(N_freq_bands,N_pairs);
featCOH=nan(N_freq_bands,N_pairs);
featMAX=nan(N_freq_bands,N_pairs);

% a) PSD estimate (Welch's periodogram):
win_length=make_odd(PSD_window*Fs);
overlap=ceil(win_length*(PSD_overlap/100));

% assuming just two pairs corresponding to left/right:
N_pxx=floor(max([256 2^nextpow2(win_length)])/2)+1;
N_channels=size(x,1);
X=NaN(N_channels,N_pxx);
X_left=NaN(1,N_pxx); X_right=NaN(1,N_pxx);               

for k=1:N_channels
    x_epoch=x(k,:);
    x_epoch(isnan(x_epoch))=[];
    
    if(length(x_epoch)>=win_length)
        [X(k,:),fp]=pwelch(x_epoch,win_length,overlap,[],Fs);
    end
end

N=size(X,2); Nfreq=2*(N-1); f_scale=(Nfreq/Fs);
X_left=(X(ipairs(1,:),:)).*N_pxx;
X_right=(X(ipairs(2,:),:)).*N_pxx;            
x_filt = nan(size(x));


for n=1:N_freq_bands
    ibandpass=ceil(freq_bands(n,1)*f_scale):floor(freq_bands(n,2)*f_scale);        
    ibandpass=ibandpass+1;
    ibandpass(ibandpass<1)=1; ibandpass(ibandpass>N)=N;    
    [b1,a1]=butter(5,freq_bands(n,2)/(Fs/2),'low');    
    [b2,a2]=butter(5,freq_bands(n,1)/(Fs/2),'high');        

    for p=1:N_channels
        tmp = x(p,:);tmp=filtfilt(b1,a1,tmp); x_filt(p,:)=filtfilt(b2,a2,tmp);
    end
    for p=1:N_pairs
    % EEG symmetry index
        featBSI(n,p)=1-nanmean(abs( (X_left(p,ibandpass) - X_right(p,ibandpass)) ./ ...
                          (X_left(p,ibandpass) + X_right(p,ibandpass)) ));
    % EEG asynchrony index
        cc=abs(corr(X_left(p,ibandpass)',X_right(p,ibandpass)'));
        featASY(n,p)=(cc-featBSI(n,p));

    % EEG correlation
        x1=x_filt(ipairs(1,p),:);
        x2=x_filt(ipairs(2,p),:);
        env1=( hilbert(x1) );
        env2=( hilbert(x2) );



   
        env1=abs( env1 ).^2;
        env2=abs( env2 ).^2;                

        featCOR(n,p)=corr(env1',env2');

        [cc,lag]=xcorr(x1,x2,128,'biased');

        [crap,imax]=max(abs(cc));
        featLAG(n,p)=lag(imax);
        
    end
end
coh = nan(N_pairs,N_pxx);
for p=1:N_pairs        
    x1=x(ipairs(1,p),:);
    x2=x(ipairs(2,p),:);
    x1(isnan(x1))=[];
    x2(isnan(x2))=[];        
    pxx=X(ipairs(1,p),:);
    pyy=X(ipairs(2,p),:);
    [pxy,fp]=cpsd(x1,x2,win_length,overlap,[],Fs);        
    coh(p,:)=(abs(pxy).^2)'./(pxx.*pyy);

    N=size(coh,2); Nfreq=2*(N-1); f_scale=(Nfreq/Fs);
    for n=1:N_freq_bands
        ibandpass=ceil(freq_bands(n,1)*f_scale):floor(freq_bands(n,2)*f_scale);        
        ibandpass=ibandpass+1;
        ibandpass(ibandpass<1)=1; ibandpass(ibandpass>N)=N;    
    
        featCOH(n,p)=nanmean(coh(p,ibandpass));
        [crap,imax]=max(coh(p,ibandpass));
        featMAX(n,p)=fp(ibandpass(imax));
    end
end
montage1= [1:4 9:12; 5:8 13:16]'; idx1 = zeros(1,size(montage1,1));
for n_m =  1:size(montage1,1)
   idx1(n_m) = intersect(find(ipairs(1,:) == montage1(n_m,1)), find(ipairs(2,:)==montage1(n_m,2)));
end

montage2= [1 5 9 13 3 7 11 15; 2 6 10 14 4 8 12 16]'; idx2 = zeros(1,size(montage2,1));
for n_m =  1:size(montage2,1)
   idx2(n_m) = intersect(find(ipairs(1,:) == montage2(n_m,1)), find(ipairs(2,:)==montage2(n_m,2)));
end

montage3= [1:12; 5:16]'; idx3 = zeros(1,size(montage3,1));
for n_m =  1:size(montage3,1)
   idx3(n_m) = intersect(find(ipairs(1,:) == montage3(n_m,1)), find(ipairs(2,:)==montage3(n_m,2)));
end

montage4= [1 5 9 13 2 6 9 14 3 7 11 15; 2 6 10 14 3 7 11 15 4 8 12 16]'; idx4 = zeros(1,size(montage4,1));
for n_m =  1:size(montage4,1)
   idx4(n_m) = intersect(find(ipairs(1,:) == montage4(n_m,1)), find(ipairs(2,:)==montage4(n_m,2)));
end

montage5= [1 2 3 5 6 7 9 10 11; 6 7 8 10 11 12 14 15 16]'; idx5 = zeros(1,size(montage5,1));
for n_m =  1:size(montage5,1)
   idx5(n_m) = intersect(find(ipairs(1,:) == montage5(n_m,1)), find(ipairs(2,:)==montage5(n_m,2)));
end

montage6= [2 3 4 6 7 8 10 11 12; 5 6 7 9 10 11 13 14 15]'; idx6 = zeros(1,size(montage6,1));
for n_m =  1:size(montage6,1)
   idx6(n_m) = intersect(find(ipairs(1,:) == montage6(n_m,1)), find(ipairs(2,:)==montage6(n_m,2)));
end

featx=[nanmean(featLAG(:,idx1)')'  nanmean(featASY(:,idx1)')' nanmean(featBSI(:,idx1)')' nanmean(featMAX(:,idx1)')' nanmean(featCOH(:,idx1)')' nanmean(featCOR(:,idx1)')' ...
       nanmean(featLAG(:,idx2)')'  nanmean(featASY(:,idx2)')' nanmean(featBSI(:,idx2)')' nanmean(featMAX(:,idx2)')' nanmean(featCOH(:,idx2)')' nanmean(featCOR(:,idx2)')' ...
       nanmean(featLAG(:,idx3)')'  nanmean(featASY(:,idx3)')' nanmean(featBSI(:,idx3)')' nanmean(featMAX(:,idx3)')' nanmean(featCOH(:,idx3)')' nanmean(featCOR(:,idx3)')' ...
       nanmean(featLAG(:,idx4)')'  nanmean(featASY(:,idx4)')' nanmean(featBSI(:,idx4)')' nanmean(featMAX(:,idx4)')' nanmean(featCOH(:,idx4)')' nanmean(featCOR(:,idx4)')' ...
       nanmean(featLAG(:,idx5)')'  nanmean(featASY(:,idx5)')' nanmean(featBSI(:,idx5)')' nanmean(featMAX(:,idx5)')' nanmean(featCOH(:,idx5)')' nanmean(featCOR(:,idx5)')' ...
       nanmean(featLAG(:,idx6)')'  nanmean(featASY(:,idx6)')' nanmean(featBSI(:,idx6)')' nanmean(featMAX(:,idx6)')' nanmean(featCOH(:,idx5)')' nanmean(featCOR(:,idx6)')'];
featx = featx(:);