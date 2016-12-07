clear all;
addpath('./code/') ;
p = mfilename('fullpath'); fprintf('%s\n',p);
classID = {'1','2','3'};
rand('seed',1);
nFeat = 1632+9+180+144;
[safenameT(:,1), safenameT(:,2), safenameT(:,3), safenameT(:,4)] = textread('./code/test_safe.csv', '%d %d %d %d', -1, 'delimiter',',');
[safenameTr(:,1), safenameTr(:,2), safenameTr(:,3), safenameTr(:,4), safenameTr(:,5)] = textread('./code/train_safe.csv', '%d %d %d %d %d', -1, 'delimiter',',');

for ii=1:size(classID,2)
    folderTr = ['./feat/train_' classID{ii} '/'];
    mysafeidxTr = intersect(intersect(find(safenameTr(:,1)==ii),find(safenameTr(:,5)==1)),find(safenameTr(:,4)==1));
    fprintf('\nPreictal feature loading and imputation');
    newF_P = zeros(150*20,nFeat);
    cur = 1;
    for i=1:length(mysafeidxTr)
        if ~mod(i,100), fprintf(' %2d ', i); end
        
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_feat.mat']);
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_featARCSP.mat']);
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_connFull.mat']);
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_featAR.mat']);
        clear tmp;  tmp(:,:) = featARCSP(1,:,:); featARCSP = tmp;
        for z = 1:size(feat,1)
            for zz = 1:size(feat,3)
                idx = find(isnan(feat(z,:,zz)));
                feat(z,idx,zz) = mean_nan(feat(z,:,zz));
            end
            for zz = 1:size(featAR,3)
                idx = find(isnan(featAR(z,:,zz)));
                featAR(z,idx,zz) = mean_nan(featAR(z,:,zz));
            end
        end
        feat(:,:,[21:52,89:95]) = log(1+feat(:,:,[21:52,89:95]));
        
        for zz = 1:size(featARCSP,2)
            idx = find(isnan(featARCSP(:,zz)));
            featARCSP(idx,zz) = mean_nan(featARCSP(:,zz));
        end
        
        for zz = 1:size(featCon,2)
            idx = find(isnan(featCon(:,zz)));
            featCon(idx,zz) = mean_nan(featCon(:,zz));
        end
        
        for z = 1:size(featARCSP,1)
            tmpF = feat(:,z,:); tmpF = tmpF(:); tmpF_AR = featAR(:,z,:); tmpF_AR = tmpF_AR(:);
            if all(~isnan([tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)]))
                newF_P(cur,:) = [tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)];
                cur = cur + 1;
            end
        end
    end
    newF_P(cur:end,:) = [];
    
    folderTr = ['./feat/test_' classID{ii} '/'];
    mysafeidxT = intersect(intersect(find(safenameT(:,1)==ii),find(safenameT(:,4)==1)),find(safenameT(:,3)==1));
    fprintf('\nPreictal feature loading and imputation');
    newF_P1 = zeros(150*20,nFeat);
    cur = 1;  
    for i=1:length(mysafeidxT)
        if ~mod(i,100), fprintf(' %2d ', i); end
        load([folderTr classID{ii} '_' num2str(safenameT(mysafeidxT(i),2)) '_feat.mat']);
        load([folderTr classID{ii} '_' num2str(safenameT(mysafeidxT(i),2)) '_featARCSP.mat']);
        load([folderTr classID{ii} '_' num2str(safenameT(mysafeidxT(i),2)) '_connFull.mat']);
        load([folderTr classID{ii} '_' num2str(safenameT(mysafeidxT(i),2)) '_featAR.mat']);
        
        clear tmp;  tmp(:,:) = featARCSP(1,:,:); featARCSP = tmp;
        for z = 1:size(feat,1)
            for zz = 1:size(feat,3)
                idx = find(isnan(feat(z,:,zz)));
                feat(z,idx,zz) = mean_nan(feat(z,:,zz));
            end
            for zz = 1:size(featAR,3)
                idx = find(isnan(featAR(z,:,zz)));
                featAR(z,idx,zz) = mean_nan(featAR(z,:,zz));
            end
        end
        feat(:,:,[21:52,89:95]) = log(1+feat(:,:,[21:52,89:95]));
        
        for zz = 1:size(featARCSP,2)
            idx = find(isnan(featARCSP(:,zz)));
            featARCSP(idx,zz) = mean_nan(featARCSP(:,zz));
        end
        for zz = 1:size(featCon,2)
            idx = find(isnan(featCon(:,zz)));
            featCon(idx,zz) = mean_nan(featCon(:,zz));
        end
        for z = 1:size(featARCSP,1)
            tmpF = feat(:,z,:); tmpF = tmpF(:); tmpF_AR = featAR(:,z,:); tmpF_AR = tmpF_AR(:);
            if all(~isnan([tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)]))
                newF_P1(cur,:) = [tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)];
                cur = cur + 1;
            end
        end
    end
    newF_P1(cur:end,:) = [];
    
    
    folderTr = ['./feat/train_' classID{ii} '/'];
    fprintf('\nInterictal feature loading and imputation');
    mysafeidxTr = intersect(intersect(find(safenameTr(:,1)==ii),find(safenameTr(:,5)==1)),find(safenameTr(:,4)==0));
    newF_I = zeros(2000*20,nFeat);
    cur = 1;
    for i=1:length(mysafeidxTr)
        if ~mod(i,100), fprintf(' %2d ', i); end
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_feat.mat']);
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_featARCSP.mat']);
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_connFull.mat']);
        load([folderTr classID{ii} '_' num2str(safenameTr(mysafeidxTr(i),2)) '_' num2str(safenameTr(mysafeidxTr(i),3)) '_featAR.mat']);
        clear tmp;  tmp(:,:) = featARCSP(1,:,:); featARCSP = tmp;
        for z = 1:size(feat,1)
            for zz = 1:size(feat,3)
                idx = find(isnan(feat(z,:,zz)));
                feat(z,idx,zz) = mean_nan(feat(z,:,zz));
            end
            for zz = 1:size(featAR,3)
                idx = find(isnan(featAR(z,:,zz)));
                featAR(z,idx,zz) = mean_nan(featAR(z,:,zz));
            end
            
        end
        feat(:,:,[21:52,89:95]) = log(1+feat(:,:,[21:52,89:95]));
        
        for zz = 1:size(featARCSP,2)
            idx = find(isnan(featARCSP(:,zz)));
            featARCSP(idx,zz) = mean_nan(featARCSP(:,zz));
        end
        for zz = 1:size(featCon,2)
            idx = find(isnan(featCon(:,zz)));
            featCon(idx,zz) = mean_nan(featCon(:,zz));
        end
        
        
        for z = 1:size(featARCSP,1)
            tmpF = feat(:,z,:); tmpF = tmpF(:); tmpF_AR = featAR(:,z,:); tmpF_AR = tmpF_AR(:);
            if all(~isnan([tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)]))
                newF_I(cur,:) = [tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)];
                cur = cur + 1;
            end
        end
    end
    newF_I(cur:end,:) = [];
    
    folderTe = ['./feat/test_' classID{ii} '_new/'];
    myfl = dir([folderTe '*_feat.mat']);
    newF_T = zeros(size(myfl,1)*20,nFeat);
    fprintf('\nTest feature loading and imputation');
    goodidx = [];  cur =1;
    for i=1:size(myfl,1)
        if ~mod(i,100), fprintf(' %2d ', i); end
        
        load([folderTe 'new_' classID{ii} '_' num2str(i) '_feat.mat'])
        load([folderTe 'new_' classID{ii} '_' num2str(i) '_featARCSP.mat'])
        load([folderTe 'new_' classID{ii} '_' num2str(i) '_connFull.mat'])
        load([folderTe 'new_' classID{ii} '_' num2str(i) '_featAR.mat'])
        clear tmp;  tmp(:,:) = featARCSP(1,:,:); featARCSP = tmp;
        for z = 1:size(feat,1)
            for zz = 1:size(feat,3)
                idx = find(isnan(feat(z,:,zz)));
                feat(z,idx,zz) = mean_nan(feat(z,:,zz));
            end
            for zz = 1:size(featAR,3)
                idx = find(isnan(featAR(z,:,zz)));
                featAR(z,idx,zz) = mean_nan(featAR(z,:,zz));
            end
            
        end
        feat(:,:,[21:52,89:95]) = log(1+feat(:,:,[21:52,89:95]));
        
        for zz = 1:size(featCon,2)
            idx = find(isnan(featCon(:,zz)));
            featCon(idx,zz) = mean_nan(featCon(:,zz));
        end
        for zz = 1:size(featARCSP,2)
            idx = find(isnan(featARCSP(:,zz)));
            featARCSP(idx,zz) = mean_nan(featARCSP(:,zz));
        end
        
        for z = 1:size(featARCSP,1)
            tmpF = feat(:,z,:); tmpF = tmpF(:); tmpF_AR = featAR(:,z,:); tmpF_AR = tmpF_AR(:);
            newF_T(cur,:) = [tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)];
            if all(~isnan([tmpF' tmpF_AR' featARCSP(z,:) featCon(z,:)]))
                goodidx = [goodidx cur];
            end
            cur = cur + 1;
        end
    end
    newF_T(cur:end,:) = [];
    fprintf('\nNormalising...');
    nFeatI = round(nFeat/3);
    
    [crap, netnormt1.a, netnormt1.b]=normalise_tr_mv([newF_P(:,1:nFeatI)' newF_P1(:,1:nFeatI)' newF_I(:,1:nFeatI)' newF_T(goodidx,1:nFeatI)']); clear crap;
    [crap, netnormt2.a, netnormt2.b]=normalise_tr_mv([newF_P(:,nFeatI+1:2*nFeatI)' newF_P1(:,nFeatI+1:2*nFeatI)' newF_I(:,nFeatI+1:2*nFeatI)' newF_T(goodidx,nFeatI+1:2*nFeatI)']); clear crap;
    [crap, netnormt3.a, netnormt3.b]=normalise_tr_mv([newF_P(:,2*nFeatI+1:end)' newF_P1(:,2*nFeatI+1:end)'  newF_I(:,2*nFeatI+1:end)' newF_T(goodidx,2*nFeatI+1:end)']); clear crap;
    netnorm.a = [netnormt1.a; netnormt2.a; netnormt3.a];
    netnorm.b = [netnormt1.b; netnormt2.b; netnormt3.b];
    newF_I = normalise_te_mv(newF_I',netnorm.a, netnorm.b);
    newF_P = normalise_te_mv([newF_P'],netnorm.a, netnorm.b);
    newF_P1 = normalise_te_mv([newF_P1'],netnorm.a, netnorm.b);
    newF_T = normalise_te_mv(newF_T',netnorm.a, netnorm.b);
    
    datalabels = [ones(1, size(newF_P,1) + size(newF_P1,1)) -ones(1, size(newF_I,1))];
    save(['train_ALL' classID{ii}], 'newF_I', 'newF_P', 'newF_P1', 'newF_T', 'goodidx' ,'datalabels');
end
