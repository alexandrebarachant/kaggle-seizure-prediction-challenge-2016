addpath('./code/') ;
mydir = {'train_1','train_2','train_3','test_1','test_2','test_3','test_1_new','test_2_new','test_3_new'};
srateNew = 128;
windL = 30; % window length in s
windS = 30; % window shift in s
clear safename;
[safename(:,1), safename(:,2), safename(:,3), safename(:,4)] = textread('./code/test_safe.csv', '%d %d %d %d', -1, 'delimiter',',');
for i1 = 1:3
    filtcoef = [];
    intD = []; preD = [];
    names = dir(['../data/' mydir{i1} '/*_1.mat*']);
    for i2=3:size(names,1)
        %fprintf('\n%d-%d',i1,i2-2);
        load(['../data/' mydir{i1} '/' names(i2).name]);
        fprintf('.');
        if dataStruct.sequence>1 && dataStruct.sequence<6
            continue;
        end
        dataStruct.data = dataStruct.data';
        
        % Highpass Filter coefficient
        Fchp = 0.5;
        alpha = (2*pi*Fchp)/double(dataStruct.iEEGsamplingRate);
        
        % 60Hz notch filter
        fs = double(dataStruct.iEEGsamplingRate);               %#sampling rate
        f0 = 60;                %#notch frequency
        fn = fs/2;              %#Nyquist frequency
        freqRatio = f0/fn;      %#ratio of notch freq. to Nyquist freq.
        notchWidth = 0.05;       %#width of the notch
        %Compute zeros
        myzeros = [exp( sqrt(-1)*pi*freqRatio ), exp( -sqrt(-1)*pi*freqRatio )];
        %Compute poles
        mypoles = (1-notchWidth) * myzeros;
        b = poly( myzeros ); %# Get moving average filter coefficients
        a = poly( mypoles ); %# Get autoregressive filter coefficients
        
        
        
        for ii=1:size(dataStruct.data,1)
            dataStruct.data(ii,:) = dataStruct.data(ii,:) - mean(dataStruct.data(ii,:)); % centering
            LPOutput = filter(b,a,dataStruct.data(ii,:));
            HPOutput = zeros(size(LPOutput));
            HPOutput(1) = LPOutput(1);
            for iIndex = 2:length(LPOutput)
                HPOutput(iIndex) = (1 - alpha)*HPOutput(iIndex-1) + LPOutput(iIndex) - LPOutput(iIndex-1);
            end
            dataStruct.data(ii,:) = HPOutput;
        end
        dataStruct.data = double(dataStruct.data(:,dataStruct.iEEGsamplingRate*4:end-dataStruct.iEEGsamplingRate*4)); % removing 4 first and last seconds due to filtering
        dataStruct.data = (resample(dataStruct.data',srateNew,double(dataStruct.iEEGsamplingRate)))';
        if min(range(dataStruct.data')) > 0
            if dataStruct.sequence==6
                preD = [preD dataStruct.data];
            else
                intD = [intD dataStruct.data];
            end
        end
    end
    filtcoef = csp(preD,intD);
    
    names = dir(['../data/' mydir{i1}]);
    for i2=3:size(names,1)
        
        fprintf('\n%d-%d',i1,i2-2);
        load(['../data/' mydir{i1} '/' names(i2).name]);
        dataStruct.data = dataStruct.data';
        for ii=1:size(dataStruct.data,1)
            dataStruct.data(ii,:) = dataStruct.data(ii,:) - mean(dataStruct.data(ii,:)); % centering
            LPOutput = filter(b,a,dataStruct.data(ii,:));
            HPOutput = zeros(size(LPOutput));
            HPOutput(1) = LPOutput(1);
            for iIndex = 2:length(LPOutput)
                HPOutput(iIndex) = (1 - alpha)*HPOutput(iIndex-1) + LPOutput(iIndex) - LPOutput(iIndex-1);
            end
            dataStruct.data(ii,:) = HPOutput;
        end
        dataStruct.data = double(dataStruct.data(:,dataStruct.iEEGsamplingRate*4:end-dataStruct.iEEGsamplingRate*4)); % removing 4 first and last seconds due to filtering
        dataStruct.data = (resample(dataStruct.data',srateNew,double(dataStruct.iEEGsamplingRate)))';
        dataStruct.data = spatFilt(dataStruct.data,filtcoef,1);
        featARCSP = NaN(size(dataStruct.data,1),fix(((length(dataStruct.data(1,:)))-windL*srateNew+windS*srateNew)/(windS*srateNew)),9);
        
        for ii=1:size(dataStruct.data,1)
            fprintf('.');
            dataCh = enframe(dataStruct.data(ii,:),windL*srateNew,windS*srateNew);
            for iii=1:size(dataCh,1)
                featARCSP(ii,iii,:) = SPC_extract_featuresAR(dataCh(iii,:)');
            end
        end
        
        fprintf('.');
        if exist(['./feat/' mydir{i1} '/']) ~= 7
            mkdir(['./feat/' mydir{i1} '/']);
        end
        save(['./feat/' mydir{i1} '/' names(i2).name(1:end-4) '_featARCSP'],'featARCSP');
    end
    
    %save(['../feat/' mydir{i1} '/filtcoef'],'filtcoef');
    mysafeidx = safename(find(safename(:,1)==i1),2);
    
    for i2=mysafeidx'
        fprintf('\n%d-%d',i1,i2);
        load(['../data/' mydir{i1+3} '/' num2str(i1) '_' num2str(i2) '.mat']);
        dataStruct.data = dataStruct.data';
        for ii=1:size(dataStruct.data,1)
            dataStruct.data(ii,:) = dataStruct.data(ii,:) - mean(dataStruct.data(ii,:)); % centering
            LPOutput = filter(b,a,dataStruct.data(ii,:));
            HPOutput = zeros(size(LPOutput));
            HPOutput(1) = LPOutput(1);
            for iIndex = 2:length(LPOutput)
                HPOutput(iIndex) = (1 - alpha)*HPOutput(iIndex-1) + LPOutput(iIndex) - LPOutput(iIndex-1);
            end
            dataStruct.data(ii,:) = HPOutput;
        end
        dataStruct.data = double(dataStruct.data(:,dataStruct.iEEGsamplingRate*4:end-dataStruct.iEEGsamplingRate*4)); % removing 4 first and last seconds due to filtering
        dataStruct.data = (resample(dataStruct.data',srateNew,double(dataStruct.iEEGsamplingRate)))';
        dataStruct.data = spatFilt(dataStruct.data,filtcoef,1);
        featARCSP = NaN(size(dataStruct.data,1),fix(((length(dataStruct.data(1,:)))-windL*srateNew+windS*srateNew)/(windS*srateNew)),9);
        
        for ii=1:size(dataStruct.data,1)
            fprintf('.');
            dataCh = enframe(dataStruct.data(ii,:),windL*srateNew,windS*srateNew);
            for iii=1:size(dataCh,1)
                featARCSP(ii,iii,:) = SPC_extract_featuresAR(dataCh(iii,:)');
            end
        end
        
        fprintf('.');
        if exist(['./feat/' mydir{i1+3} '/']) ~= 7
            mkdir(['./feat/' mydir{i1+3} '/']);
        end
        save(['./feat/' mydir{i1+3} '/' num2str(i1) '_' num2str(i2) '_featARCSP'],'featARCSP');
    end
    
    names = dir(['../data/' mydir{i1+6}]);
    for i2=3:size(names,1)
        
        fprintf('\n%d-%d',i1,i2-2);
        load(['../data/' mydir{i1+6} '/' names(i2).name]);
        dataStruct.data = dataStruct.data';
        for ii=1:size(dataStruct.data,1)
            dataStruct.data(ii,:) = dataStruct.data(ii,:) - mean(dataStruct.data(ii,:)); % centering
            LPOutput = filter(b,a,dataStruct.data(ii,:));
            HPOutput = zeros(size(LPOutput));
            HPOutput(1) = LPOutput(1);
            for iIndex = 2:length(LPOutput)
                HPOutput(iIndex) = (1 - alpha)*HPOutput(iIndex-1) + LPOutput(iIndex) - LPOutput(iIndex-1);
            end
            dataStruct.data(ii,:) = HPOutput;
        end
        dataStruct.data = double(dataStruct.data(:,dataStruct.iEEGsamplingRate*4:end-dataStruct.iEEGsamplingRate*4)); % removing 4 first and last seconds due to filtering
        dataStruct.data = (resample(dataStruct.data',srateNew,double(dataStruct.iEEGsamplingRate)))';
        dataStruct.data = spatFilt(dataStruct.data,filtcoef,1);
        featARCSP = NaN(size(dataStruct.data,1),fix(((length(dataStruct.data(1,:)))-windL*srateNew+windS*srateNew)/(windS*srateNew)),9);
        
        for ii=1:size(dataStruct.data,1)
            fprintf('.');
            dataCh = enframe(dataStruct.data(ii,:),windL*srateNew,windS*srateNew);
            for iii=1:size(dataCh,1)
                featARCSP(ii,iii,:) = SPC_extract_featuresAR(dataCh(iii,:)');
            end
        end
        
        fprintf('.');
        if exist(['./feat/' mydir{i1+6} '/']) ~= 7
            mkdir(['./feat/' mydir{i1+6} '/']);
        end
        save(['./feat/' mydir{i1+6} '/' names(i2).name(1:end-4) '_featARCSP'],'featARCSP');
    end
end
