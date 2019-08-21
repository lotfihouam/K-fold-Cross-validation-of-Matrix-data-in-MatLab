%% Read dataset
D.R = (csvread(PAR.dataFname)); 

%%
[row,col,val] = find(D.R);
nzl = length(row);              % Get number of non zero elements
nfolds=10                       % number of folds

%%  fold creation
CVO = cvpartition(nzl,'k',nfolds);
err = zeros(CVO.NumTestSets,1);    compute the error for each fold

for i = 1:CVO.NumTestSets
    iptrain = find(CVO.training(i));
    iptest = find(CVO.test(i));
    %% To Create Test Data
    O.Test=D.R*0;
    for kk = 1 : length(iptest)
        ii = row(iptest(kk)) ;
        jj= col(iptest(kk)) ;
        O.Test(ii,jj) = D.R(ii,jj);
    end
    %% To Create Training Data
    O.Train=D.R*0;
    for kk = 1 : length(iptrain) ;
        ii = row(iptrain(kk)) ;
        jj= col(iptrain(kk)) ;
        O.Train(ii,jj) = D.R(ii,jj);
    end
     
    %%
    D.MaskTrain = (O.Train~=0);  %  1 when Train dataset has non zero value
        
    %% Compute Cosine similarity
    similarity = cosineSim(O.Train);
    
    % Impute ratings
    kind = 'user';
    Pratings =predRatings(O.Train,similarity,kind);
    
    %% Retain only the imputed data for testing
    O.RI = Pratings.*(~D.MaskTrain);        % Will only contains the newly generated (D.MaskTrain mask for train data)

    
    %% ERROR
    err(i) = RMSError(O.RI, O.Test);
    
end
cvErr = sum(err)/nfolds
