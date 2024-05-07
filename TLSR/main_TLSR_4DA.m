clc;clear
addpath utilities\
global acc_TLSR1 acc_TLSR2 std_TLSR1 std_TLSR2
acc_TLSR1 = [];std_TLSR1 = [];
acc_TLSR2 = [];std_TLSR2 = [];
fprintf('TLSR starts...\n');
Ex_CaltechOffice('amazon','Caltech10')
Ex_CaltechOffice('amazon','dslr')
Ex_CaltechOffice('amazon','webcam')
Ex_CaltechOffice('Caltech10','amazon')
Ex_CaltechOffice('Caltech10','dslr')
Ex_CaltechOffice('Caltech10','webcam')
Ex_CaltechOffice('dslr','amazon')
Ex_CaltechOffice('dslr','Caltech10')
Ex_CaltechOffice('dslr','webcam')
Ex_CaltechOffice('webcam','amazon')
Ex_CaltechOffice('webcam','Caltech10')
Ex_CaltechOffice('webcam','dslr')
fprintf('TLSR1 Average Accuracy: %.2f±%.2f\n',mean(acc_TLSR1),mean(std_TLSR1));
fprintf('TLSR2 Average Accuracy: %.2f±%.2f\n',mean(acc_TLSR2),mean(std_TLSR2));
function Ex_CaltechOffice(src,tgt)
global acc_TLSR1 acc_TLSR2 std_TLSR1 std_TLSR2
 load(['data/' src '_SURF_L10.mat']);     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
fts = zscore(fts,1);
fts = normr(fts);
Source = fts;               clear fts
Source_lbl = labels;           clear labels

load(['data/' tgt '_SURF_L10.mat']);     % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
fts = zscore(fts,1);
fts = normr(fts);
Target = fts;               clear fts
Target_lbl = labels;            clear labels;
str_print  = ['Source (' src ') ---> Target (' tgt ') '];
fprintf('| %-40s|',str_print);
%--------------------II. run experiments-----------------------------------
round = 10; %
if strcmp(src,'amazon')==1
    nPerClassS = 20;
else
    nPerClassS = 8;
end
nPerClassT = 3;
train_num = nPerClassS + nPerClassT;

for iter = 1 : round
    inds = split(Source_lbl, nPerClassS);
    [inds2,indsTest] = split(Target_lbl, nPerClassT);
    
    Xr1 = Source(inds,:); Yr1 = Source_lbl(inds);
    Xr2 = Target(inds2,:);Yr2 = Target_lbl(inds2);
    
    Xs = Xr1'; Ys = Yr1;
    Xt = Xr2'; Yt = Yr2;
    Xst = [Xs, Xt];Yst = [Ys;Yt];
    
    TtData = (Target(indsTest,:))';
    TtLabel = (Target_lbl(indsTest));
   %% TLSR1
    params = [];
    params.lambda = 500; % For auxiliary variables
    params.beta = 0.01; % low-rank regularization
    params.train_num = train_num;
   
    % TLSR trainig
    tic
    [Qs, Qt] = TLSR(Xst,Yst,Xs,Xt,Ys,Yt,params);
    TrTime(iter) = toc;
    % TSLR testing 
    tic
    [Accuracy_TLSR1(iter),~,~] = classification(binaryH(TtLabel), Qs*TtData, Qt*TtData);
    TtTime(iter) = toc;
   %% TCA & TLSR2
    options = [];
    options.lambda =0.1;
    options.gamma = 0.1;
    options.T = 10;
    options.dim =10;
    options.kernel_type = 'primal';
    % dim= pool(tuneDim);
    [Zs2,Zt2,~] = TCA(Xst',TtData',options);

    Zs2 = Zs2';Zt2 = Zt2';
    ns = size(Xs,2);
    Zs_s = Zs2(:,1:ns);
    Zs_t = Zs2(:,ns+1:end);
    Xs = [Xs;Zs_s];
    Xt = [Xt;Zs_t];
    Xst = [Xs,Xt];
    TtData = [TtData;Zt2];
    [Qs, Qt] = TLSR(Xst,Yst,Xs,Xt,Ys,Yt,params);
    tic
    [Accuracy_TLSR2(iter),~,~] = classification(binaryH(TtLabel), Qs*TtData, Qt*TtData);
    TtTime(iter) = toc;

end
ave_acc_TLSR1 = mean(Accuracy_TLSR1*100);std_acc_TLSR1 =std(Accuracy_TLSR1*100);
ave_acc_TLSR2 = mean(Accuracy_TLSR2*100);std_acc_TLSR2 =std(Accuracy_TLSR2*100);
acc_TLSR1 = [acc_TLSR1 ave_acc_TLSR1];std_TLSR1 = [std_TLSR1 std_acc_TLSR1];
acc_TLSR2 = [acc_TLSR2 ave_acc_TLSR2];std_TLSR2 = [std_TLSR2 std_acc_TLSR2];
fprintf(' TLSR1: %.2f±%.2f | ',ave_acc_TLSR1,std_acc_TLSR1);
fprintf(' TLSR2: %.2f±%.2f |\n',ave_acc_TLSR2,std_acc_TLSR2);
end


function [idx1 idx2] = split(Y,nPerClass, ratio)
% [idx1 idx2] = split(X,Y,nPerClass)
idx1 = [];  idx2 = [];
for C = 1 : max(Y)
    idx = find(Y == C);
    rand('state',sum(100*clock));
    % rand(1);
    rn = randperm(length(idx));
    if exist('ratio')
        nPerClass = floor(length(idx)*ratio);
    end
    idx1 = [idx1; idx( rn(1:min(nPerClass,length(idx))) ) ];
    idx2 = [idx2; idx( rn(min(nPerClass,length(idx))+1:end) ) ];
end
end