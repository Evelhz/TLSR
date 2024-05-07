function [Qs, Qt] =  TLSR1(Xst,Yst,Xs,Xt,Ys,Yt,params)
% configuration for the optimization
iter=0;
maxIter = 50;
lr1 = 5e-5;  
lr2 = 5e-4; 
dim_data = size(Xst,1)*2;
num_class = numel(unique(Ys));
num_train_tot = numel(Yst);
gamma = 1/num_class; % fixed

% hyper parameter
lambda = params.lambda;
beta = params.beta;  

% initialization 
Htr = binaryH(Yst);
X_hat = blkdiag(Xs,Xt);
XXt = X_hat*X_hat';
Q = normr(Htr * X_hat' /(X_hat * X_hat' + 1e-4 * eye(dim_data)));
S = zeros(num_class, num_train_tot);
B = 2 * Htr - ones(num_class, num_train_tot);
T = Htr + B .* S;

% main loop
F_bar = formCellF(zeros(size(Xs)),Xt,Ys,Yt);
F = formCellF(Xs,Xt,Ys,Yt);
while iter <= maxIter
    iter = iter + 1;   
    
    [subGrdQ1,~] = computeSubGrd(Q, F);
    [subGrdQ2,~] = computeSubGrd(Q, F_bar);

    Grd_tr = 2*Q*(XXt)-2*(T*X_hat');
    Grd_all = Grd_tr + beta*(subGrdQ1  - gamma*subGrdQ2);
    Q = Q - lr1*Grd_all;

    Grd_rlx = 2*(1+lambda)*T - 2*Q*X_hat - 2*lambda*(Htr+B.*S);
    T =  T - lr2*Grd_rlx ; 
    S = max((T - Htr) .* B, 0);
end
Qs = Q(:,1:dim_data/2);
Qt = Q(:,dim_data/2+1:end);