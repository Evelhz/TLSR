function [Grd_sub1,obj] =  computeSubGrd(T, F)
num_class = numel(F);
T_rank = size(F{1},1);
eigThd = 0.005;  % default: 0.005
Obj_sub1 = NaN(1, num_class);
Grd_sub1 = NaN(num_class, T_rank, num_class);
 

for k = 1:num_class
    [m, n]=size(T*F{k});
    [U, Sigma, V] = svd((T*F{k}), 'econ');
    Obj_sub1(k) = sum(diag(Sigma));
    F_rank=min([m,n]);
    r = sum(diag(Sigma)<=eigThd);
    if r==0
        r=r+1;
    end
    if F_rank-r==0
        r=r-1;
    end
    RndMat = orth(rand(r, r))*diag(rand(r, 1))*orth(rand(r, r))';
    U1 = U(:, 1:end-r);
    V1 = V(:, 1:end-r);
    U2 = U(:, end-r+1:end);
    V2 = V(:, end-r+1:end);
    W = U2*RndMat*V2';
    UU=U1*V1';
    Grd_sub1(:,:,k) = (UU + W)*(F{k})';
end
Grd_sub1 = sum(Grd_sub1, 3);
obj = sum(Obj_sub1(k));