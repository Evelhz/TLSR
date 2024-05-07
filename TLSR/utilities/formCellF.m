function F =  formCellF(Xs,Xt,Ys,Yt)
numClass = numel(unique(Ys));
groupcnt = 1;
for cc=1:numClass
    Xfeats = Xs(:,Ys==cc);
    Xfeatt = Xt(:,Yt==cc);
    Xfeat = blkdiag(Xfeats,Xfeatt);
    F{groupcnt} = Xfeat;
    groupcnt=groupcnt+1;
end