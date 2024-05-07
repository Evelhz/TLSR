function [accuracy, Zt,prediction] = classification( Hlabel,Gamma_s, Gamma_t)
% classify process

err = [];
prediction = [];
weight_fusion = linspace(0,1,21);
len_weight_fusion = length(weight_fusion);
errnum = zeros(1,len_weight_fusion);
score_est = cell(1,len_weight_fusion);
for featureid=1:size(Gamma_s,2)
    
    spcode_s = Gamma_s(:,featureid);
    spcode_t = Gamma_t(:,featureid);
    
    for i_score = 1:len_weight_fusion
        fusion = weight_fusion(i_score);
        score_est{i_score} = fusion  * spcode_s + (1-fusion) * spcode_t;
        
        score_gt = Hlabel(:,featureid);
        [maxv_est, maxind_est] = max(score_est{i_score});  % classifying
        [maxv_gt, maxind_gt] = max(score_gt);
        
        if(maxind_est~=maxind_gt)
            errnum(i_score) = errnum(i_score) + 1;
        end
        prediction(featureid,i_score) = maxind_est;
    end
    
end

for i = 1 : len_weight_fusion
    accuracy_tmp(i) = (size(Gamma_s,2)-errnum(i))/size(Gamma_s,2);
end

[accuracy, inds] = max(accuracy_tmp);
prediction = prediction(:,inds);
fusion = weight_fusion(inds);
Zt = fusion  * Gamma_s + (1-fusion) * Gamma_t;
