function [svd_ent, fisher] = inf_theory(win);


Y = zeros(20,length(win)-20);

for i = 1:length(win)-20
    for j = 0:19
    Y(j+1,i) = win(i+j);
    end
end

svds = svd(Y);
norm_svd_vec = svds./sum(svds);




%%%%%%%%%%%%%%% compute svd entropy

svd_ent = -sum(norm_svd_vec.*(log2(norm_svd_vec)));


%%%%%%%%%%%%%%% fisher information

H = 0;

for i = 1:length(norm_svd_vec)-1
    
    H = H + ((norm_svd_vec(i+1) - norm_svd_vec(i))^2 /norm_svd_vec(i));
    
end

fisher = H;
