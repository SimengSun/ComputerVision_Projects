function [ x,y,rmax ] = anms_2( cimg,max_pts )
%ANMS Summary of this function goes here
%   Detailed explanation goes here
id = find(cimg);
metric = cimg(id);
N = max_pts;
N_sort = min([4*N,length(id)]);
[Y,X] = ind2sub(size(cimg),id);
[~,id_sort] = sort(metric,'descend');
id_max = id_sort(1:N_sort);
metric_max = metric(id_max);
Y_max = Y(id_max);
X_max = X(id_max);
dmin = zeros(N_sort,1);
for i=1:N_sort
    ii = find(metric_max>=0.9*metric_max(i));
    id_search = i:ii(end);
    if length(id_search)==1
        dmin(i) = inf;
    else
        X_search = X_max(id_search(2:end));
        Y_search = Y_max(id_search(2:end));
        point = [X_max(i),Y_max(i)];
        [~,D] = knnsearch([X_search,Y_search],point);
        dmin(i)=D;
    end
end
[dmin_sort, id_Dsort] = sort(dmin,'descend');
rmax = dmin_sort(N);
x = X_max(id_Dsort(1:N));
y = Y_max(id_Dsort(1:N));
end

