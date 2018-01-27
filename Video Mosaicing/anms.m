% File name: anms.m
% Author:
% Date created:

function [y, x, rmax] = anms(cimg, max_pts)
% Input:
% cimg = corner strength map
% max_pts = number of corners desired

% Output:
% [x, y] = coordinates of corners
% rmax = suppression radius used to get max_pts corners

% Write Your Own Code Here
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
y = X_max(id_Dsort(1:N));
x = Y_max(id_Dsort(1:N));
end
