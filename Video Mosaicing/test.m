

ds2 = [1 1; 2 1; 1 2; 2 2; 5 3; 5 4; 6 3; 6 4];
ds1= [2 3; 5 2];

kdtree = KDTreeSearcher(ds2);
indices = knnsearch(kdtree, ds1, 'K', 2);
close_1 = ds2(indices(:,1),:);
close_2 = ds2(indices(:,2),:);
dist_1 = sum((ds1-close_1).^2,2);
dist_2 =sum((ds1-close_2).^2,2);
match = indices(:,1);
match(dist_1 ./ dist_2 >= 0.4) = -1;