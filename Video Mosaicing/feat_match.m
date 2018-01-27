% File name: feat_match.m
% Author:
% Date created:

function [match] = feat_match(descs1, descs2)
% Input:
%   descs1 is a 64x(n1) matrix of double values
%   descs2 is a 64x(n2) matrix of double values

% Output:
%   match is n1x1 vector of integers where m(i) points to the index of the
%   descriptor in p2 that matches with the descriptor p1(:,i).
%   If no match is found, m(i) = -1

%{
1. find 2 nearest neighbors
2. if dist(1st close) / dist(2nd close) < 0.6, match (1st close)
   else match(-1)
%}  
    descs1 = descs1';
    descs2 = descs2';
    kdtree = KDTreeSearcher(descs2);
    indices = knnsearch(kdtree, descs1, 'K', 2);
    close_1 = descs2(indices(:,1),:);
    close_2 = descs2(indices(:,2),:);
    dist_1 = sum((descs1-close_1).^2,2);
    dist_2 = sum((descs1-close_2).^2,2);
    match = indices(:,1);
    match(dist_1 ./ dist_2 >= 0.65) = -1;
    
end