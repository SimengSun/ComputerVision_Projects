% File name: ransac_est_homography.m
% Author:
% Date created:

function [H, inlier_ind] = ransac_est_homography(x1, y1, x2, y2, thresh)
% Input:
%    y1, x1, y2, x2 are the corresponding point coordinate vectors Nx1 such
%    that (y1i, x1i) matches (x2i, y2i) after a preliminary matching
%    thresh is the threshold on distance used to determine if transformed
%    points agree

% Output:
%    H is the 3x3 matrix computed in the final step of RANSAC
%    inlier_ind is the nx1 vector with indices of points in the arrays x1, y1,
%    x2, y2 that were found to be inliers

    %{
        assume x1,y1 is u; x2, y2 is v;
        v = inv(H)u
    %} 
    iterations = 1000;
    inlier_ind = [];
    H = ones(3,3);
    idx = 1:size(x1,1);
    best_inlier_cnt = 0;
    for i = 1:iterations
        % randomly sample 4 pairs
        sample_id = randsample(idx, 4);
        % compute homography
        H_est = est_homography(x2(sample_id), y2(sample_id), ...
                           x1(sample_id), y1(sample_id));
        % compute inliers
        h = inv(H_est);
        ue_x = h(1,1)*x2 + h(1,2)*y2 + h(1,3); %estimate x 
        ue_y = h(2,1)*x2 + h(2,2)*y2 + h(2,3); %estimate x 
        ue_z = h(3,1)*x2 + h(3,2)*y2 + h(3,3); %estimate x 
        ue_x = ue_x ./ ue_z;
        ue_y = ue_y ./ ue_z;
        
        SSD = (ue_x - x1) .^ 2 + (ue_y - y1) .^ 2;
        inlier_cnt = sum(SSD <= thresh);
        if inlier_cnt > best_inlier_cnt
            best_inlier_cnt = inlier_cnt;
            inlier_ind = [find(SSD<=thresh)];
            H = H_est;
        end
    end
    
end