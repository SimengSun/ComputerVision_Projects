% File name: feat_desc.m
% Author:
% Date created:

function [descs] = feat_desc(img, x, y)
% Input:
%    img = double (height)x(width) array (grayscale image) with values in the
%    range 0-255
%    x = nx1 vector representing the column coordinates of corners
%    y = nx1 vector representing the row coordinates of corners

% Output:
%   descs = 64xn matrix of double values with column i being the 64 dimensional
%   descriptor computed at location (xi, yi) in im

% Write Your Code Here
    
    G = [2,4,5,4,2;4,9,12,9,4;5,12,15,12,5;4,9,12,9,4;2,4,5,4,2];
    G = 1/159.*G;
    
    img = im2double(img);
    
    % pad whole image with 20 pts (l,r,u,d)
    h = size(img, 1);
    w = size(img, 2);
    I = padarray(img, [20 20]);
    
    
    % select windows for each point
    x = x+20;
    y = y+20;
    descs = zeros(64, size(x,1));
    for i = 1:size(x,1)
        s_x = x(i) - 20;
        s_y = y(i) - 20;
        e_x = x(i) + 20;
        e_y = y(i) + 20;
        window = I(s_y:e_y, s_x:e_x);
        % filter window with 2d gaussian
        window = conv2(window, G, 'same');
        % subsample by 5x5
        desc = zeros(64,1);
        cnt = 1;
        for j = 1: 5: 36
           for k = 1: 5: 36
               w = window(j:j+4, k:k+4);
               desc(cnt) = max(w(:));
               cnt = cnt+1;
           end
        end
        descs(:,i) = (desc - mean(desc) ) ./ std(desc);
          
    end
   

end