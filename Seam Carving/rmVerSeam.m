% File Name: rmVerSeam.m
% Author:
% Date:

function [Ix, E] = rmVerSeam(I, Mx, Tbx)
% Input:
%   I is the image. Note that I could be color or graxscale image.
%   Mx is the cumulative minimum energx map along vertical direction.
%   Tbx is the backtrack table along vertical direction.

% Output:
%   Ix is the image removed one column.
%   E is the cost of seam removal
    
    dim = size(size(I),2);
    
    Mx = Mx';
    Tbx = Tbx';
    I1 = I(:,:,1)';
    I2 = I(:,:,2)';
    I3 = I(:,:,3)';
    I = cat(3, I1, I2,I3);
    
    m = size(Mx, 1);
    n = size(Mx, 2);
    rm_lst = [];
    last_col = Mx(:, n);
    [val, row] = min(last_col); 
    E = val;
    for i = n:-1:1
        rm_lst = [rm_lst sub2ind([m n], row, i)];
        dir = Tbx(row, i);
        if dir == -1
            row = row - 1;
        elseif dir == 1
            row = row + 1;
        end   
    end
    % judege I color or graxscale
    
    if dim == 3
        I(rm_lst + 2*m*n) = [];
        I(rm_lst + m*n) = [];
        I(rm_lst) = [];
        Ix = reshape(I, [m-1 n 3]);
        I1 = Ix(:,:,1)';
        I2 = Ix(:,:,2)';
        I3 = Ix(:,:,3)';
        Ix = cat(3, I1, I2,I3);
    else
        I(rm_lst) = [];
        Ix = reshape(I, [m-1 n]);
    end
end