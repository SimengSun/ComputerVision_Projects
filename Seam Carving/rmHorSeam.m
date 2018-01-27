% File Name: rmHorSeam.m
% Author:
% Date:

function [Iy, E] = rmHorSeam(I, My, Tby)
% Input:
%   I is the image. Note that I could be color or grayscale image.
%   My is the cumulative minimum energy map along horizontal direction.
%   Tby is the backtrack table along horizontal direction.

% Output:
%   Iy is the image removed one row.
%   E is the cost of seam removal
    
    dim = size(size(I),2);

    m = size(My, 1);
    n = size(My, 2);
    rm_lst = [];
    last_col = My(:, n);
    [val, row] = min(last_col); 
    E = val;
    for i = n:-1:1
        rm_lst = [rm_lst sub2ind([m n], row, i)];
        dir = Tby(row, i);
        if dir == -1
            row = row - 1;
        elseif dir == 1
            row = row + 1;
        end   
    end
    % judege I color or grayscale
    
    if dim == 3
        I(rm_lst + 2*m*n) = [];
        I(rm_lst + m*n) = [];
        I(rm_lst) = [];
        Iy = reshape(I, [m-1 n 3]);
    else
        I(rm_lst) = [];
        Iy = reshape(I, [m-1 n]);
    end
end