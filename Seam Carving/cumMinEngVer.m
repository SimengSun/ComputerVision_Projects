% File Name: cumMinEngVer.m
% Author: Simeng Sun
% Date: Oct. 22nd

function [Mx, Tbx] = cumMinEngVer(e)
% Input:
%   e is the energy map

% Output:
%   Mx is the cumulative minimum energy map along vertical direction.
%   Tbx is the backtrack table along vertical direction.

    n = size(e, 1);
    m = size(e, 2);
    maxd = 2^15;
    
    Mx = zeros(size(e));
    Tbx = zeros(size(e));
    
    Mx(1, :) = e(1, :);
    
    for row = 2 : n
        % pad last row with two large number at both sides
        last = [maxd Mx(row-1, :) maxd];
        % construct three vector, by adding the above three
        current = e(row, :);
        val_candidates = [current + last(1:end-2);
                          current + last(2:end-1);
                          current + last(3:end)];
        % select from three vector the minimum value
        [val, idx] = min(val_candidates);
        % store Mx and Tbx
        Mx(row,:) = val;
        Tbx(row, :) = idx-2;
    end

end