% File Name: cumMinEngHor.m
% Author:
% Date:

function [My, Tby] = cumMinEngHor(e)
% Input:
%   e is the energy map.

% Output:
%   My is the cumulative minimum energy map along horizontal direction.
%   Tby is the backtrack table along horizontal direction.

    n = size(e, 1);
    m = size(e, 2);
    maxd = 2^15;
    
    My = zeros(size(e));
    Tby = zeros(size(e));
    
    My(:, 1) = e(:, 1);
    
    for col = 2 : m
        % pad last row with two large number at both sides
        last = [maxd; My(:, col-1); maxd];
        % construct three vector, by adding the above three
        current = e(:, col);
        val_candidates = [(current + last(1:end-2))';
                          (current + last(2:end-1))';
                          (current + last(3:end))'];
        % select from three vector the minimum value
        [val, idx] = min(val_candidates);
        % store Mx and Tbx
        My(:, col) = val';
        Tby(:, col) = idx'-2;
    end
end