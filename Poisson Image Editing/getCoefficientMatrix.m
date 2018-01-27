function coeffA = getCoefficientMatrix(indexes)
%% Enter Your Code Here
    % constructing matrix A for laplacian operator
    % N-by-N matrix, (i,i) = 4; four neighbors = -1;
    maxid = max(indexes(:));
    height = size(indexes,1);
    width = size(indexes,2);
    %coeffA = 4 * eye(maxid);
    coeffA = 4*speye(maxid);
    
    [subx, suby] = find(indexes > 0);
    center = indexes(indexes > 0);
    
    %up
    subx_up(subx > 1) = subx(subx>1) - 1;
    subx_up(subx == 1) = 1;  % deal with first row 
    idx_up = sub2ind(size(indexes), subx_up', suby);
    up_neighbor = indexes(idx_up);
    lst = find((up_neighbor > 0) & (up_neighbor ~= center)); % valid list to fill -1
    a_n1 = center(lst);
    a_n2 = up_neighbor(lst);
    A_idx = sub2ind(size(coeffA), a_n1, a_n2);
    coeffA(A_idx) = -1;
    A_idx = sub2ind(size(coeffA), a_n2, a_n1);
    coeffA(A_idx) = -1;
    
    %down
    subx_down(subx < height) = subx(subx < height) + 1;
    subx_down(subx == height) = height;  % deal with last row 
    idx_down = sub2ind(size(indexes), subx_down', suby);
    down_neighbor = indexes(idx_down);
    lst = find((down_neighbor > 0) & (down_neighbor ~= center)); % valid list to fill -1
    a_n1 = center(lst);
    a_n2 = down_neighbor(lst);
    A_idx = sub2ind(size(coeffA), a_n1, a_n2);
    coeffA(A_idx) = -1;
    A_idx = sub2ind(size(coeffA), a_n2, a_n1);
    coeffA(A_idx) = -1;
    
    %left
    suby_left(suby > 1) = suby(suby>1) - 1;
    suby_left(suby == 1) = 1;  % deal with first column
    idx_left = sub2ind(size(indexes), subx, suby_left');
    left_neighbor = indexes(idx_left);
    lst = find((left_neighbor > 0) & (left_neighbor ~= center)); % valid list to fill -1
    a_n1 = center(lst);
    a_n2 = left_neighbor(lst);
    A_idx = sub2ind(size(coeffA), a_n1, a_n2);
    coeffA(A_idx) = -1;
    A_idx = sub2ind(size(coeffA), a_n2, a_n1);
    coeffA(A_idx) = -1;
    
    %right
    suby_right(suby < width) = suby(suby < width) + 1;
    suby_right(suby == width) = 1;  % deal with last column
    idx_right = sub2ind(size(indexes), subx, suby_right');
    right_neighbor = indexes(idx_right);
    lst = find((right_neighbor > 0) & (right_neighbor ~= center)); % valid list to fill -1
    a_n1 = center(lst);
    a_n2 = right_neighbor(lst);
    A_idx = sub2ind(size(coeffA), a_n1, a_n2);
    coeffA(A_idx) = -1;
    A_idx = sub2ind(size(coeffA), a_n2, a_n1);
    coeffA(A_idx) = -1;
    
end
