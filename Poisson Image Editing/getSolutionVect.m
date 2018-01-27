function solVectorb = getSolutionVect(indexes, source, target, offsetX, offsetY)
%% Enter Your Code Here

    height = size(target,1);
    width = size(target,2);
    laplacian = [0 -1 0; -1 4 -1; 0 -1 0];

    %
    indexes_slice = indexes(offsetY:offsetY+size(source,1) - 1, offsetX:offsetX+size(source,2) - 1);
    mask = double((indexes_slice > 0));
    %source(mask == 0) = 0;
    
    
    %%%%%%%%%%%%
%     target_slice = target(offsetY:offsetY+size(source,1) - 1, ...
%         offsetX:offsetX+size(source,2) - 1);
%     source(mask == 0) = target_slice(indexes_slice == 0);
    %%%%%%%%%%%%
    
    
    conved_src = conv2(source, laplacian, 'same');
    tar = double(target);%conv2(target, laplacian, 'same');
    
    % src laplacian
    solVectorb = conved_src(indexes_slice > 0);
    
    %deal with boundary
    idx = find(indexes > 0);
    [subx, suby] = find(indexes > 0);
    
    %up
    subx_up(subx > 1) = subx(subx>1) - 1;
    subx_up(subx == 1) = 1;  % deal with first row 
    idx_up = sub2ind(size(indexes), subx_up', suby);
    up_neighbor = indexes(idx_up);
    % upper neighbor outside blending area
    up_bdry_idx = idx_up(up_neighbor == 0);
    up_bdry_value = tar(up_bdry_idx);
    solVectorb(up_neighbor == 0) = solVectorb(up_neighbor == 0) + up_bdry_value;
    
    %down
    subx_down(subx < height) = subx(subx < height) + 1;
    subx_down(subx == height) = height;  % deal with last row 
    idx_down = sub2ind(size(indexes), subx_down', suby);
    down_neighbor = indexes(idx_down);
    % down neighbor outside blending area
    down_bdry_idx = idx_down(down_neighbor == 0);
    down_bdry_value = tar(down_bdry_idx);
    solVectorb(down_neighbor == 0) = solVectorb(down_neighbor == 0) + down_bdry_value;
    
    %left
    suby_left(suby > 1) = suby(suby>1) - 1;
    suby_left(suby == 1) = 1;  % deal with first column
    idx_left = sub2ind(size(indexes), subx, suby_left');
    left_neighbor = indexes(idx_left);
    % left neighbor outside blending area
    left_bdry_idx = idx_left(left_neighbor == 0);
    left_bdry_value = tar(left_bdry_idx);
    solVectorb(left_neighbor == 0) = solVectorb(left_neighbor == 0) + left_bdry_value;

    
    %right
    suby_right(suby < width) = suby(suby < width) + 1;
    suby_right(suby == width) = 1;  % deal with last column
    idx_right = sub2ind(size(indexes), subx, suby_right');
    right_neighbor = indexes(idx_right);
    % right neighbor outside blending area
    right_bdry_idx = idx_right(right_neighbor == 0);
    right_bdry_value = tar(right_bdry_idx);
    solVectorb(right_neighbor == 0) = solVectorb(right_neighbor == 0) + right_bdry_value;

end
