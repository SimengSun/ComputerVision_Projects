function indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)
%% Enter Your Code Here

    % slice the blending area out, and index in vertical direction
    indexes = zeros(targetH, targetW);
    cnt = 1;
    indexes_slice = indexes(offsetY:offsetY + size(mask, 1) - 1,...
        offsetX: offsetX + size(mask,2) - 1);
    lst = find(mask > 0);
    indexes_slice(lst) = 1 : size(lst);
    indexes(offsetY:offsetY + size(mask, 1) - 1,...
        offsetX: offsetX + size(mask,2) - 1) = indexes_slice;
end