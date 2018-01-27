function resultImg = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY)
%% Enter Your Code Here
    targetH = size(targetImg, 1);
    targetW = size(targetImg,2);
    
    indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY);
    fprintf("end computing indexes\n");
    
    coeffA = getCoefficientMatrix(indexes);
    fprintf("end computing A\n");
    coeffA = sparse(coeffA);
    
    src_r = sourceImg(:,:,1);
    src_g = sourceImg(:,:,2);
    src_b = sourceImg(:,:,3);
    
    tar_r = targetImg(:,:,1);
    tar_g = targetImg(:,:,2);
    tar_b = targetImg(:,:,3);
    
    b_r = getSolutionVect(indexes, src_r, tar_r, offsetX, offsetY);
    fprintf("end computing b for red channel\n");
    b_g = getSolutionVect(indexes, src_g, tar_g, offsetX, offsetY);
    fprintf("end computing b for green channel\n");
    b_b = getSolutionVect(indexes, src_b, tar_b, offsetX, offsetY);
    fprintf("end computing b for blue channel\n");
    
    red = uint8(coeffA \ b_r);
    green = uint8(coeffA \ b_g);
    blue = uint8(coeffA \ b_b);
    
    resultImg = reconstructImg(indexes, red, green, blue, targetImg);
    fprintf("end reconstructing image\n");
    
end