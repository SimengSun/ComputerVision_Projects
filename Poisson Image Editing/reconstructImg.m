function resultImg = reconstructImg(indexes, red, green, blue, targetImg)
%% Enter Your Code Here
    % indexes [h' w']
    % red green blue [1 N]
    % targetImg [h' w' 3] 
    resultImg = targetImg;
    idx = find(indexes > 0);
    img_size = size(indexes,1) * size(indexes,2);
    
    resultImg(idx) = red;
    resultImg(idx+img_size) = green;
    resultImg(idx+2*img_size) = blue;
   
end