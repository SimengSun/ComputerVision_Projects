function mask = maskImage(Img)
%% Enter Your Code Here
    
    him = imshow(Img);
    e = imfreehand(gca);
    mask = createMask(e, him);
    save('mask', 'mask');
end

