function [video_mosaic] = myvideomosaic(img_mosaic)
    
    h = figure(2); clf;
    whitebg(h,[0 0 0]);

    fname = 'test_hmg_12.avi';

    try
        % VideoWriter based video creation
        h_avi = VideoWriter(fname, 'Uncompressed AVI');
        h_avi.FrameRate = 10;
        h_avi.open();
    catch
        % Fallback deprecated avifile based video creation
        h_avi = avifile(fname,'fps',10);
    end
    f_num = size(img_mosaic, 1);
    for j = 1:f_num
        imagesc(img_mosaic{j});
        axis image; axis off; drawnow;
        try 
            h_avi.writeVideo(getframe(gcf));
        catch
            h_avi = addframe(h_avi, getframe(gcf));
        end
        disp(['end frame ', num2str(j)]);
    end

    try
        % VideoWriter based video creation
        h_avi.close();
    catch
        % Fallback deprecated avifile based video creation
        h_avi = close(h_avi);
    end
    clear h_avi;
end