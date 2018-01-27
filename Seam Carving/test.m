clear;
clc;
I = [10 2 3 4;
     7 7 4 3;
     1 4 5 8;
     0 2 4 4];
 [Mx, Tx] = cumMinEngVer(I);
 [Ix, E] = rmVerSeam(I, Mx, Tx);
 [Mxx, Txx] = cumMinEngVer(Ix);
 [Ixx, E] = rmVerSeam(Ix, Mxx, Txx);
%[Ic, T] = carv(I, 0, 2);

 
%%
clear
clc
I = imread('totoro.jpg');
[Ic,T] = carv(I, 30, 60);
imshow(Ic);


%%

h = figure(2); clf;
whitebg(h,[0 0 0]);

fname = 'carving_video.avi';

try
    % VideoWriter based video creation
    h_avi = VideoWriter(fname, 'Uncompressed AVI');
    h_avi.FrameRate = 10;
    h_avi.open();
catch
    % Fallback deprecated avifile based video creation
    h_avi = avifile(fname,'fps',10);
end

for j = 90:-1:0
    imagesc(imread([num2str(j) '.jpg']));
    axis image; axis off; drawnow;
    try 
        h_avi.writeVideo(getframe(gcf));
    catch
        h_avi = addframe(h_avi, getframe(gcf));
    end
end

try
    % VideoWriter based video creation
    h_avi.close();
catch
    % Fallback deprecated avifile based video creation
    h_avi = close(h_avi);
end
clear h_avi;

%%

clear
clc
I = imread('Test2.jpg');
ret = carv_1(I, 50, 50);

%%

h = figure(2); clf;
whitebg(h,[0 0 0]);

fname = 'carving_video_Test2.avi';

try
    % VideoWriter based video creation
    h_avi = VideoWriter(fname, 'Uncompressed AVI');
    h_avi.FrameRate = 10;
    h_avi.open();
catch
    % Fallback deprecated avifile based video creation
    h_avi = avifile(fname,'fps',10);
end
start = size(ret, 2);
for j = start:-1:1
    imagesc(ret{j});
    axis image; axis off; drawnow;
    try 
        h_avi.writeVideo(getframe(gcf));
    catch
        h_avi = addframe(h_avi, getframe(gcf));
    end
end

try
    % VideoWriter based video creation
    h_avi.close();
catch
    % Fallback deprecated avifile based video creation
    h_avi = close(h_avi);
end
clear h_avi;

