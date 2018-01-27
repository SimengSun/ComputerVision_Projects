clear
clc

frame_num = 85;
offset = 64;
frame_cell = cell(frame_num,3);
for i = 1:frame_num
    name = ['1_' num2str(i+offset) '.jpg'];
    frame_cell{i}{1} = imread(name);

    name = ['2_' num2str(i+2+offset) '.jpg'];
    frame_cell{i}{2} = imread(name);

    name = ['3_' num2str(i-6+offset) '.jpg'];
    frame_cell{i}{3} = imread(name);

end

fprintf('end read frames');
warning('off','all');
img_mosaic = mymosaic(frame_cell);
myvideomosaic(img_mosaic)