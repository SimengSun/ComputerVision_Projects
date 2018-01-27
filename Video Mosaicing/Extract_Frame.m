clc;
clear;
video_file='2.3.MOV';
video=VideoReader(video_file);
frame_number=floor(video.Duration * video.FrameRate);

for i=1:frame_number
    image_name=strcat('3_',num2str(i));
    image_name=strcat(image_name,'.jpg');
    I=read(video,i);            
    imwrite(I,image_name,'jpg');      
    I=[];
end