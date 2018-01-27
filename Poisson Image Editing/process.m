clear;
clc;

sourceImg = imread('woman.jpeg');
sourceImg = imresize(sourceImg,0.2);
size(sourceImg)
targetImg = imread('cliff.jpeg');

offsetX = 115;
offsetY = 150;
% offsetX = 480;
% offsetY = 30;

%load mask;
mask = maskImage(sourceImg);

result = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY);

imshow(result);

%%

clear;
clc;

sourceImg = imread('1.jpg');
targetImg = imread('grass.jpg');

offsetX = 1;
offsetY = 1;

load mask_cup;

result = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY);

imshow(result);


%%

clear;
clc;

sourceImg = imread('SourceImage.jpg');
sourceImg = imresize(sourceImg, 0.35);
targetImg = imread('TargetImage.jpg');

offsetX = 250;
offsetY = 180;

mask = maskImage(sourceImg);

result = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY);

imshow(result);

%%
clear;
clc;

sourceImg = imread('facechange.jpg');
size(sourceImg)
targetImg = imread('jon_1.jpg');

offsetX = 1;
offsetY = 1;

%load mask;
mask = maskImage(sourceImg);

result = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY);

imshow(result);



%% test 

clear;
clc;

mask = [0 0 0 0 0;
        0 0 1 1 1;
        0 1 1 0 0;
        0 0 1 0 0];
    
target = [2 2 2 2 2 2 2 2;
          3 3 3 3 3 3 3 3;
          4 4 4 4 4 4 4 4;
          5 5 5 5 5 5 5 5;
          6 6 6 6 6 6 6 6];

source = [10 10 10 10 10;
          10 10 20 20 20;
          10 20 20 10 10 ;
          10 10 20 10 10];

h = 5;
w = 8;
offsetX = 2;
offsetY = 1;

indexes = getIndexes(mask, h, w, offsetX, offsetY);
disp(indexes)
A = getCoefficientMatrix(indexes);
disp(A);
laplacian = [0 -1 0; -1 4 -1; 0 -1 0];
disp(conv2(source, laplacian, 'same'));
disp(target);
b = getSolutionVect(indexes, source, target, offsetX, offsetY);
disp(b);