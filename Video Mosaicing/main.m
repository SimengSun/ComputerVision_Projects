clear
clc

I_1 = imread('1_75.jpg');
I_2 = imread('2_77.jpg');
I_1 = im2double(I_1);
I1 = rgb2gray(I_1);
[h1, w1] = size(I1);
I2 = rgb2gray(I_2);
[ cimg1 ] = corner_detector( I1 );
[ x1,y1,rmax1 ] = anms( cimg1,800 );
[ cimg2 ] = corner_detector( I2 );
[ x2,y2,rmax2 ] = anms( cimg2,800 );
descs1 = feat_desc(I1, x1, y1);
descs2 = feat_desc(I2, x2, y2);
match = feat_match(descs1, descs2);
f_idx = match(match ~= -1);
x1 = x1(match > 0);
x2 = x2(f_idx);
y1 = y1(match > 0);
y2 = y2(f_idx);

[H, inlier_ind] = ransac_est_homography(x1, y1, ...
x2, y2, 0.5);
% figure; ax = axes;
% showMatchedFeatures(I1,I2,[x1,y1],[x2,y2],'montage','Parent',ax);

inl_x1 = x1(inlier_ind);
inl_y1 = y1(inlier_ind);
inl_x2 = x2(inlier_ind);
inl_y2 = y2(inlier_ind);


%showMatchedFeatures(I1,I2,[inl_x1,inl_y1],[inl_x2,inl_y2],'montage','Parent',ax);

inv_H = inv(H);
M = 250;
N = 100;

[m, n ,~] = size(I_1);
backg = zeros([m+2*N, n+2*M,3]);

[xx, yy] = meshgrid(1:n+2*M, 1:m+2*N);
xx = xx - M;
yy = yy - N;
x1_ = inv_H(1,1)*xx(:) + inv_H(1,2)*yy(:) + inv_H(1,3);
y1_ = inv_H(2,1)*xx(:) + inv_H(2,2)*yy(:) + inv_H(2,3);
z1_ = inv_H(3,1)*xx(:) + inv_H(3,2)*yy(:) + inv_H(3,3);
x1_ = x1_ ./ z1_;
y1_ = y1_ ./ z1_;

%%

% I1_(:,:,1) = reshape(interp2(I_1(:,:,1), x1_(:), y1_(:)), [h1,w1]);
% I1_(:,:,2) = reshape(interp2(I_1(:,:,2), x1_(:), y1_(:)), [h1,w1]);
% I1_(:,:,3) = reshape(interp2(I_1(:,:,3), x1_(:), y1_(:)), [h1,w1]);
% 
% 
% imshow(I1_);

%%

M = 250;
N = 50;

NN = size(x1_, 1);
x1_ = round(x1_);
y1_ = round(y1_);
for i = 1: NN   
    [ii, jj] = ind2sub(size(backg),i);
    if (x1_(i)>=1)&&(x1_(i)<=size(I1,2))&&(y1_(i)>=1)&&(y1_(i)<=size(I1,1))
        backg(ii,jj,1) = I_1(y1_(i),x1_(i),1);
        backg(ii,jj,2) = I_1(y1_(i),x1_(i),2);
        backg(ii,jj,3) = I_1(y1_(i),x1_(i),3);
    end   
end


% backg(:,:,1) = reshape(interp2(I_1(:,:,1), x1_(:), y1_(:), 'spline', 0), [m+2*N, n+2*M]);
% backg(:,:,2) = reshape(interp2(I_1(:,:,2), x1_(:), y1_(:), 'spline', 0), [m+2*N, n+2*M]);
% backg(:,:,3) = reshape(interp2(I_1(:,:,3), x1_(:), y1_(:), 'spline', 0), [m+2*N, n+2*M]);


%%
% I1_ = I1_(:);
idx = find(isnan(I1_));
I1_ = im2uint8(I1_);
imshow(I1_);
disp(size(idx));
I1_(idx) = I_2(idx);
I1_ = reshape(I1_, [h1 w1 3]);

imshow(I1_)

