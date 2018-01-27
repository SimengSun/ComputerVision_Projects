% File name: mymosaic.m
% Author:
% Date created:

function [img_mosaic] = mymosaic(img_input)
% Input:
%   img_input is a cell array of color images (HxWx3 uint8 values in the
%   range [0,255])
%
% Output:
% img_mosaic is the output mosaic

    frame_num = size(img_input,1);
    img_mosaic = cell(frame_num,1);
    for fn = 1: frame_num
        I_1 = img_input{fn}{1};
        I_2 = img_input{fn}{2};
        I_3 = img_input{fn}{3};
        I_1 = im2double(I_1);
        I_2 = im2double(I_2);
        I_3 = im2double(I_3);
        %if fn == 1
            I1 = rgb2gray(I_1);
            I2 = rgb2gray(I_2);
            I3 = rgb2gray(I_3);

            [ cimg1 ] = corner_detector( I1 );
            [ x1,y1,rmax1 ] = anms( cimg1,1200 );

            [ cimg2 ] = corner_detector( I2 );
            [ x2,y2,rmax2 ] = anms( cimg2,1200 );

            [ cimg3 ] = corner_detector( I3 );
            [ x3,y3,rmax3 ] = anms( cimg3,1200 );

            descs1 = feat_desc(I1, x1, y1);
            descs2 = feat_desc(I2, x2, y2);
            descs3 = feat_desc(I3, x3, y3);


            %%
            match_a = feat_match(descs1, descs2);
            match_b = feat_match(descs3, descs2);
            fa_idx = match_a(match_a ~= -1);
            fb_idx = match_b(match_b ~= -1);

            x1 = x1(match_a > 0);
            xa_2 = x2(fa_idx);
            y1 = y1(match_a > 0);
            ya_2 = y2(fa_idx);

            x3 = x3(match_b > 0);
            xb_2 = x2(fb_idx);
            y3 = y3(match_b > 0);
            yb_2 = y2(fb_idx);

            [H1, inlier_ind_1] = ransac_est_homography(x1, y1, xa_2, ya_2, 0.5);
            [H2, inlier_ind_2] = ransac_est_homography(x3, y3, xb_2, yb_2, 0.5);
        %end
        
        %%
        M = 250;
        N = 100;
        [m, n ,~] = size(I_1);
        backg = zeros([m+2*N, n+2*M,3]);


        inv_H1 = inv(H1);
        inv_H2 = inv(H2);


        [xx, yy] = meshgrid(1:n+2*M, 1:m+2*N);
        xx = xx - M;
        yy = yy - N;
        x1_1 = inv_H1(1,1)*xx(:) + inv_H1(1,2)*yy(:) + inv_H1(1,3);
        y1_1 = inv_H1(2,1)*xx(:) + inv_H1(2,2)*yy(:) + inv_H1(2,3);
        z1_1 = inv_H1(3,1)*xx(:) + inv_H1(3,2)*yy(:) + inv_H1(3,3);
        x1_1 = x1_1 ./ z1_1;
        y1_1 = y1_1 ./ z1_1;

        x1_2 = inv_H2(1,1)*xx(:) + inv_H2(1,2)*yy(:) + inv_H2(1,3);
        y1_2 = inv_H2(2,1)*xx(:) + inv_H2(2,2)*yy(:) + inv_H2(2,3);
        z1_2 = inv_H2(3,1)*xx(:) + inv_H2(3,2)*yy(:) + inv_H2(3,3);
        x1_2 = x1_2 ./ z1_2;
        y1_2 = y1_2 ./ z1_2;

        %%

        NN = size(x1_1, 1);
        x1_1 = round(x1_1);
        y1_1 = round(y1_1);
        for i = 1: NN   
            [ii, jj] = ind2sub(size(backg),i);
            if (x1_1(i)>=1)&&(x1_1(i)<=size(I1,2))&&(y1_1(i)>=1)&&(y1_1(i)<=size(I1,1))
                backg(ii,jj,1) = I_1(y1_1(i),x1_1(i),1);
                backg(ii,jj,2) = I_1(y1_1(i),x1_1(i),2);
                backg(ii,jj,3) = I_1(y1_1(i),x1_1(i),3);
            end   
        end
        
        
        down = repmat(1:-0.05:0.05, [920 1]);
        up = repmat(0.05:0.05:1, [720 1]);
        up_ = repmat(0.05:0.05:1, [920 1]);
        down_ = repmat(1:-0.05:0.05, [720 1]);
        s = 20;

        backg(:, M+1:M+s,1) = backg(:, M+1:M+s,1) .* down;
        backg(:, M+1:M+s,2) = backg(:, M+1:M+s,2) .* down;
        backg(:, M+1:M+s,3) = backg(:, M+1:M+s,3) .* down;
        

        NN = size(x1_2, 1);
        x1_2 = round(x1_2);
        y1_2 = round(y1_2);
        for i = 1: NN   
            [ii, jj] = ind2sub(size(backg),i);
            if (x1_2(i)>=1)&&(x1_2(i)<=size(I1,2))&&(y1_2(i)>=1)&&(y1_2(i)<=size(I1,1))
                backg(ii,jj,1) = I_3(y1_2(i),x1_2(i),1);
                backg(ii,jj,2) = I_3(y1_2(i),x1_2(i),2);
                backg(ii,jj,3) = I_3(y1_2(i),x1_2(i),3);
            end   
        end
        
        backg(:, M+1280+1-s:M+1280,1) = backg(:, M+1280+1-s:M+1280,1) .* up_;
        backg(:, M+1280+1-s:M+1280,2) = backg(:, M+1280+1-s:M+1280,2) .* up_;
        backg(:, M+1280+1-s:M+1280,3) = backg(:, M+1280+1-s:M+1280,3) .* up_;

        I_2 = im2double(I_2);


        backg(N+1:N+720, M+1:M+s,1) = backg(N+1:N+720, M+1:M+s,1) + I_2(:, 1:s,1) .* up;
        backg(N+1:N+720, M+1:M+s,2) = backg(N+1:N+720, M+1:M+s,2) + I_2(:, 1:s,2) .* up;
        backg(N+1:N+720, M+1:M+s,3) = backg(N+1:N+720, M+1:M+s,3) + I_2(:, 1:s,3) .* up;

        backg(N+1:N+720, M+1280+1-s:M+1280,1) = backg(N+1:N+720,M+1280+1-s:M+1280,1) + I_2(:, end-s+1:end,1) .* down_;
        backg(N+1:N+720,M+1280+1-s:M+1280,2) = backg(N+1:N+720,M+1280+1-s:M+1280,2) + I_2(:, end-s+1:end,2) .* down_;
        backg(N+1:N+720, M+1280+1-s:M+1280,3) = backg(N+1:N+720, M+1280+1-s:M+1280,3) + I_2(:, end-s+1:end,3) .* down_;


        backg(N+1:N+720, M+1+s:M+1280-s,1) = I_2(:,s+1:end-s,1);
        backg(N+1:N+720, M+1+s:M+1280-s,2) = I_2(:,s+1:end-s,2);
        backg(N+1:N+720, M+1+s:M+1280-s,3) = I_2(:,s+1:end-s,3);
        

        img_mosaic{fn} = backg;
        disp(['end mosaic frame', num2str(fn)]);
    end
end