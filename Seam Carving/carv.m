% File Name: carv.m
% Author: Simeng Sun
% Date: 22nd

function [Ic, T] = carv(I, nr, nc)
% Input:
%   I is the image being resized
%   [nr, nc] is the numbers of rows and columns to remove.
% 
% Output: 
% Ic is the resized image
% T is the transport map

    m = size(I, 1);
    n = size(I, 2);
    
    T = zeros(size(nr+1, nc+1));
    % option table, 0: left neighbor(remove col); 
    % 1: top neighbor(remove row);
    option = zeros(nr+1, nc+1);
    TI = cell(nr+1, nc+1);
    TI{1,1} = I;
    
    
    %% initialize
    option(2:nr+1, 1) = ones(nr, 1); 
    % init first row of T
    Ix = I;
    for i = 2: nc+1
        [Mx, Tx] = cumMinEngVer(genEngMap(TI{1, i-1}));
        [Ix, cost] = rmVerSeam(TI{1,i-1}, Mx, Tx);
        TI{1, i} = Ix;
        T(1, i) = T(1, i-1) + cost;
    end
    % init first col of T
    Iy = I;
    for i = 2: nr+1
        [My, Ty] = cumMinEngHor(genEngMap(TI{i-1, 1}));
        [Iy, cost] = rmHorSeam(Iy, My, Ty);
        TI{i, 1} = Iy;
        T(i, 1) = T(i-1, 1) + cost;
    end
    
    %% propagation
    for r = 2: nr+1
        for c = 2: nc+1
            % try remove col
            [Mx, Tx] = cumMinEngVer(genEngMap(TI{r, c-1}));
            [Ix, cst_x] = rmVerSeam(TI{r, c-1}, Mx, Tx);
            cst_x_total = cst_x + T(r, c-1);
            % try remove row
            [My, Ty] = cumMinEngHor(genEngMap(TI{r-1, c}));
            [Iy, cst_y] = rmHorSeam(TI{r-1, c}, My, Ty);
            cst_y_total = cst_y + T(r-1, c);
            % compare
            if cst_x_total < cst_y_total
                T(r,c) = cst_x_total;
                option(r,c) = 0;
                TI{r,c} = Ix;
            else
                T(r,c) = cst_y_total;
                option(r,c) = 1;
                TI{r,c} = Iy;
            end
        end
        fprintf(['end row ' num2str(r) '\n']);
    end
    
    Ic = TI{nr+1, nc+1};
    
%     i = nr+1;
%     j = nc+1;
%     cnt = 0;
%     while 1
%         filename = [num2str(cnt) '.jpg'];
%         cnt = cnt+1;
%         imwrite(TI{i,j}, filename);
%         fprintf(['end save ' filename '\n']);
%         if i == 1 && j == 1
%             break;
%         end
%         if option(i,j) == 0
%             j = j-1;
%         else
%             i = i -1;
%         end
%     end
    
end