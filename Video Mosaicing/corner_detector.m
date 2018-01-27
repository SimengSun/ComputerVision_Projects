% File name: corner_detector.m
% Author:
% Date created:

function [cimg] = corner_detector(img)
% Input:
% img is an image

% Output:
% cimg is a corner matrix

% Write Your Code Here

H = size(img,1);
W = size(img,2);
cimg = zeros(H,W);
points = detectHarrisFeatures(img);
N = points.Count;
location = points.Location;
metric = points.Metric;
location = [floor(location(:,1)),floor(location(:,2))];

for i=1:N
cimg(location(i,2),location(i,1)) = metric(i);
end


end