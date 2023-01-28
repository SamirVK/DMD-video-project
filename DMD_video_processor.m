% This script imports a video file as a 3D matrix and 
% calculates the associated DMD matrix from the SVD and p-inv.
% Author: Samir Karam

clear all; close all; clc

%% Initialize variables and read in color frames from vidObj
frames = 379;
height = 136;
width = 240;
vidObj = VideoReader('monte_carlo_240x136.mp4');
colorVidFrames = read(vidObj, [1 frames]);

%% Convert colour frames to greyscale
for f = 1:frames
    J = rgb2gray(colorVidFrames(:,:,:,f)); 
    gFrames(:,f) = J(:);                     
end
clear colorVidFrames
whos gFrames

%% Print a grayscale frame
% toPrint = reshape(gFrames(:,1),[height,width]); 
% figure
% imshow(toPrint)

%% Create X and Y matrices
X = gFrames(:,1:end-1);
X = double(X);
Y = gFrames(:,2:end);
Y = double(Y);

%% Low-rank SVD
%[U,S,V] = svd(X,'econ');
%X_160 = U(:,1:160)*S(1:160,1:160)*V(:,1:160)';

%% Get the DMD matrix
X_dagger = pinv(X);
A = Y*X_dagger;

%% Get the e'values and e'vectors of A
%[eV, D] = eig(A);

%% Build the DMD video matrix
Z = zeros(height*width,frames);
Z(:,1) = X(:,1);

%% Forecast using DMD matrix 
for n = 1:frames-1
    Z(:,n+1) = A*Z(:,n); 
end

%% Compare end norms
norm(Y(:,end) - Z(:,end))

%% make a movie from the DMD video matrix frames:
toPrint = reshape(Z,[height,width,379]);
figure
f = 1;
while f <= 379
    imshow(mat2gray(toPrint(:,:,f),[0,255]))
    pause(1/60);
    f=f+1;
end