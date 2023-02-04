% This script imports a video file as a 3D matrix and 
% calculates the associated DMD matrix from the SVD and p-inv.
% Author: Samir Karam

clear all; close all; clc

%% Initialize variables and read in color frames from vidObj

vidObj = VideoReader('monte_carlo_low.mp4');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frames = vidObj.NumFrames;
height = vidObj.height;
width = vidObj.width;
colorVidFrames = read(vidObj, [1 frames]);

%% Convert colour frames to greyscale
for f = 1:frames
    J = rgb2gray(colorVidFrames(:,:,:,f)); 
    gFrames(:,f) = J(:);                     
end
clear colorVidFrames
whos gFrames

%% Print a grayscale frame from the gFrames matrix
toPrint = reshape(gFrames(:,1),[height,width]); 
figure
imshow(toPrint)

%% GENERALIZED Low-rank SVD (CHOOSE to be less than or equal to 95)
% We want to find the DMD matrix A that satisfies the argmin problem. 
% This involves computing the Moore-Penrose pseudoinverse X_dagger and 
% multiplying on the left by the shifted matrix Y whose columns are the 
% next time step from those in the X matrix.

s = 90; % Desired energy level%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gFrames = double(gFrames);
% X = gFrames(:,1:end-1);
% Y = gFrames(:,2:end);
% X = double(X);
% Y = double(Y);
[U,S,V] = svd(gFrames,'econ');
singular_values = diag(S);
total_energy = sum(diag(S));
cum_energy = 0;
i = 1;
while 1
    cum_energy = cum_energy + singular_values(i);
    percent_energy = cum_energy / total_energy * 100;
    if percent_energy >= s
        break
    end
    i = i + 1;
end
gFrames_s = U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
gFrames_lowrank = U(:,1:i)'*gFrames_s;
X = gFrames_lowrank(:,1:end-1);
Y = gFrames_lowrank(:,2:end);
X_dagger = pinv(X);
A = Y * X_dagger;

%% Get the e'values and e'vectors of A
[eV, D] = eig(A);

%% Build the DMD video matrix
Z = zeros(i,frames);
Z(:,1) = X(:,1);

%% Forecast using DMD matrix 
for n = 1:frames-1
    Z(:,n+1) = A*Z(:,n); 
end

%% Compare end norms
norm(Y(:,end) - Z(:,end))

%% make a movie from the DMD video matrix frames:
% Recover original dimensionality
movie = U(:,1:i) * Z;

toPrint = reshape(movie,[height,width,379]);
figure
f = 1;
while f <= frames
    imshow(mat2gray(toPrint(:,:,f),[0,255]))
    pause(1/120);
    f = f + 1;
end

