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
% toPrint = reshape(gFrames(:,1),[height,width]); 
% figure
% imshow(toPrint)

%% GENERALIZED Low-rank SVD (CHOOSE between 1 and 100)
% We want to find the DMD matrix A that satisfies the argmin problem. 
% This involves computing the Moore-Penrose pseudoinverse X_dagger and 
% multiplying on the left by the shifted matrix Y whose columns are the 
% next time step from those in the X matrix.

s = 100; % Desired energy level %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gFrames = double(gFrames);

[U,S,V] = svd(gFrames,'econ');
singular_values = diag(S);
total_energy = sum(diag(S));
cum_energy = 0;
i = 1;
while 1
    cum_energy = cum_energy + singular_values(i);
    percent_energy = cum_energy / total_energy * 100;
    if percent_energy > s
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
[eVectors, eValues] = eig(A);
mu = diag(eValues);

%% Plotting Discrete E'values (Code reused from Jason Bramburger's DMD.m)

% % make axis lines
% line = -15:15;
% 
% % Unit circle
% th = linspace(0,2*pi,1000);
% xcos = cos(th);
% ysin = sin(th);
% 
% figure(1)
% plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
% hold on 
% plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
% plot(xcos,ysin,'k--','LineWidth',2) % unit circle
% plot(mu,'.','Color',[1 69/255 79/255],'Markersize',30)
% hold off
% xlabel('Re(\mu)')
% ylabel('Im(\mu)')
% set(gca,'FontSize',16,'Xlim',[-1.2 1.2],'Ylim',[-1.2 1.2])

%% Build the DMD video matrix
Z = zeros(i,frames);
Z(:,1) = X(:,1);

%% Forecast using DMD matrix 
for n = 1:frames-1
    Z(:,n+1) = A*Z(:,n); 
end

%% Construct the "low-rank" DMD frames
% Solve the linear system given by the eigenbasis expansion
c = eVectors \ X(:,1);
dt = 1;

% Determine the indices of the low-rank modes 
modes_low = zeros(i,1);
for j = 1:i
    if abs(mu(j)) <= 1
        modes_low(j) = 1;
    end 
end 

% Construct the x matrix, which is a simple sum of the modes 
x = zeros(i,frames);
for t = 1:frames
    for k = 1:i
        x(:,t) = x(:,t) + c(k) * eVectors(:, k) * mu(k)^((t-1)/dt);
    end 
end

% Construct the X_lowrank DMD matrix for the low-rank modes
x_low = zeros(i,frames);
for t = 1:frames
    for k = 1:i
        if modes_low(k) == 1
            x(:,t) = x(:,t) + c(k) * eVectors(:, k) * mu(k)^((t-1)/dt);
        end
    end 
end

%% Construct the sparse DMD frames and fix negative values for both the
% sparse and low-rank DMD matrices

x_sparse = x - abs(x_low);

% Put back the negative entries from x_sparse into x_low
negatives = x_sparse;
for t = 1:frames
    for j = 1:i
        if x_sparse(j,t) < 0
            negatives(j,t) = 0;
        end
    end
end
x_low = negatives + abs(x_low);

% Then subtract them from x_sparse
x_sparse = x_sparse - negatives;

%% Print frames from matrices
subplot(2,2,1)
snapshot = U(:,1:i) * X;
toPrint = reshape(snapshot,[height,width,frames-1]);
imshow(mat2gray(toPrint(:,:,342),[0,255]))
title('Original')

subplot(2,2,2)
x_real = real(x);
snapshot = U(:,1:i) * x_real;
toPrint = reshape(snapshot,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,342),[0,255]))
title('Reconstructed')

subplot(2,2,3)
x_low_real = real(x_low);
snapshot = U(:,1:i) * x_low_real;
toPrint = reshape(snapshot,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,342),[0,255]))
title('Foreground')

subplot(2,2,4)
x_sparse_real = real(x_sparse);
snapshot = U(:,1:i) * x_sparse_real;
toPrint = reshape(snapshot,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,342),[0,255]))
title('Background')

%% Make a movie from the DMD video matrix frames:
% Recover original dimensionality

% movie = U(:,1:i) * Z;
% 
% toPrint = reshape(movie,[height,width,frames]);
% figure(2)
% f = 1;
% while f <= frames
%     imshow(mat2gray(toPrint(:,:,f),[0,255]))
%     pause(1/120);
%     f = f + 1;
% end

