% -------------------------------------------------------------------------
% This script imports a video file as a 3D matrix, calculates the low rank
% DMD matrix using a low or full rank SVD approximation, and incorporates
% a low\high pass filter to decompose the input signal into separate 
% foreground and background video streams.
% 
% Works with .mp4 files
% -------------------------------------------------------------------------
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

%% GENERALIZED Low-rank SVD (CHOOSE between 1 and 100)
% Mysteriously, some levels do not work!
% We want to find the DMD matrix A that satisfies the argmin problem. 
% This involves computing the Moore-Penrose pseudoinverse X_dagger and 
% multiplying on the left by the shifted matrix Y whose columns are the 
% next time step from those in the X matrix.

% Desired energy level %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = 100; 
gFrames = double(gFrames);

[U,S,V] = svd(gFrames,'econ');
singular_values = diag(S);
total_energy = sum(singular_values);
cum_energy = 0;
i = 1;
while i < frames
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
[eVectors, eValues] = eig(A);
mu = diag(eValues);

%% Plotting Discrete E'values (Code reused from Jason Bramburger's DMD.m)

% make axis lines
line = -15:15;

% Unit circle
th = linspace(0,2*pi,1000);
xcos = cos(th);
ysin = sin(th);

figure(1)
subplot(1,2,1)
plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on 
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(xcos,ysin,'k--','LineWidth',2) % unit circle
plot(mu,'.','Color',[1 69/255 79/255],'Markersize',10)
hold off
xlabel('Re(\mu)')
ylabel('Im(\mu)')
set(gca,'FontSize',16,'Xlim',[-1.2 1.2],'Ylim',[-1.2 1.2])
ZoomPlot()

subplot(1,2,2)
plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on 
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega),imag(omega),'.b','MarkerSize', 10)
hold off
xlabel('Re(\omega)')
ylabel('Im(\omega)')
set(gca,'FontSize',16,'Xlim',[-1.2 1.2],'Ylim',[-1.2 1.2])
ZoomPlot()

%% Build the DMD video matrix and forecast
Z = zeros(i,frames);
Z(:,1) = X(:,1);
for n = 1:frames-1
    Z(:,n+1) = A*Z(:,n); 
end

%% Construct "low-rank" (background) and "sparse" (foreground) DMD frames
% Solve the linear system given by the eigenbasis expansion
b = eVectors \ X(:,1);
dt = 1;
% Compute the continuous-time DMD eigenvalues
omega = log(mu)/dt;

% Find the low rank modes by putting a 1 in the corresponding entry of 
% modes_low for those modes that are lower than a given threshold
modes_low = zeros(i,1);
for j = 1:i
    if abs(omega(j)) <= 0.01
        modes_low(j) = 1;
    end 
end 

% Construct the x matrix, which is a simple sum of the modes 
x = zeros(i,frames);
for t = 1:frames
    for k = 1:i
        x(:,t) = x(:,t) + b(k) * eVectors(:, k) * mu(k)^((t-1)/dt);
    end 
end

% Construct the X_lowrank DMD matrix for the background modes
x_low = zeros(i,frames);
for t = 1:frames
    for k = 1:i
        if modes_low(k) == 1
            x_low(:,t) = x_low(:,t) + b(k) * eVectors(:, k) * mu(k)^((t-1)/dt);
        end
    end 
end

% Construct the x_sparse DMD matrix for the foreground modes
x_sparse = x - abs(x_low);

% Put back the negative entries from x_sparse into x_low
negatives = x_sparse;
for t = 1:frames
    for j = 1:i
        if real(negatives(j,t)) > 0
            negatives(j,t) = 0;
        end
    end
end
% Nathan Kutz uses this step to make the problem mathematically rigorous,
% but somehow the results seem better without it.
% x_low = negatives + abs(x_low);

% Then subtract them from x_sparse
x_sparse = x_sparse - negatives;

%% Plot of Original Video and DMD video

subplot(1,2,1)
toPrint = reshape(gFrames,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,24),[0,255]))
title('Original Video')

original = toPrint;

subplot(1,2,2)
x_real = real(x);
snapshot = U(:,1:i) * x_real;
toPrint = reshape(snapshot,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,24),[0,255]))
title('DMD Reconstruction')

reconstructed = toPrint;
norm(original(:,:,24)-toPrint(:,:,24))

%% Plot of Low-Rank Video 
subplot(3,2,3)
snapshot = U(:,1:i) * gFrames_lowrank;
toPrint = reshape(snapshot,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,24),[0,255]))
title('Low Rank Video')

%% Plot of foreground and background
subplot(1,2,1)
x_low_real = real(x_low);
snapshot1 = U(:,1:i) * x_low_real;
toPrint = reshape(snapshot1,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,24),[0,255]))
title('DMD Background')

subplot(1,2,2)
x_sparse_real = real(x_sparse);
snapshot2 = U(:,1:i) * (x_sparse_real);
[min,max] = bounds(snapshot2, 'all');
toPrint = reshape(snapshot2,[height,width,frames]);
imshow(mat2gray(toPrint(:,:,24),[min/2,max/2]))
title('DMD Foreground')

%% Make a movie with no background from the DMD video matrix frames:

movie = U(:,1:i) * x_sparse_real;
toPrint = reshape(movie,[height,width,frames]);
[min,max] = bounds(movie, 'all');
figure(2)
f = 1;
while f <= frames
    imshow(mat2gray(toPrint(:,:,f),[min/2,max/2]))
    pause(1/120);
    f = f + 1;
end

