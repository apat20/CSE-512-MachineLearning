%% Reading the data both train and validation data
clear all;
clc;

valData = csvread('valData.csv',0,1);
valLabels = csvread('valLabels.csv',0,1);

trainData = csvread('trainData.csv',0,1);
trainLabels = csvread('trainLabels.csv',0,1);

% Enter the value of lambda
% lambda = 0.01;
x_train = trainData';
x_val = valData';

y_train = trainLabels;
y_val = valLabels;
%% Learning the model on the training data
[m,n] = size(x_train);
X_bar_train = [x_train;(ones(5000,1))'];
X_bar_val = [x_val;(ones(5000,1))'];

I = eye(3000);
Z = zeros(3000,1);
I_bar = [I,Z;Z',0];

C = X_bar_train*X_bar_train' + lambda*I_bar;
d = X_bar_train*y_train;

weight_vector = mldivide(C,d);
w = weight_vector(1:3000,1);
bias = weight_vector(3001,1);

%% Computing the training error on the training data
% cvErrs = zeros(5000,1);
diff_train = (zeros(5000,1));

for i = 1:n
    disp(i);
    diff_train(i) = weight_vector'*X_bar_train(:,i) - y_train(i);
%     cvErrs(i) = weight_vector'*X_bar(:,i) - y(i)/1 - ((X_bar(:,i))'*mldivide(C,X_bar(:,i)));
end

sumErrors_train = sum(diff_train.^2);
RMSE_train = sqrt(sumErrors_train/n);
%% Computing the validation error on the validation data

diff_val = (zeros(5000,1));

for i = 1:n
    disp(i);
    diff_val(i) = weight_vector'*X_bar_val(:,i) - y_val(i);
%     cvErrs(i) = weight_vector'*X_bar(:,i) - y(i)/1 - ((X_bar(:,i))'*mldivide(C,X_bar(:,i)));
end

sumErrors_val = sum(diff_val.^2);
RMSE_val = sqrt(sumErrors_val/n);

%% Printing the output
fprintf('RMSE_train:')
disp(RMSE_train);

fprintf('RMSE_val:')
disp(RMSE_val);


%% Computing the LOOCV errors on the training data set:

lambda = 0.01;

[w_train,b_train,obj_train,cvErrs_train] = ridgeReg(x_train,y_train,lambda);

%%
[m,n] = size(cvErrs_train);
sum_cvErrors = sum(cvErrs_train.^2);
RMSE_cvErrs = sqrt(sum_cvErrors/m);

fprintf('RMSE_cvErrs:')
disp(RMSE_cvErrs);





