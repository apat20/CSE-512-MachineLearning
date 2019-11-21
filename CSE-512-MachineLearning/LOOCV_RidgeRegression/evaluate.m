%% Reading the data both train and validation data
clear all;
clc;

valData = csvread('valData.csv',0,1);
valLabels = csvread('valLabels.csv',0,1);

% trainData_new = zeros(5000,30001);
% trainLabels_new = zeros(5000,1);

trainData = csvread('trainData.csv',0,1);
trainLabels = csvread('trainLabels.csv',0,1);

% for i = 1:5000
%     trainData_new(i,:) = trainData(i,:).^2;
%     trainLabels_new(i) = trainLabels(i)^2;
% end

% Enter the value of lambda
lambda = 0.09;
x_train = [trainData;valData]';

y_train = [trainLabels;valLabels];

%% Learning the model on the training data
[m,n] = size(x_train);
X_bar_train = [x_train;(ones(n,1))'];

I = eye(3000);
Z = zeros(3000,1);
I_bar = [I,Z;Z',0];

C = X_bar_train*X_bar_train' + lambda*I_bar;
d = X_bar_train*y_train;

weight_vector = mldivide(C,d);
w = weight_vector(1:3000,1);
bias = weight_vector(3001,1);

% for i = 1:n
%     if w(i) >= 5
%         disp(i);
%         disp(w(i));
%     end
% end

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

%% Reading the testing data:

testData = csvread('testData_new.csv',0,1);

[a,b] = size(testData);
x_test = [testData';(ones(a,1))'];
w = weight_vector(1:3000,1);
bias = weight_vector(3001,1);

y_pred = weight_vector'*x_test;
csvwrite('y_pred_new_1.csv',y_pred')
%%
complete_data = csvread('testData_new.csv');
Index = complete_data(:,1);
csvwrite('Index.csv', Index);

%% 
lambda = [0.01,0.1,1,10,100,1000];
Training_RMSE = [1.1205,1.2238,1.5780,2.1900,2.9709,3.3316];
Validation_RMSE = [2.5792,2.1575,1.9968,2.3477,3.0171,3.3454];
LOOCV_RMSE = [1.2552,1.3037,1.5962,2.1913,2.9710,3.3316];

num_batches = size(lambda,2);
x1 = [1:num_batches];
plot(x1,Training_RMSE,'b','LineWidth',2);
hold on;
plot(x1,Validation_RMSE,'--b','LineWidth',2);
hold on;
plot(x1,LOOCV_RMSE,'--*r','LineWidth',2);
set(gca,'XTickLabel',lambda);
hLegend = legend('Training','Validation','LOOCV', 'Location','NorthEast');
X_label = xlabel('Values of Lambda','FontSize',25);
Y_label = ylabel('RMSE values','Interpreter','Latex','FontSize',25);
grid on;

export_fig Lambda_vs_RMSE.pdf -transparent




