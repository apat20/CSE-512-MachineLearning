clear all;
clc
% Loading the data 
load q2_1_data.mat
[m,n] = size(trD);

%%  TRAINING
% Converting the training data to the desired dimensions
train_labels = trLb;
train_data = trD';
%C = 0.1;
C = 10;

% Solving the dual problem and get the value of the multipliers 'alpha'
[alpha,obj_val] = solveDual(train_data, train_labels, C, n); 

% Computing the 'weights' and the 'bias' of the primal problem from the dual
% variable 'alpha'
[w, b] = computeWeightBias1(alpha, train_data, train_labels, C, n);

%% VALIDATION 
% Converting the validation data to the desired dimensions
[c,d] = size(valD);
val_labels = valLb;
val_data = valD';

% Computing the Kernel for the validation data
K_val = val_data*val_data';

for i = 1:d
    val_pred(i,1) = sign(w*val_data(i,:)' + b);
end

accuracy = sum(val_labels == val_pred) / numel(val_labels);
accuracyPercentage = 100*accuracy;

% figure;
cm = confusionmat(val_labels, val_pred);
