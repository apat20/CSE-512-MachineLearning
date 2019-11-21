% % [heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');
% % model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');
% % [predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model);
% 
scales = [8, 16, 32, 64];
normH = 16;
normW = 16;
bowCs = HW5_BoW.learnDictionary(scales, normH, normW);

%%

[trIds, trLbs] = ml_load('bigbangtheory_v3/train.mat',  'imIds', 'lbs');             
tstIds = ml_load('bigbangtheory_v3/test.mat', 'imIds');

load('boWCs.mat');

trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs');
tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs');

%%

load('features.mat');

%% Tuning RBF Kernel

% model = svmtrain(trLbs, trD', '-s 0 -v 5');

% model = svmtrain(trLbs, trD');
model = svmtrain(trLbs, trD', '-s 0 -c 100 -g 30');
[predClass, ~, ~] = svmpredict(ones(size(tstD, 2), 1), tstD', model);

%% Using Exponential Chi-Square kernel

% [trainK, testK] = cmpExpX2Kernel(trD', tstD', 2);
[trainK, testK] = cmpExpX2Kernel(trD', tstD', 0.7);

%% Tuning the Exponential Chi-Square kernel

model = svmtrain(trLbs, [(1:size(trainK, 1))', trainK], '-t 4 -c 64 -g 3');
[predClass, ~, ~] = svmpredict(ones(size(testK, 1), 1), [(1:size(testK, 1))', testK], model);
% model = svmtrain(trLbs, [(1:size(trainK, 1))', trainK], '-s 0 -t 4 -c 64 -v 5');
%%
csvwrite("tstIds.csv", tstIds');
csvwrite("submission_7.csv", predClass);
             


