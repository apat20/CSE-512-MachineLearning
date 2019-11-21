%load trainAnno.mat
[train_data, train_labels, val_data, val_labels, trainRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();

% 'm' denotes the number of data points in the data.
train_data = train_data';
[m,n] = size(train_data);
C = 10;
alpha = solveDual(train_data, train_labels, C, m);
[weights,bias] = computeWeightBias1(alpha, train_data, train_labels, C, m);

% output_file = '/hw4data/savefile.mat';
output_file = 'question_3_4_1';
data_set = 'val';

% transposing the weights to meet the desired dimensions in helper
% functions
weights_new = weights';

% Generate the output file.
HW4_Utils.genRsltFile(weights_new, bias, data_set, output_file);

% Generate AP and Precision-Recall Curve.
[AP, Prec, Rec] = HW4_Utils.cmpAP('question_3_4_1.mat', data_set);
