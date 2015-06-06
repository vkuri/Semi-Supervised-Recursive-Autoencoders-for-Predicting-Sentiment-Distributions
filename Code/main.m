%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions
% Implementation based on Socher's paper
% http://www.socher.org/index.php/Main/Semi-SupervisedRecursiveAutoencodersForPredictingSentimentDistributions
% UCSD Neural Networks project CSE 291(aka CSE 253)
% Authors : Vincent Kuri, Sanjeev Shenoy, Madhavi Yenugala
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear all variables and start training afresh
clearvars;clc;clear;

% Make sure you add ./tools folder to your path because it contains important functions used in the code
% Using genpath to include all subfolders
addpath(genpath('./tools'));

%%%%%%%%%%%%%%%%%%%%%%%%
% Setting parameters for training
%%%%%%%%%%%%%%%%%%%%%%%%
% Empty parameter structure
parameters = struct;

% Dimensionality of the word size, change according to the word size required
parameters.word_size = 50;

% Regularization parameters, adapted from the paper
% Regularization for W(1), W(2), W(Label), L
parameters.regularization = [1e-05, 1e-04, 1e07, 1e-02];

% Weightage for errors
parameters.error_weightage = 0.2;

% Distribution interval
parameters.distribution_interval = 0.05;

% Weightage of cost for nonterminals
parameters.beta = 0.5;

% Normalization function
parameters.norm_func = @norm1tanh;

% Normalization function prime
parameters.norm_func_prime = @norm1tanh_prime;

%%%%%%%%%%%%%%%%%%%%%%%%
% Minfunc parameters
%%%%%%%%%%%%%%%%%%%%%%%%
% Empty options structure
options = struct;

% Maximum number of iterations
options.maxIter = 70;

% Display mode
options.display = 'iter';

% Optimization method, can try different methods for minfunc as well
options.Method = 'lbfgs';

%%%%%%%%%%%%%%%%%%%%%%%%
% Read the dataset
%%%%%%%%%%%%%%%%%%%%%%%%
params_dataset = struct;

% Setting the path for the dataset
params_dataset.path = '../Dataset/';
%params_dataset.path = 'Semi-Supervised-Recursive-Autoencoders-for-Predicting-Sentiment-Distributions/Dataset/';

% Setting filenames for the positive and negative datasets
params_dataset.filename_positive = 'rt-polarity.pos';
params_dataset.filename_negative = 'rt-polarity.neg';

% Setting the paths for the binarized positive and negative reviews
params_dataset.pos_binarized = 'rt-polarity_pos_binarized.mat';
params_dataset.neg_binarized = 'rt-polarity_neg_binarized.mat';
params_dataset.cv_obj = 'cv_obj.mat';

% Specify the fold
params_dataset.kfold = 1;

% Preprofile settings
% params_dataset.filename_preprofile = 'preprofile_our.mat';
params_dataset.filename_preprofile = 'preprofile.mat';

% Load from preprofile if it exists
preprofile_path = strcat(params_dataset.path, params_dataset.filename_preprofile);
if exist(preprofile_path, 'file') == 2
    % load(preprofile_path, 'labels','train_ind','test_ind', 'cv_ind', 'ww', 'dictNum', 'test_nums');
    load(preprofile_path, 'labels','train_ind','test_ind', 'cv_ind', 'We2', 'allSNum', 'test_nums');
else
    read_dataset(params_dataset, parameters);
    % load(preprofile_path, 'labels','train_ind','test_ind', 'cv_ind', 'ww', 'dictNum', 'test_nums');
    load(preprofile_path, 'labels','train_ind','test_ind', 'cv_ind', 'We2', 'allSNum', 'test_nums');
end


% Comment if we are using our own preprofile.mat
ww = We2;
dictNum = allSNum;
% End Comment

dictLength = length(ww);
num_training_ex = length(dictNum(train_ind));

ei = struct;
ei.dimensionality = parameters.word_size;
ei.outputsize = 1;
ei.alpha = parameters.error_weightage;
ei.lambda = parameters.regularization(4);
ei.vocab = length(ww);
vocabulary = ww';
datacell = dictNum;
output = labels;

dim = ei.dimensionality;
out = ei.outputsize;

r  = sqrt(6) / sqrt(2*dim+1);
params.W1 = rand(dim,2*dim)*2*r-r;
params.b1 = rand(dim,1)*2*r-r;
params.W2 = rand(2*dim,dim)*2*r-r;
params.b2 = rand(2*dim,1)*2*r-r;
params.Wl = rand(out,dim)*2*r-r;
params.bl = rand(out,1)*2*r-r;
params.W = 1e-3*(rand(length(vocabulary),dim)*2*r-r);

init = stack2params(params);

ei.paramslength = size(init,1);
r = randperm(length(test_ind));
test_ind = test_ind(r);
r = randperm(length(train_ind));
train_ind = train_ind(r);

training_data = datacell(train_ind);
testing_data = datacell(test_ind);
labels_test = output(test_ind);
labels_train = output(train_ind);

l = length(training_data);
% Deleting sentences that are less than 2 words
for i=1:length(training_data)
    if i>l
        break;
    end
    temp = training_data{i};
    if length(temp) < 2
        training_data(i) = [];
        labels_train(i) = [];
        l = l-1;
    end
end

l = length(testing_data);
for i=1:length(testing_data)
    if i>l
        break;
    end
    temp = testing_data{i};
    if length(temp) < 2
        testing_data(i) = [];
        labels_test(i) = [];
        l = l-1;
    end
end



options = struct;
options.maxIter = 70;
options.Method = 'lbfgs';
options.display = 'iter';

datacell = training_data;
labels = labels_train;


[opt_params,opt_value,exitflag,out1] = minFunc(@autoencoder,init, options, ei, parameters, datacell, labels, vocabulary);

just_pred = 1;

datacell = testing_data;
labels = labels_test;
[~, ~, pred] = autoencoder(opt_params, ei, datacell, labels, vocabulary, just_pred);
pred = 1*(pred>0.5);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

datacell = training_data;
labels = labels_train;
[~, ~, pred] = autoencoder(opt_params, ei, datacell, labels, vocabulary, just_pred);
pred = 1*(pred>0.5);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);






