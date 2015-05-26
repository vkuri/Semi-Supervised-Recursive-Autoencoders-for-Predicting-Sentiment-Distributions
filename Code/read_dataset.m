function [ output_args ] = read_dataset(params_dataset, parameters)
% Function for reading the dataset

% Making use of vocabulary list from Socher's paper
load(strcat(params_dataset,'vocab.mat'), 'words');

% Randomly initialize each word in the vocabulary list with a weight
distribution_inteval = 0.1;
word_weights = (rand([parameters.word_size size(words,2)]) * 2 * distribution_inteval) - distribution_inteval;

end