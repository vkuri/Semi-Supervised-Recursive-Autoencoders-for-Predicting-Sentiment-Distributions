function [f,g] = autoencoder(datacell, vocabulary, output, ei, init)
%AUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

% ei has these fields
% dimensionality = no. of dimensions the word has
% outputsize = no. of classes
% depth = no. of words in the sentence
% alpha - alpha value
% lambda - lambda value

% input and output --
% input - txnxd vector
% output - txo vector - o = outputsize
% issues : i don't understand the norm1tanh_prime function. it's giving weird answer

    dim = ei.dimensionality;
    out = ei.outputsize;

    params.W1 = rand(dim,2*dim);
    params.b1 = rand(dim,1);
    params.W2 = rand(2*dim,dim);
    params.b2 = rand(2*dim,1);
    params.Wl = rand(out,dim);
    params.bl = rand(out,1);

    f = 0;
    g = params2stack(stack2params(params)*0);
    init = g;
    for i = 1:t
        vocabIndices = datacell{i};
        input = vocabulary(vocabIndices, :);
        %this should ideally be autoencoder(@norm1tanh, @norm1tanh_prime, init, ei, input(i,:), out(i,:));
        [f1 g1] = autoencoder(@norm1tanh, @norm1tanh_prime, init, ei, input(i,:), output(i,:), vocabIndices);
        f = f + f1;
        g = g + g1;
    end
    
    f = f +  0.5 * ei.lambda * norm(init)^2;
    g = g + ei.lambda*g;
end

