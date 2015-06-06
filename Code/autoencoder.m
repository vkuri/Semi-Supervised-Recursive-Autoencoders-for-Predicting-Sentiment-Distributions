function [f,g] = autoencoder(init, ei, parameters, datacell, output, vocabulary)
   
    t = length(datacell);
    f = 0;
    g = zeros(size(init));
    init = params2stack(init, ei);
    update_flag = 0;
    derivs = initStack(ei);
    nodes_total = 0;
    dim = ei.dimensionality;
    init2 = init;
    init2.W = vocabulary + init.W;

    existing_tree = cell(t, 1);
    % Calculating Cost and Gradient for all nodes
    lambda = ei.alpha * parameters.regularization;
    for i = 1:t
        existing_tree{i} = struct;
        if mod(i,1000) == 0
            i;
        end
        vocabIndices = datacell{i};
        input = init.W(vocabIndices,:);
        input = input + vocabulary(vocabIndices,:);

        ei.depth = length(vocabIndices);
        [cost, grads, tree] = calculate(parameters.norm_func, parameters.norm_func_prime, init, ei, parameters, input, vocabIndices, output(i), update_flag, existing_tree{i});
        existing_tree{i}.tree = tree;
        f = f + cost;
        derivs.b1 = derivs.b1 + grads.b1;
        derivs.b2 = derivs.b2 + grads.b2;
        derivs.W1 = derivs.W1 + grads.W1;
        derivs.W2 = derivs.W2 + grads.W2;
        derivs.W = derivs.W + grads.W;
        nodes_total = nodes_total + ei.depth - 1;
    end

    f = 1/nodes_total*f + lambda(1)/2 * (init.W1(:)' * init.W1(:) + init.W2(:)' * init.W2(:));
    g = 1/nodes_total*[derivs.W1(:);derivs.W2(:);derivs.b1(:);derivs.b2(:)] + [lambda(1)*init.W1(:);lambda(1)*init.W2(:);zeros(3*dim,1)];


    f = f + lambda(2)/2 * init.W(:)' * init.W(:);
    g = [g; 1/nodes_total*derivs.W(:) + lambda(2)*init.W(:)];


    % Calculating the Cost and Gradient for leaf nodes
    update_flag = 1;
    lambda = (1 - ei.alpha) * parameters.regularization;
    Wsize = length(init.W(:));
    Bsize = ei.outputsize;
    Wlsize = ei.outputsize * dim;
    temp = g(end-Wsize+1:end);
    g(end-Wsize+1:end) = 0;
    g = [g; zeros(Bsize+Wlsize,1)];
    g(end-Wsize+1:end) = temp;
    nodes_total = 0;
    init = init2;

    for i = 1:t
        if mod(i,1000) == 0
            i;
        end
        vocabIndices = datacell{i};
        input = init.W(vocabIndices,:);
        input = input + vocabulary(vocabIndices,:);

        ei.depth = length(vocabIndices);
        [cost, grads, ~] = calculate(parameters.norm_func, parameters.norm_func_prime, init, ei, parameters, input, vocabIndices, output(i), update_flag, existing_tree{i}.tree);
        f = f + cost;
        derivs.b1 = derivs.b1 + grads.b1;
        derivs.b2 = derivs.b2 + grads.b2;
        derivs.W1 = derivs.W1 + grads.W1;
        derivs.W2 = derivs.W2 + grads.W2;
        derivs.W = derivs.W + grads.W;
        nodes_total = nodes_total + 1;
    end

    f = 1/nodes_total*f + lambda(1)/2 * (init.W1(:)' * init.W1(:) + init.W2(:)' * init.W2(:));
    g = 1/nodes_total*[derivs.W1(:);derivs.W2(:);derivs.b1(:);derivs.b2(:)] + [lambda(1)*init.W1(:);lambda(1)*init.W2(:);zeros(3*dim,1)];

    derivs.Wl = 1/nodes_total*[derivs.Wl(:);derivs.bl] + [lambda(4)*init.Wl(:);zeros(1,1)];
    f = f + lambda(4)/2 * init.Wl(:)' * init.Wl(:);
    g = [g(:); derivs.Wl(:)];

    f = f + lambda(2)/2 * init.W(:)' * init.W(:);
    g = [g; 1/nodes_total*derivs.W(:) + lambda(2)*init.W(:)];
end

