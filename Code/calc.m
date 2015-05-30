function [f,g] = calc(fun, fun_prime, params, ei, input, output)
    f = 0;
%     tree structure:
%     if a sentence has n words and each word is of dimension dim=d
%     the tree constructed is of depth d
%     there are 2 nodes at every level except the root node
%     every level has children c1, c2 and parent p
%     lowest level - level 1 - doesn't have c1 and c2
%     highest level - doesn't have parent
%     size of W1 : 2dxd
%     size of W2 : dx2d
%     size of input: nxd

    depth = ei.depth;
    %depth = inputsize
    
    dim = ei.dimensionality;
    alpha = ei.alpha;
    
    tree = cell(depth, 1);
   
    narray = ones(depth,1);

    mine = 10000000000000000;
    
%     in each iteration d: the parent is constructed. we needn't construct the 
%     parent for level d=depth, so the for loop runs only till depth-1
    for d = 1:depth-1
        tree{d} = struct;
%         this loop runs till when input size is > 1. that is till there is one
%         node
        for i = 1:size(input,1)-1
            act = params.W1*[input(i,:) input(i+1,:)]' + params.b1;     %size: dx1
            p = fun(act);                                               %size: dx1
            rec = params.W2*p + params.b2;                              %size: 2dx1   
            c1c2d = fun(rec);                                           %size: 2dx1
            n = narray(i,1)/(narray(i,1) + narray(i+1,1));
            
            e_rec = n * (norm(input(i,:)' - c1c2d(1:dim,:)))^2 + (1-n)*(norm(input(i+1,:)' - c1c2d(dim+1:2*dim,:)))^2;
            
            if e_rec < mine
                tree{d}.c1 = input(i,:)';
                tree{d}.c2 = input(i+1, :)';
                tree{d}.c1c2d = c1c2d;
                tree{d}.n1 = narray(i);
                tree{d}.n2 = narray(i+1);
                tree{d}.p = p;
                tree{d}.rec = rec;
                tree{d}.act = act;
                mine = e_rec;
                mini = i;
            end
        end
        f = f + e_rec;
%       tree{d} has the correct children and parent by this stage

%       This is the classification error           %Wl size: oxd
        g = params.Wl*tree{d}.p + params.bl;       %size: ox1
        t = sigmoid(g);         
        tree{d}.eta = (1-alpha)*(t - output);      %output size: ox1
        
        e_cl = - dot(output, log(t));
        f = f + e_cl;
        
        n = tree{d}.n1/(tree{d}.n1+tree{d}.n2);
        
        tree{d}.gam = alpha*[n*(tree{d}.c1 - tree{d}.c1c2d(1:dim, :)) ; (1-n)*(tree{d}.c2 - tree{d}.c1c2d(dim+1:2*dim, :))].*fun_prime(tree{d}.rec); %size: 2dx1
 
        narray(mini,:) = tree{d}.n1 + tree{d}.n2 + 1;
        narray(mini+1,:) = [];
        input(mini, :) = tree{d}.p;
        input(mini+1, :) = [];
    end
    
 
    tree{depth}.c1 = tree{depth-1}.p;
    tree{depth}.c2 = zeros(dim, 1);
    tree{depth}.del2 = zeros(dim, 1);
    
    act = tree{depth-1}.act;                            %size dx1
%     TODO: this equation needs to be checked
    tree{depth}.del1 = fun_prime(act) .* (params.W2'*tree{d}.gam + params.Wl'*tree{d}.eta);  %size dx1
    
    W1l = params.W1(:, 1:dim);
    W1r = params.W1(:, dim+1:2*dim);
    
    for d = depth-1:-1:1
%         determine if current children are left or right child of the previous layer
        if isequal(tree{d}.p, tree{d+1}.c1)
            V = W1l;
            dp = tree{d+1}.del1;
        else
            V = W1r;
            dp = tree{d+1}.del2;
        end
        
%         see if the nodes are leaf nodes or not.
        if tree{d}.n1 == 1
            tree{d}.del1 = V'*dp;
        else
            act = tree{d-1}.act;%params.W1l*tree{d}.c1 + params.b1;
            tree{d}.del1 = fun_prime(act) .* (V*dp + [params.W2]'*tree{d}.gam + params.Wl'*tree{d}.eta);
        end
        
        if tree{d}.n2 == 1
            tree{d}.del2 = V'*dp;
        else
            act = tree{d-1}.act;% params.W1r*tree{depth}.c1 + params.b1;
            tree{d}.del2 = fun_prime(act) .* (V*dp + [params.W2]'*tree{d}.gam + params.Wl'*tree{d}.eta);
        end
    end
    
    derivs = zeros(size(stack2params(params),1));
    derivs = params2stack(derivs, ei);
    
    for d = 1:depth
        derivs.W1 = derivs.W1 + [tree{d}.del1 * tree{d}.c1'; tree{d}.del2 *tree{d}.c2']'; 
        derivs.b1 = derivs.b1 + tree{d}.del1 + tree{d}.del2;
        %TODO: need to check this
        if d ~= depth
            derivs.W2 = derivs.W2 + tree{d}.gam*tree{d}.p';
            derivs.b2 = derivs.b2 + tree{d}.gam;
            derivs.Wl = derivs.Wl + tree{d}.eta * tree{d}.p';
            derivs.bl = derivs.bl + tree{d}.eta;
        end
    end
    g = stack2params(derivs);
end
    