function [f,g, pred] = calc(fun, fun_prime, params, ei, input, output, vocabIndices, just_pred)
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
%     input - size nxdim
%     stack.W - size vxdim
%     W - nxdim
    inputCopy = input;
    W = params.W(vocabIndices,:);
    % input = input.*W;
    depth = ei.depth;
    %depth = inputsize
    
    dim = ei.dimensionality;
    alpha = ei.alpha;
    
    tree = cell(2*depth-1, 1);
    
    for i = 1:depth
        tree{i}.node = input(i,:)';
        tree{i}.numnodes = 1;
        tree{i}.lc = -1;
        tree{i}.rc = -1;
    end
   
    narray = ones(depth,1);

    indices = [1:depth];
   
    
%     in each iteration d: the parent is constructed. we needn't construct the 
%     parent for level d=depth, so the for loop runs only till depth-1
    for d = 1:depth-1
        mine = 1e50;
        tree{depth+d} = struct;
%         this loop runs till when input size is > 1. that is till there is one
%         node
        flag1 = 0;
        for i = 1:size(input,1)-1
            act = params.W1*[input(i,:) input(i+1,:)]' + params.b1;     %size: dx1
            p = fun(act);                                               %size: dx1
            p_norm = bsxfun(@rdivide,p,sqrt(sum(p.^2,1)));
            %p = p/norm(p);
            rec = params.W2*p_norm + params.b2;                              %size: 2dx1   
            c1c2d = fun(rec);                                           %size: 2dx1
            c1n = bsxfun(@rdivide,c1c2d(1:dim,:),norm(c1c2d(1:dim,:)));
            c2n = bsxfun(@rdivide,c1c2d(dim+1:2*dim,:),norm(c1c2d(dim+1:2*dim,:)));

            y1 = c1c2d(1:dim,:);
            y2 = c1c2d(dim+1:2*dim,:);
            y1c1 = alpha*(y1-input(i,:)');
            y2c2 = alpha*(y2-input(i+1,:)');
            
            e_rec = 1/2*sum((y1c1.*(y1-input(i,:)')) + (y2c2.*(y2-input(i+1,:)')));

            n = narray(i,1)/(narray(i,1) + narray(i+1,1));  
            % e_rec = 1*(n * (norm(input(i,:)' - c1c2d(1:dim,:)))^2 + (1-n)*(norm(input(i+1,:)' - c1c2d(dim+1:2*dim,:)))^2);

            if e_rec < mine
                flag1 = 1;
                tree{depth+d}.c1 = input(i,:)';
                tree{depth+d}.c2 = input(i+1, :)';
                
                tree{depth+d}.c1c2dn = [c1n; c2n];
                tree{depth+d}.c1c2d = c1c2d;
                tree{depth+d}.y1 = y1;
                tree{depth+d}.y2 = y2;
                tree{depth+d}.y1c1 = y1c1;
                tree{depth+d}.y2c2 = y2c2;
                tree{depth+d}.e_rec = e_rec;
                tree{depth+d}.p_norm = p_norm;

                tree{depth+d}.n1 = narray(i);
                tree{depth+d}.n2 = narray(i+1);

                tree{depth+d}.node = p;
                tree{depth+d}.rec = rec;
                tree{depth+d}.delc1 = alpha*n*(tree{depth+d}.c1 - c1c2d(1:dim,:));
                tree{depth+d}.delc2 = alpha*(1-n)*(tree{depth+d}.c2 - c1c2d(dim+1:2*dim,:));
                tree{depth+d}.delc1 = 0;
                tree{depth+d}.delc2 = 0;
                tree{depth+d}.act = act;
                mini = i;
                mine = e_rec;
                tree{depth+d}.lc = indices(i);
                tree{depth+d}.rc = indices(i+1);
                tree{indices(i)}.par = depth+d;
                tree{indices(i+1)}.par = depth+d;
            end
        end
      %  fprintf('mine: %f %d\n', mine,d);
        if flag1 == 0
            flag1
        end
        if mine == 1e50
            fprintf('mine: %f %d\n', mine,d);
        end
      
        f = f + alpha * mine;
      
%       This is the classification error           %Wl size: oxd
        g = params.Wl*tree{depth +d}.node + params.bl;       %size: ox1
        prediction = sigmoid(g);
        
        e_cl = 0.5*(prediction-output)^2;      
        tree{depth+d}.eta = (1-alpha)*(prediction - output)*prediction*(1-prediction);
  
%         tree{depth+d}.eta = (1-alpha)*(prediction - output);      %output size: ox1
%         e_cl = -dot(output, log(prediction));
%         
        f = f + (1 - alpha) * e_cl;
        
        %sum(tree{depth+d}.c1c2dn - tree{depth+d}.c1c2d)
        n = tree{depth+d}.n1/(tree{depth+d}.n1+tree{depth+d}.n2);
        
        rec1 = tree{depth+d}.c1c2d(1:dim, :);
        rec2 = tree{depth+d}.c1c2d(dim+1:2*dim, :);
        % gam1 = fun_prime(rec1)*(tree{depth+d}.c1 - tree{depth+d}.c1c2d(1:dim, :));
        % gam2 = fun_prime(rec2)*(tree{depth+d}.c2 - tree{depth+d}.c1c2d(dim+1:2*dim, :));
        gam1 = fun_prime(rec1)*tree{depth+d}.y1c1;
        gam2 = fun_prime(rec2)*tree{depth+d}.y2c2;

        tree{depth+d}.gam1 = gam1;
        tree{depth+d}.gam2 = gam2;
        
        tree{depth+d}.gam = -2*alpha*[n*gam1 ; (1-n)*gam2]; %size: 2dx1
        %tree{depth+d}.gam = -2*alpha*fun_prime(tree{depth+d}.rec)*[n*(tree{depth+d}.c1 - tree{depth+d}.c1c2d(1:dim, :)) ; (1-n)*(tree{depth+d}.c2 - tree{depth+d}.c1c2d(dim+1:2*dim, :))]; %size: 2dx1
 
        narray(mini,:) = tree{depth+d}.n1 + tree{depth+d}.n2;
        narray(mini+1,:) = [];
     
        tree{depth+d}.numnodes = narray(mini,:);
        
        input(mini, :) = tree{depth+d}.p_norm;
        input(mini+1, :) = [];
        indices(mini) = depth+d;
        indices(mini+1) = [];
    end

   % for i = depth+1:2*depth-1
   %      fprintf('%d %d %d %d\n', i, tree{i}.lc, tree{i}.rc, depth);
   % end
    
    if just_pred
        g = params.Wl*tree{2*depth-1}.node + params.bl;       %size: ox1
        t = sigmoid(g);  
        pred = t;
        f = -1;
        g = [];
        return;
    end
    
   
    dd = 2*depth-1;
    node = tree{2*depth-1}.node;                            %size dx1
    tree{dd}.del = fun_prime(node) * (params.W2'*tree{dd}.gam + params.Wl'*tree{dd}.eta);  %size dx1
    
    W1l = params.W1(:, 1:dim);
    W1r = params.W1(:, dim+1:2*dim);
    
    for d = 2*depth-2:-1:1
%         determine if current children are left or right child of the previous layer
        parent = tree{d}.par;
        if tree{parent}.lc == d
            V = W1l;
            delc = tree{parent}.delc1;
        else
            V = W1r;
            delc = tree{parent}.delc2;
        end
        
        dp = tree{parent}.del;
        
%         see if the nodes are leaf nodes or not.
        if tree{d}.numnodes == 1
            tree{d}.del = V'*dp + delc;
        else
            node = tree{d}.node;
            W21 = params.W2(:,1:dim);
            W22 = params.W2(:,dim+1,2*dim);
            tree{d}.del = fun_prime(node) * (V'*dp + W21*tree{d}.gam1 + W22*tree{d}.gam2);
        end
    end
    
    derivs = initStack(ei);
    for d = depth+1:dd    
        derivs.W1 = derivs.W1 + tree{d}.del * tree{d}.p_norm';
        derivs.b1 = derivs.b1 + tree{d}.del;
        derivs.W2 = derivs.W2 + tree{d}.gam*tree{d}.node';
        derivs.b2 = derivs.b2 + [tree{d}.gam1*tree{d}.p_norm' tree{d}.gam2*tree{d}.p_norm'];
        derivs.Wl = derivs.Wl + tree{d}.eta * tree{d}.node';
        derivs.bl = derivs.bl + tree{d}.eta;
    end
    
    for d = 1:depth
        derivs.W(vocabIndices(d),:) = derivs.W(vocabIndices(d),:) + tree{d}.del';
    end
    g = stack2params(derivs);
    pred = [];
end
    