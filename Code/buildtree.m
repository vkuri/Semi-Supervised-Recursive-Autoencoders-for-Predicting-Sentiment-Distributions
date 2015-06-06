function tree = buildtree(fun, fun_prime, params, ei, parameters, input, label, update_flag, existing_tree)
    depth = ei.depth;
    dim = ei.dimensionality;
    alpha = ei.alpha;
    narray = ones(depth,1);
    indices = [1:depth];
    tree = cell(2*depth-1, 1);
    beta = parameters.beta;

    for i = 1:depth
        tree{i}.node = input(i,:)';
        tree{i}.numnodes = 1;
        tree{i}.lc = -1;
        tree{i}.rc = -1;
        tree{i}.par = -1;
    end

	if update_flag == 0
		for d = 1:depth-1
        	mine = 1e50;
        	tree{depth+d} = struct;

			for i = 1:size(input,1)-1
	            act = params.W1*[input(i,:) input(i+1,:)]' + params.b1;     %size: dx1
	            p = fun(act);                                               %size: dx1
	            p_norm = p./norm(p);
	            rec = params.W2*p_norm + params.b2;                              %size: 2dx1   
	            c1c2d = fun(rec);                                           %size: 2dx1
	            y1 = bsxfun(@rdivide,c1c2d(1:dim,:),norm(c1c2d(1:dim,:)));
	            y2 = bsxfun(@rdivide,c1c2d(dim+1:2*dim,:),norm(c1c2d(dim+1:2*dim,:)));

	            y1c1 = alpha*(y1-input(i,:)');
	            y2c2 = alpha*(y2-input(i+1,:)');
	            
	            e_rec = 1/2*sum((y1c1.*(y1-input(i,:)')) + (y2c2.*(y2-input(i+1,:)')));

	            n = narray(i,1)/(narray(i,1) + narray(i+1,1));

	            if e_rec < mine
	                tree{depth+d}.c1 = input(i,:)';
	                tree{depth+d}.c2 = input(i+1, :)';
	                tree{depth+d}.c1c2dn = [y1; y2];
	                tree{depth+d}.c1c2d = c1c2d;
	                tree{depth+d}.y1c1 = y1c1;
	                tree{depth+d}.y2c2 = y2c2;
	                tree{depth+d}.e_rec = e_rec;
	                tree{depth+d}.p_norm = p_norm;
	                tree{depth+d}.n1 = narray(i);
	                tree{depth+d}.n2 = narray(i+1);
	                tree{depth+d}.node = p;
	                tree{depth+d}.rec = rec;
	                tree{depth+d}.act = act;
	                tree{depth+d}.lc = indices(i);
	                tree{depth+d}.rc = indices(i+1);
	                tree{depth+d}.gam1 = fun_prime(c1c2d(1:dim,:))*y1c1;
	                tree{depth+d}.gam2 = fun_prime(c1c2d(dim+1:2*dim,:))*y2c2;
	                tree{indices(i)}.par = depth+d;
	                tree{indices(i+1)}.par = depth+d;
	                mini = i;
	                mine = e_rec;
	            end
			end

			narray(mini,:) = tree{depth+d}.n1 + tree{depth+d}.n2;
	        narray(mini+1,:) = [];
	     
	        tree{depth+d}.numnodes = narray(mini,:);
	        
	        input(mini, :) = tree{depth+d}.p_norm;
	        input(mini+1, :) = [];
	        indices(mini) = depth+d;
	        indices(mini+1) = [];
		end
	else
		%classify single words
		for d=1:depth
			sig = sigmoid(params.Wl*input(d,:)' + params.bl);
			pred = (1 - alpha) * (label - sig);
			tree{d}.e_cl = 1/2*(pred * (label - sig));
			tree{d}.delta = -pred*sigmoid_prime(sig);
			tree{d}.par = existing_tree{d}.par;
		end

		for d=depth+1:2*depth-1
			p = fun(params.W1*[existing_tree{d}.c1;existing_tree{d}.c2] + params.b1);
			p_norm = p./norm(p);
			sig = sigmoid(params.Wl*p_norm + params.bl);
			pred = beta * (1 - alpha) * (label - sig);
			tree{d}.delta = -pred*sigmoid_prime(sig);

			cost = 1/2*(pred*(pred - sig));
			tree{d}.e_cl = cost;
			tree{d}.lc = existing_tree{d}.lc;
			tree{d}.rc = existing_tree{d}.rc;
			tree{d}.n1 = existing_tree{d}.n1;
			tree{d}.n2 = existing_tree{d}.n2;
			tree{d}.gam1 = existing_tree{d}.gam1;
			tree{d}.gam2 = existing_tree{d}.gam2;
			tree{d}.numnodes = existing_tree{d}.numnodes;
			tree{d}.c1 = existing_tree{d}.c1;
			tree{d}.c2 = existing_tree{d}.c2;
			tree{d}.y1c1 = existing_tree{d}.y1c1;
			tree{d}.y2c2 = existing_tree{d}.y2c2;
			
			
			if isfield(existing_tree{d},'par')
				tree{d}.par = existing_tree{d}.par;
			end
			tree{d}.p_norm = p_norm;
			tree{d}.node = p;
		end
	end
end