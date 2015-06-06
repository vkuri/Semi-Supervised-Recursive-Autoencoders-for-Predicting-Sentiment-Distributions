function [f,g,tree] = calculate(fun, fun_prime, params, ei, parameters, input, vocabIndices, label, update_flag, existing_tree)
	% TODO : Initialize gradients
	depth = ei.depth;
	tree_cost = 0;
	tree = struct;
	derivs = initStack(ei);
	gradSentence = zeros(size(input));

	if update_flag == 0
		tree = buildtree(fun, fun_prime, params, ei, parameters, input, label, update_flag. []);
		for d=depth+1:end
			tree_cost = tree_cost + tree{d}.e_rec;
		end
	else
		tree = buildtree(fun, fun_prime, params, ei, parameters, input, label, update_flag. existing_tree);
		for d=depth+1:end
			tree_cost = tree_cost + tree{d}.e_cl;
		end
	end

	%Back Propagation
	W1l = params.W1(:, 1:dim);
    W1r = params.W1(:, dim+1:2*dim);

    for d = 2*depth-2:-1:1
    	% determine if current children are left or right child of the previous layer
	    parent = tree{d}.par;
	    if tree{parent}.lc == d
	    	V = W1l;
	    else
	    	V = W1r;
	    end    
	    dp = tree{parent}.del;

    	%leaf node
    	if tree{d}.numnodes == 1
    		if update_flag == 0
    			% Figure out which node
    			gradSentence(d,:) = gradSentence(d,:) + V'*dp;
    		else
    			derivs.Wl = derivs.Wl + tree{d}.delta * tree{d}.p_norm;
    			derivs.bl = derivs.bl + tree{d}.delta;
    			gradSentence(d,:) = gradSentence(d,:) + V'*dp + params.Wl'*tree{d}.delta;
    		end
    	else
	        if update_flag == 0
	        	tree{d}.del = fun_prime(tree{d}.node) * (V'*dp + W21*tree{d}.gam1 + W22*tree{d}.gam2);
	        else
	        	derivs.bl = derivs.bl + tree{d}.delta;
	        	tree{d}.del = fun_prime(tree{d}.node) * (V'*dp + W21*tree{d}.gam1 + W22*tree{d}.gam2 + params.Wl'* tree{d}.delta);
	        end

	        derivs.b1 = derivs.b1 + tree{d}.del;
	        derivs.b2 = derivs.b2 + tree{d}.gam1 + tree{d}.gam2;
	        derivs.W1 = derivs.W1 + tree{d}.del * tree{d}.p_norm';
	        derivs.W2 = derivs.W2 + [tree{d}.gam1*tree{d}.p_norm' tree{d}.gam2*tree{d}.p_norm'];
	     end
	end

	for d=1:depth
		derivs.W(vocabIndices,:) = derivs.W(vocabIndices,:) + gradSentence(d,:);
	end
	f = tree_cost;
	g = derivs;

end