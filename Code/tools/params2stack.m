function stack = params2stack(params, ei)

dim = ei.dimensionality;
depth = ei.depth;

stack = cell(depth,1); 

for d = 1:depth
    % Create layer d
    stack{d} = struct;
    
    s = (4*dim^2 + 3*dim)*(d-1) + 1;
    e = s + 4*dim^2 + 3*dim;
    
    stack{d}.W1 = reshape(params(s : s+2*dim^2-1), dim, 2*dim); 
    stack{d}.b1 = reshape(params(s+2*dim^2 : s+2*dim^2+dim-1), dim, 1); 
    stack{d}.W2 = reshape(params(s+2*dim^2+dim : s+4*dim^2+dim-1), 2*dim, dim); 
    stack{d}.b2 = reshape(params(e-2*dim+1 : e), 2*dim, 1); 
    
end

end