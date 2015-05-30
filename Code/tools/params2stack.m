function stack = params2stack(params, ei)

dim = ei.dimensionality;
out = ei.outputsize;
l = size(params,1); 
stack = struct;
stack.W1 = reshape(params(1 : 2*dim^2), dim, 2*dim); 
stack.b1 = reshape(params(2*dim^2+1 : 2*dim^2+dim), dim, 1); 
stack.W2 = reshape(params(2*dim^2+dim+1 : 4*dim^2+dim), 2*dim, dim); 
size(params(4*dim^2+dim+1 : 4*dim^2+3^dim))
stack.b2 = reshape(params(4*dim^2+dim+1 : 4*dim^2+3*dim), 2*dim, 1); 
stack.Wl = reshape(params(4*dim^2+3*dim+1 : 4*dim^2+3*dim+out*dim), out, dim);
stack.bl = reshape(params(4*dim^2+3*dim+out*dim+1:l), out, 1);