function stack = initStack(ei)

dim = ei.dimensionality;
out = ei.outputsize;
vocab = ei.vocab;
stack = struct;
stack.W1 = zeros(dim, 2*dim); 
stack.b1 = zeros(dim, 1); 
stack.W2 = zeros(2*dim, dim); 
stack.b2 = zeros(2*dim, 1); 
stack.Wl = zeros(out, dim);
stack.bl = zeros(out, 1);
stack.W = zeros(vocab, dim);