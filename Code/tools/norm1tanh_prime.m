function out = norm1tanh_prime(x)
	nrm = norm(x);
	y = (x-x.^3);     
	out = diag(1-x.^2)./nrm - y*x'./nrm^3;
end