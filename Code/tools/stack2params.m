function [params] = stack2params(stack)

params = [];

    
for d = 1:numel(stack)
    params = [params; stack{d}.W1(:); stack{d}.b1(:); stack{d}.W2(:); stack{d}.b2(:)];
 
    assert(size(stack{d}.W1, 1) == size(stack{d}.b1, 1), ...
        ['The bias should be a *column* vector of ' ...
         int2str(size(stack{d}.W1, 1)) 'x1']);
     % no layer size constrain with conv nets
%      if d < numel(stack)
%         assert(mod(size(stack{d+1}.W1, 2), size(stack{d}.W1, 1)) == 0, ...
%             ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
%              ' should have matching sizes.']);
%      end
end

end