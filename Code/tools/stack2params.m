function [params] = stack2params(stack)

params = [stack.W1(:); stack.b1(:); stack.W2(:); stack.b2(:); stack.Wl(:); stack.bl(:)];
 
%    assert(size(stack{d}.W1, 1) == size(stack{d}.b1, 1), ...
 %       ['The bias should be a *column* vector of ' ...
  %       int2str(size(stack{d}.W1, 1)) 'x1']);
     % no layer size constrain with conv nets
%      if d < numel(stack)
%         assert(mod(size(stack{d+1}.W1, 2), size(stack{d}.W1, 1)) == 0, ...
%             ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
%              ' should have matching sizes.']);
%      end
end