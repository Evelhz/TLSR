function [H] = binaryH(label)
for i = 1 : length(label)
    a = label(i);
    H(a, i) = 1;
end