A = read_mtx('bcsstk24.mtx');
b = ones(size(A,1), 1);
x = A \ b;
disp(x);
