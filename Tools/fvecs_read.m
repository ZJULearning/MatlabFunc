% Read a set of vectors stored in the fvec format (int + n * float)
% The function returns a set of output vector (one vector per column)
%
% Syntax:
%   v = fvecs_read (filename)     -> read all vectors
%   v = fvecs_read (filename, n)  -> read n vectors
%   v = fvecs_read (filename, [a b]) -> read the vectors from a to b (indices starts from 1)
function v = fvecs_read (filename, bounds)

% open the file and count the number of descriptors
fid = fopen (filename, 'rb');

if fid == -1
  error ('I/O error : Unable to open the file %s\n', filename)
end

% Read the vector size
d = fread (fid, 1, 'int');
vecsizeof = 1 * 4 + d * 4;

% Get the number of vectrors
fseek (fid, 0, 1);
a = 1;
bmax = ftell (fid) / vecsizeof;
b = bmax;

if nargin >= 2
  if length (bounds) == 1
    b = bounds;

  elseif length (bounds) == 2
    a = bounds(1);
    b = bounds(2);
  end
end

assert (a >= 1);
if b > bmax
  b = bmax;
end

if b == 0 | b < a
  v = [];
  fclose (fid);
  return;
end

% compute the number of vectors that are really read and go in starting positions
n = b - a + 1;
fseek (fid, (a - 1) * vecsizeof, -1);

% read n vectors
v = fread (fid, (d + 1) * n, 'float=>single');
v = reshape (v, d + 1, n);

% Check if the first column (dimension of the vectors) is correct
assert (sum (v (1, 2:end) == v(1, 1)) == n - 1);
v = v (2:end, :);

fclose (fid);
