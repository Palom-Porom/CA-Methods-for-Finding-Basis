function A = read_mtx(filename)
  printf("=== Benchmark for file: %s ===\n", filename);

  % ---------------------------
  % 1. Замер времени твоей функции
  % ---------------------------
  tic;
  t1 = tic;
  A = read_custom(filename);
  time_custom = toc(t1);

  printf("Custom read time: %.6f sec\n", time_custom);
  printf("Matrix size: %dx%d, nnz=%d\n", rows(A), columns(A), nnz(A));

  % ---------------------------
  % 2. Попытка сравнения с mmread (если доступна)
  % ---------------------------
  has_mmread = exist("mmread", "file");

  if has_mmread
    t2 = tic;
    B = mmread(filename);
    time_mmread = toc(t2);

    printf("mmread time: %.6f sec\n", time_mmread);

    % Проверка корректности
    diff_norm = norm(A - B, 'fro');
    printf("Difference (Frobenius norm): %.6e\n", diff_norm);
  else
    time_mmread = NaN;
    printf("mmread not available\n");
  endif

  % ---------------------------
  % 3. График времени
  % ---------------------------
  figure;
  times = [time_custom, time_mmread];
  labels = {'Custom', 'mmread'};

  bar(times);
  set(gca, 'xticklabel', labels);
  ylabel('Time (sec)');
  title('Read Performance Comparison');

  % ---------------------------
  % 4. График структуры матрицы
  % ---------------------------
  figure;
  spy(A);
  title('Sparsity Pattern of Matrix');

endfunction


% ============================
% Твоя функция (чуть оптимизирована)
% ============================
function A = read_custom(filename)
  fid = fopen(filename, 'r');

  header = fgetl(fid);
  is_symmetric = ~isempty(strfind(lower(header), 'symmetric'));

  line = fgetl(fid);
  while startsWith(line, '%')
    line = fgetl(fid);
  endwhile

  dims = sscanf(line, '%d %d %d');
  n = dims(1);
  m = dims(2);
  nnz = dims(3);

  % Предвыделение памяти (ускоряет!)
  if is_symmetric
    max_nnz = nnz * 2;
  else
    max_nnz = nnz;
  endif

  rows = zeros(max_nnz, 1);
  cols = zeros(max_nnz, 1);
  vals = zeros(max_nnz, 1);

  idx = 0;

  for i = 1:nnz
    data = fscanf(fid, '%d %d %f', 3);
    r = data(1);
    c = data(2);
    v = data(3);

    idx += 1;
    rows(idx) = r;
    cols(idx) = c;
    vals(idx) = v;

    if is_symmetric && r != c
      idx += 1;
      rows(idx) = c;
      cols(idx) = r;
      vals(idx) = v;
    endif
  endfor

  fclose(fid);

  % Обрезаем лишнее
  rows = rows(1:idx);
  cols = cols(1:idx);
  vals = vals(1:idx);

  A = sparse(rows, cols, vals, n, m);
endfunction
