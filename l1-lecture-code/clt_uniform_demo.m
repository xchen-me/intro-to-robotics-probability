function results = clt_uniform_demo()
%CLT_UNIFORM_DEMO Sum IID Uniform[0,1] random variables, then standardize.
% Author: Xu Chen
% Setup: 100000 independent trials, 12 independent draws in each row.
% Assumptions: IID uniforms, E[X]=1/2, Var[X]=1/12; no shared bias.
% Takeaway: raw sums broaden; centered/scaled sums approach N(0,1).
% Two uniforms sum to an EXACT triangle; finite sums are not exactly normal.
% Run: addpath('code/l1'); results = clt_uniform_demo;
% Base MATLAB R2016b or newer; no Statistics Toolbox required.
% Python/MATLAB seeds do not imply identical draws across generators.

rng(2026, 'twister');          % Reproducible within MATLAB
M = 100000;
ns = [1 2 3 12];
X = rand(M, max(ns));         % Columns are independent Uniform[0,1] draws
C = cumsum(X, 2);             % Column n is S_n = X_1 + ... + X_n
results = zeros(numel(ns), 5);
output = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'code-results', 'l1');
if ~exist(output, 'dir'), mkdir(output); end
x = linspace(-4, 4, 401);
phi = exp(-x.^2/2)/sqrt(2*pi); % Standard normal density, no normpdf needed
fRaw = figure('Name', 'Raw sums', 'Position', [100 100 1400 350]);
fStd = figure('Name', 'Standardized sums', 'Position', [100 480 1400 350]);
for j = 1:numel(ns)
    n = ns(j);
    S = C(:, n);
    Z = (S - n/2)/sqrt(n/12); % Analytic mean/variance, not sample estimates
    results(j,:) = [n, mean(S), var(S,1), n/2, n/12];
    fprintf('n=%2d: mean=%.5f (theory %.5f), variance=%.5f (theory %.5f)\n', ...
        n, mean(S), n/2, var(S,1), n/12);

    figure(fRaw); subplot(1,4,j);
    histogram(S, linspace(0,n,101), 'Normalization', 'pdf', ...
        'FaceColor', [0.294 0.180 0.514], 'EdgeColor', 'none'); hold on;
    if n == 2
        q = linspace(0,2,201);
        plot(q, max(0,1-abs(q-1)), 'LineWidth', 1.8);
    end
    title(sprintf('$n=%d$',n), 'Interpreter','latex');
    xlabel('$s$', 'Interpreter','latex');
    ylabel('$\widehat{p}_{S_n}(s)$', 'Interpreter','latex'); box off;

    figure(fStd); subplot(1,4,j);
    % Full support bins avoid renormalizing only the visible [-4,4] window.
    histogram(Z, linspace(-sqrt(3*n),sqrt(3*n),101), 'Normalization','pdf', ...
        'FaceColor', [0.294 0.180 0.514], 'EdgeColor', 'none'); hold on;
    plot(x, phi, 'LineWidth', 1.8); xlim([-4 4]);
    title(sprintf('$n=%d$',n), 'Interpreter','latex');
    xlabel('$z$', 'Interpreter','latex');
    ylabel('$\widehat{p}_{Z_n}(z)$', 'Interpreter','latex'); box off;
end
% Normalization='pdf' divides counts by total count AND bin width.
% Thus sum(bar_height * bin_width)=1, unlike a count histogram.
set(fRaw, 'PaperPositionMode','auto'); set(fStd, 'PaperPositionMode','auto');
print(fRaw, fullfile(output,'clt_raw_matlab.png'), '-dpng','-r180');
print(fStd, fullfile(output,'clt_standardized_matlab.png'), '-dpng','-r180');
save(fullfile(output,'clt_matlab_results.mat'), 'results','ns','M');
% Results columns: n, measured mean, measured variance, theoretical mean, variance.
end
