
function []=InformationAndDiagEntropy()
% Get ready random matrices size and number of random matrices
Mat_sizes = [4 8 16 32 100 150];
repeats = 1000;

% Then load Solveig's data
Data_dir = '/Users/elie/Dropbox/Solveig/Neuro/J. Neuroscience/selectivity';
load(fullfile(Data_dir, 'entropy_Bird_Dist_matrices_1322units.mat')) % variable: entrop
load(fullfile(Data_dir, 'gmClusters_residuals_1to8_w_ND.mat')) % variable name: indices
% avec 5 clusters: 3 = bird cluster (rouge), 5 = distance cluster (bleu)
colors = [[0 0.2 0.8]; [0.8 0.8 0]; [1 0.2 0.2]; [0.1 1 0.1]; [0 1 0.9]];

% Indentify cells belonging to the bird cluster and extract MI Bird and
% entropy of birds matrices for those
ind_B = find(indices(:,5)==3);
totMat_B(:,1) = entrop.miBird(ind_B);
totMat_B(:,2) = entrop.entropBird(ind_B);
dat_B = sortrows(totMat_B, 1);
% Fit the data of the bird cluster
coeff_B = polyfit(dat_B(:,1), dat_B(:,2), 1);
yfit_B = polyval(coeff_B, dat_B(:,1));
mdl_B = LinearModel.fit(dat_B(:,2), dat_B(:,1))
 
% Indentify cells belonging to the bird cluster and extract MI Bird and
% entropy of birds matrices for those
ind_D = find(indices(:,5)==5);
totMat_D(:,1) = entrop.miDist(ind_D);
totMat_D(:,2) = entrop.entropDist(ind_D); 
dat_D = sortrows(totMat_D, 1);
% Fit the data of the bird cluster
coeff_D = polyfit(dat_D(:,1), dat_D(:,2), 1);
yfit_D = polyval(coeff_D, dat_D(:,1));
mdl_D = LinearModel.fit(dat_D(:,2), dat_D(:,1))
    
% Calculate random matrices and plot data
for ii=1:length(Mat_sizes)
    Mat_size=Mat_sizes(ii);
    [E1,I1] = findInfAndEnt(Mat_size,repeats,'random');
    [E2,I2] = findInfAndEnt(Mat_size,repeats,'related');
    [E3,I3] = findInfAndEnt2(Mat_size,repeats,'constrained','random');
    [E4,I4] = findInfAndEnt2(Mat_size,repeats,'constrained','related');
    [E5,I5] = findInfAndEnt2(Mat_size,repeats,'free');
    Rand_NeuronLike_Info = [I1; I2; I3; I4];
    Rand_NeuronLike_Ent = [E1; E2; E3; E4];
    % Plot the Mutual information of the matrix against the entropy of the
    % diagonal
    figure(1)
    subplot(2,3,ii)
    plot(Rand_NeuronLike_Info,Rand_NeuronLike_Ent,'k.', 'MarkerSize',10)
    hold on
    plot(I5,E5,'.','MarkerFaceColor', [0.3 0.3 1], 'MarkerSize',10)
    hold on
    scatter(dat_B(:,1), dat_B(:,2), 20, 'MarkerFaceColor', colors(3, :), 'MarkerEdgeColor', 'k')
    hold on
    plot(dat_B(:,1), yfit_B, 'Color',[0.5 1 0], 'LineWidth', 2)
    legend('Random neuron like matrices','Random matrices','Bird Cluster Neurons','Neurons Fit', 'Location', 'SouthEast')
    hold off
    xlabel('Exact/Bias corrected Mutual Information for Random/Bird Matrices (bits)')
    ylabel('Normalized Entropy of the diagonal conditional probabilities')
    title(sprintf('Matrix Rand n=%d Birds Matrix',Mat_size));
    
    
    figure(2)
    subplot(2,3,ii)
    plot(Rand_NeuronLike_Info,Rand_NeuronLike_Ent,'k.', 'MarkerSize',10)
    hold on
    plot(I5,E5,'.','MarkerFaceColor', [0.3 0.3 1], 'MarkerSize',10)
    hold on
    scatter(dat_D(:,1), dat_D(:,2), 20, 'MarkerFaceColor', colors(5, :), 'MarkerEdgeColor', 'k')
    hold on
    plot(dat_D(:,1), yfit_D, 'Color', [1 0.9 0], 'LineWidth', 2)
    legend('Random neuron like matrices','Random matrices','Dist Cluster Neurons','Neurons Fit','Location', 'SouthEast')
    hold off
    xlabel('Exact/Bias corrected Mutual Information for Random/Dist Matrices (bits)')
    ylabel('Normalized Entropy of the diagonal conditional probabilities')
    title(sprintf('Matrix Rand n=%d Distance Matrix',Mat_size));
    %legend('random uniform', 'related uniform', 'random non-uniform', 'related non-uniform', 'Location', 'SouthEast')
end

% Now do the same thing but correct each value of information by the value
% obtained for totally random matrices so results are comparable with
% Solveig's data. Indeed even random matrices have some random minimum information
I_Randcorrected =  nan(length(Mat_sizes)*repeats*4,1);
E_Rand = I_Randcorrected;
for ii=1:length(Mat_sizes)
    Mat_size=Mat_sizes(ii);
    [E1,I1] = findInfAndEnt(Mat_size,repeats,'random');
    [E2,I2] = findInfAndEnt(Mat_size,repeats,'related');
    [E3,I3] = findInfAndEnt2(Mat_size,repeats,'constrained','random');
    [E4,I4] = findInfAndEnt2(Mat_size,repeats,'constrained','related');
    [E5,I5] = findInfAndEnt2(Mat_size,repeats,'free');
    % Do a fit for the random (free of assumptions) data
    coeff_F = polyfit(E5, I5, 1);
    y_fitLocal = polyval(coeff_F, E5);
    figure(3)
    plot(I5, E5,'.','MarkerFaceColor', [0.3 0.3 1], 'MarkerSize',10);
    hold on
    plot(y_fitLocal,E5, 'r', 'LineWidth',5);
    hold off
    title('Fit of the mutual information for totally random matrices vs Entropy')
    xlabel('Exact Mutual Information (bits)')
    ylabel('Normalized Entropy of the diagonal conditional probabilities')
    title(sprintf('Matrix size n=%d',Mat_sizes(ii)));
    pause(1)
    
    % Predict the expected minimum value of information for each matrix
    % given its entropy and substract that bias from the actual value of
    % information to obtain a bias free estimate of information for the
    % random matrices
    yfit_F = polyval(coeff_F, [E1 E2 E3 E4]);
    I_Randcorrected(1+(ii-1)*repeats*4 : ii*repeats*4) =[I1 I2 I3 I4] - yfit_F;
    E_Rand(1+(ii-1)*repeats*4 : ii*repeats*4) = [E1 E2 E3 E4];
end

coeff_F_all = polyfit(I_Randcorrected, E_Rand,1);
yfit_R = polyval(coeff_F_all, I_Randcorrected);
figure(4)
subplot(1,2,1)
plot(I_Randcorrected,E_Rand,'k.', 'MarkerSize',10)
hold on
scatter(dat_B(:,1), dat_B(:,2), 20, 'MarkerFaceColor', colors(3, :), 'MarkerEdgeColor', 'k')
hold on
plot(dat_B(:,1), yfit_B, 'Color',[0.5 1 0], 'LineWidth', 2)
hold on
plot(I_Randcorrected, yfit_R, 'Color', [0.7 0.7 0.7], 'LineWidth', 2)
xlabel('Bias Corrected Mutual Information (bits)')
ylabel('Normalized Entropy of the diagonal conditional probabilities')
title('Birds Matrix');
legend('Random Matrices','Bird Cluster Neurons','Neurons fit', 'Random Matrices Fit','Location','SouthEast')
hold off
subplot(1,2,2)
plot(I_Randcorrected,E_Rand,'k.', 'MarkerSize',10)
hold on
scatter(dat_D(:,1), dat_D(:,2), 20, 'MarkerFaceColor', colors(5, :), 'MarkerEdgeColor', 'k')
hold on
plot(dat_D(:,1), yfit_D, 'Color',[1 0.9 0], 'LineWidth', 2)
hold on
plot(I_Randcorrected, yfit_R,'Color', [0.7 0.7 0.7], 'LineWidth', 2)
xlabel('Bias Corrected Mutual Information (bits)')
ylabel('Normalized Entropy of the diagonal conditional probabilities')
title('Distance Matrix');
legend('Random Matrices','Dist Cluster Neurons','Neurons fit', 'Random Matrices Fit','Location','SouthEast')
hold off

% Run some models to figure out significance of entropy of clusters compare
% to random values. I code the category of matrices (neuron vs random) as
% binary values 1=neuron, 0=random
MDL_B = fitlm([[dat_B(:,1); I_Randcorrected] [ones(size(dat_B(:,1)));zeros(size(I_Randcorrected))]],[dat_B(:,2) ; E_Rand], 'interactions')
anova(MDL_B)
% Significant effect of the second column of x = random vs neurons of Birds cluster but no
% interactions (similar slopes)
MDL_D = fitlm([[dat_D(:,1); I_Randcorrected] [ones(size(dat_D(:,1)));zeros(size(I_Randcorrected))]],[dat_D(:,2) ; E_Rand], 'interactions')
anova(MDL_D)
% Significant effect of the second column of x = random vs neurons of Distance cluster but no
% interactions (similar slopes) Note that the coefficient estimate of x2 is slightly
% larger for bird cluster than distance cluster
end



                % findInfAndEnt 
function [Enorm, I] = findInfAndEnt(Mat_size, N, Distr)
%% Defining several values
% the size of the confusion matrix
if nargin==0
    Mat_size = 8;
end
% The number of iterations
if nargin<2
    N=1000;
end
% The rule to draw conditional probabilities in the diagonal of the confusion matrix
if nargin<3
    Distr = 'random'; % set to "random" for independant uniform drawing between 0 and 1 and to "related" for MonteCarlo kind of relation between values
end
    % The output variables
I=nan(N,1); % mutual information
E=nan(N,1); % Entropy


% Loop through the number of matrices n
for nn =1:N
    % Determine the values of conditional probabilities in the diagonal (k)
    % we consider that in the same line values of probability outside of the diagonal are uniform (1-k/(Mat_size-1)) 
    if strcmp(Distr, 'random')
        % idependantly determined k
        k=rand(Mat_size,1);
    elseif strcmp(Distr, 'related')
        k=nan(Mat_size,1);
        k(1) = rand;
        %related k
        for kk=1:(Mat_size-1)
            Foundit=0;
            while Foundit==0
            pot_k = randn(1,1) + k(1);
            if (0<=pot_k) && (pot_k<=1)
                k(kk+1) = pot_k;
                Foundit=1;
            end
            end
        end
    end
    
    % Calculate Information
    H_xgiveny = sum(-k.*log2(k) - (1-k).*(log2((1-k)./(Mat_size-1))))/Mat_size;
    p_xi=nan(Mat_size,1);
    for ii = 1:Mat_size
        k_local = k;
        k_local(ii) = [];
        p_xi(ii) = (k(ii) + sum((1-k_local)./(Mat_size-1)))/Mat_size;
    end
    H_x = sum(-p_xi.*log2(p_xi));
    I(nn) = H_x - H_xgiveny;
    
    % Calculate Entropy of the diagonal
    K_norm = k./sum(k);
    E(nn) = sum(-K_norm.*log2(K_norm));
end
Emax = log2(Mat_size);
%Imax = Emax;

%Inorm = I/Imax;
Enorm = E/Emax;
end


                % findInfAndEnt2
function [Enorm, I] = findInfAndEnt2(Mat_size, N, ProbaDiag,Distr)
%% Defining several values
% the size of the confusion matrix
if nargin==0
    Mat_size = 8;
end
% The number of iterations
if nargin<2
    N=1000;
end
% Should probability in the diagonal be under any contrain?
if nargin<3
    ProbaDiag = 'free';
end
% The rule to draw conditional probabilities in the diagonal of the confusion matrix
if nargin<4
    Distr = 'random'; % set to "random" for independant uniform drawing between 0 and 1 and to "related" for MonteCarlo kind of relation between values
end
    % The output variables
I=nan(N,1); % mutual information
E=nan(N,1); % Entropy


% Loop through the number of matrices n
for nn =1:N
    if strcmp(ProbaDiag,'constrained')
        % Determine the values of conditional probabilities in the diagonal (k)
        % we consider that in the same line values of probability outside of the diagonal are uniform (1-k/(Mat_size-1)) 
        if strcmp(Distr, 'random')
            % idependantly determined k
            k=rand(Mat_size,1);
        elseif strcmp(Distr, 'related')
            k=nan(Mat_size,1);
            k(1) = rand;
            %related k
            for kk=1:(Mat_size-1)
                Foundit=0;
                while Foundit==0
                pot_k = randn(1,1) + k(1);
                if (0<=pot_k) && (pot_k<=1)
                    k(kk+1) = pot_k;
                    Foundit=1;
                end
                end
            end
        end
        Mat = diag(k);

        % Create the values outside of diagonal
        for jj=1:Mat_size
            Outdiag = rand(Mat_size-1,1);
            Outdiag = Outdiag.*(1-k(jj))./sum(Outdiag);
            od=0;
            for ii=1:Mat_size
                if ii~=jj
                    od=od+1;
                    Mat(jj,ii) = Outdiag(od);
                end
            end
        end

        % Convert conditional probabilities to joint proba
        Mat_joint = Mat./Mat_size;
    elseif strcmp(ProbaDiag, 'free')
        Mat = rand(Mat_size);
        Mat = Mat./(repmat(sum(Mat,2),1,size(Mat,2)));
        k = diag(Mat);
        Mat_joint = Mat./(sum(sum(Mat)));
        
    end
    
    % Calculate Information
    I(nn) = info_matrix(Mat_joint);
    
    % Calculate Entropy of the diagonal
    K_norm = k./sum(k);
    E(nn) = sum(-K_norm.*log2(K_norm));
end
Emax = log2(Mat_size);
%Imax = Emax;

%Inorm = I/Imax;
Enorm = E/Emax;
end



