# -*- coding:utf-8 -*-
"""
name:scCAPE
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
from sklearn import cluster as clust
from scipy import sparse as ss
from scipy import optimize as so
import scipy as sp
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise_distances
import scanpy as sc
import random
import torch
import scib
from econml.grf import CausalForest
from econml.grf import MultiOutputGRF
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
from scCAPE.modelDL import CAPE
from scCAPE.dataDL import load_dataset_splits
from scCAPE.plotting import plot_loss
from scCAPE.plotting import plot_metric
# from modelDL import CAPE
# from dataDL import load_dataset_splits
# from plotting import plot_loss
# from plotting import plot_metric

################# Part1: oNMF
def minibatchkmeans(m, k):
    """
    Perform mini-batch kmeans
    Parameters
    ----------
    m: sparse matrix
    Normalized data
    k: int
    Numbers of clusters
    Returns
    -------
    Cluster centroids, cluster memberships
    """
    model = clust.MiniBatchKMeans(n_clusters=k)  # prepare k means model
    model.fit(m)  # fit with data
    return model.cluster_centers_.T, model.predict(m)


def oNMF(X, k, n_iters=500, verbose=1, residual=1e-4, tof=1e-4):
    """
    Perform non-negative orthogonal matrix factorization
    Revised based on https://github.com/thomsonlab/popalign
    Parameters
    ----------
    X: sparse matrix
    Normalized data
    k: int
    Number of factors
    n_iters: int
    Maximum number of iterations
    verbose: boolean
    Print iteration numbers if 1(True)
    residual: float
    Algorithm converged if the reconstruction error is below the residual values
    tof: float
    Tolerance of the stopping condition
    Returns
    -------
    factor loadings (W) and factor-level expressions (Z)
    """
    r, c = X.shape  # r number of features(genes), c number of samples (cells)
    A, inx = minibatchkmeans(X.T, k)  # Initialize the features (centers of the kmeans clusters)
    orthogonal = [1, 0]  # orthogonality constraints

    Y = ss.csc_matrix((np.ones(c), (inx, range(c))), shape=(k, c)).todense()
    Y = Y + 0.2
    if np.sum(orthogonal) == 2:
        S = A.T.dot(X.dot(Y.T))
    else:
        S = np.eye(k)

    X = X.todense()
    XfitPrevious = np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(n_iters):
            if orthogonal[0] == 1:
                A = np.multiply(A, (X.dot(Y.T.dot(S.T))) / (A.dot(A.T.dot(X.dot(Y.T.dot(S.T))))))
            else:
                A = np.multiply(A, (X.dot(Y.T)) / (A.dot(Y.dot(Y.T))))
            A = np.nan_to_num(A)
            A = np.maximum(A, np.spacing(1))

            if orthogonal[1] == 1:
                Y = np.multiply(Y, (S.T.dot(A.T.dot(X))) / (S.T.dot(A.T.dot(X.dot(Y.T.dot(Y))))))
            else:
                Y = np.multiply(Y, (A.T.dot(X)) / (A.T.dot(A.dot(Y))))
            Y = np.nan_to_num(Y)
            Y = np.maximum(Y, np.spacing(1))

            if np.sum(orthogonal) == 2:
                S = np.multiply(S, (A.T.dot(X.dot(Y.T))) / (A.T.dot(A.dot(S.dot(Y.dot(Y.T))))))
                S = np.maximum(S, np.spacing(1))

            if np.mod(i, 100) == 0 or i == n_iters - 1:
                if verbose:
                    print('......... Iteration #%d' % i)
                XfitThis = A.dot(S.dot(Y))
                fitRes = np.linalg.norm(XfitPrevious - XfitThis, ord='fro')
                XfitPrevious = XfitThis
                curRes = np.linalg.norm(X - XfitThis, ord='fro')
                if tof >= fitRes or residual >= curRes or i == n_iters - 1:
                    print('Orthogonal NMF performed with %d iterations\n' % (i + 1))
                    break
    return A, Y  # return factor loadings and projection for the cells in X


def nnls(W, V):
    """
    Project 'V' onto 'W' with non-negative least squares
    Parameters
    ----------
    W: array
    Factor loadings
    V: array
    Gene expression vector for a cell
    Returns
    -------
    The factor-level expression of a cell
    """
    return so.nnls(W, V)[0]


def reconstruction_errors(X, loadings, njobs=2):
    """
    Compute the mean square errors between the original data 'X'
    and reconstructed matrices using projection spaces from the list of loadings
    Parameters
    ----------
    X: sparse matrix
    Normalized data
    loadings: list
    List of factor loading arrays to try
    njobs: int
    Number of parallel jobs
    Returns
    -------
    The reconstruction errors, projections (factor-level expression)
    """
    errors = []
    projs = []
    D = X.toarray()  # dense matrix to compute error
    for j in range(len(loadings)):  # For each factor loading matrix j in loadings list
        print('Progress: %d of %d' % ((j + 1), len(loadings)), end='\r')
        Wj = loadings[j]
        with Pool(njobs) as p:
            Hj = p.starmap(nnls, [(Wj, X[:, i].toarray().flatten()) for i in
                                  range(X.shape[1])])  # project each cell i of normalized data onto the current W
        Hj = np.vstack(Hj)  # Hj is projected data onto Wj
        projs.append(Hj)  # store projection
        Dj = Wj.dot(Hj.T)  # compute reconstructed data: Dj = Wj.Hj
        errors.append(
            mean_squared_error(D, Dj))  # compute mean squared error between original data D and reconstructed data Dj
    return errors, projs


def scale_W(W):
    """
    Scale the loadings. Divide each loading vector by its L2-norm.
    Parameters
    ----------
    W: array
    Loading matrix
    Returns
    -------
    Scaled loading matrix
    """
    norms = [np.linalg.norm(np.array(W[:, i]).flatten()) for i in
             range(W.shape[1])]  # compute the L2-norm of each feature
    return np.divide(W, norms)  # divide each factor by its L2-norm


def find_best_k(oNMFres, dataset_path, alpha=1, SS_weight=0.3):
    """
    Find the optimal k by minimizing f(k;alpha,j)
    Parameters
    ----------
    oNMFres: dict
    The dict contains loading matrix, projections and errors of each factor number k
    dataset_path: path
    The path where the results will be stored
    alpha: float
    The power of the mean in selecting the optimal k
    SS_weight: float
    The weight of the Specificity Score(SS) term in selecting the optimal k
    """
    print('Finding optimal factor number k...')
    errors = oNMFres['onmf']['errors']
    nfactors = oNMFres['onmf']['nfactors']

    # Normalize errors across all k
    errors_norm = np.divide(errors, np.max(errors))

    # Set all powers and scaling parameters to try
    alphas = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
    # alphas = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25]
    if alpha not in alphas:
        alphas.append(alpha)
        alphas = np.sort(alphas).tolist()

    jrange = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # jrange=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
    if SS_weight not in jrange:
        jrange.append(SS_weight)
        jrange = np.sort(jrange).tolist()

    # Calculate max(SS_k(alpha)) for all alphas
    basescales = []
    for i in range(len(alphas)):
        SS = []
        for factor in range(len(nfactors)):
            dat = oNMFres['onmf']['projs'][factor]
            factor_std = dat.std(axis=0)
            factor_mean = dat.mean(axis=0)
            SS.append((np.mean(np.divide(factor_std, factor_mean ** alphas[i]))))
        currscale = np.max(np.array(SS))
        basescales.append(currscale)

    # Calculate f(k;alpha,j) values for all j and alpha
    fk_df = pd.DataFrame()
    minvals = []

    for i in range(len(alphas)):
        curralpha = alphas[i]
        currbasescale = basescales[i]
        currminvals = []
        SS = []
        for factor in range(len(nfactors)):
            dat = oNMFres['onmf']['projs'][factor]
            factor_std = dat.std(axis=0)
            factor_mean = dat.mean(axis=0)
            SS.append((np.mean(np.divide(factor_std, factor_mean ** alphas[i]))))
        for j in jrange:
            currscale = j / currbasescale
            curr_values = np.array(errors_norm) - currscale * (np.array(SS))  # f(k;alpha,j)
            currmin = nfactors[np.argwhere(curr_values == np.min(curr_values))[0][0]]

            curr_df = {'vals': np.append(curr_values, np.min(curr_values)),
                       'k': np.append(nfactors, currmin)}
            curr_df = pd.DataFrame(curr_df)
            curr_df['scale'] = currscale
            curr_df['a'] = curralpha
            curr_df['j'] = j
            curr_df['col'] = np.append(0 * curr_values, 1)
            fk_df = fk_df.append(curr_df, ignore_index=True)
            currminvals.append(currmin)

        minvals.append(currminvals)

    # Find best k given alpha and multiplier:
    irow = alphas.index(alpha)
    icol = jrange.index(SS_weight)
    idx_flat = irow * len(jrange) + icol
    bestk = minvals[irow][icol]

    # Save the factor loading matrix under best k
    idx = oNMFres['onmf']['nfactors'].index(bestk)
    np.save(os.path.join(dataset_path, 'W_ini.npy'), oNMFres['onmf']['loadings'][idx].T)

    # Plots
    # Plot1: Normalized error curve
    plt.scatter(nfactors, errors_norm, marker=".", s=100)
    plt.plot(nfactors, errors_norm, marker=".")
    plt.ylabel('Errors(normalized)')
    plt.savefig(os.path.join(dataset_path, 'oNMF_errors.png'), dpi=1000, bbox_inches="tight")
    plt.close()

    # Plot2: f(k; alpha, j) curves
    with sns.plotting_context('notebook', font_scale=1.7):
        g = sns.FacetGrid(fk_df, col="j", row="a", hue="col")
        g = (g.map(plt.scatter, "k", "vals", marker=".", s=400))  # .set(ylim=(0.03,0.055))
        g = (g.map(plt.plot, "k", "vals", marker="."))  # .set(ylim=(0.03,0.055))

        # Highlight the curve that uses the chosen parameters
        axes = g.axes.flatten()
        ax = axes[idx_flat]
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('magenta')
            spine.set_linewidth(2)

        plt.savefig(os.path.join(dataset_path, 'oNMF_fk_curves.png'), dpi=1000, bbox_inches="tight")
        plt.close()

    # Plot3: the optimal k under different alpha and j
    plt.figure(figsize=(5, 7))
    heat_map = sns.heatmap(minvals, annot=True, cmap='viridis')
    plt.yticks(plt.yticks()[0], alphas)
    plt.ylabel('alpha (a)')
    plt.xlabel('SS weight (j)')
    plt.yticks(rotation=0)
    plt.xticks(np.arange(len(jrange)) + 0.5, jrange)
    # highlight the k value selected in the data
    heat_map.add_patch(plt.Rectangle((icol, irow), 1, 1, fill=False, edgecolor='magenta', lw=2))
    plt.savefig(os.path.join(dataset_path, 'oNMF_optimal_k.png'), dpi=1000, bbox_inches="tight")
    plt.close()
    return bestk


def onmf(data, dataset_name, ncells=2000, nfactors=list(range(5, 16)), nreps=2, niters=500, njobs=2, alpha=1,
         SS_weight=0.3):
    """
    Perform oNMF. Compute factor loadings, factor-level expression (projections) and errors.
    And then select the optimal factor number k.
    Parameters
    ----------
    data: AnnData
    Single-cell expression object
    dataset_name: string
    The name of dataset to makedir (storing results)
    ncells: int
    Number of cells to use
    nfactors: int or list of ints
    Number(s) of factors to use
    nreps: int
    Number of repetitions to perform for each k in nfactors
    niters: int
    Maximum number of iterations to perform
    njobs: int
    Number of parallel jobs
    alpha: float
    The power of the mean in selecting the optimal k
    SS_weight: float
    The weight of the Specificity Score(SS) term in selecting the optimal k
    """

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)  # makedir
    dataset_path = os.path.join(dataset_name, 'oNMF')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)  # makedir to store oNMF results

    if type(nfactors) is int:
        nfactors = [nfactors]

    maxncells = data.shape[1]  # get total number of cells
    if 2 * ncells > maxncells:  # if ncells is larger than twice the number of cells
        ncells = int(np.floor(maxncells / 2))  # adjust number down

    # randomly select ncells and divide into training and cross-validation dataset
    idx = np.random.choice(data.shape[1], 2 * ncells, replace=False)
    idx1 = idx[0:ncells]
    idx2 = idx[ncells:]

    print('Computing W matrices...')
    with Pool(njobs) as p:
        res = p.starmap(oNMF, [(data[:, idx1], x, niters) for x in
                               np.repeat(nfactors, nreps)])  # run ONMF in parallel for each possible k nreps times

    loadings = [scale_W(res[i][0]) for i in range(len(res))]  # scale the different feature spaces
    print('Computing reconstruction errors...')
    errors, projs = reconstruction_errors(data[:, idx2],
                                          loadings, njobs)  # compute the reconstruction errors

    # choose the best out of all replicates
    mlist = np.repeat(nfactors, nreps)

    bestofreps = []
    for i in range(len(nfactors)):
        curridx = np.where(mlist == nfactors[i])[0]
        currerrors = [errors[i] for i in curridx]
        bestidx = np.argmin(currerrors)
        bestofreps.append(curridx[bestidx])

    # select only the best of the replicate feature sets
    loadings = [loadings[i] for i in bestofreps]
    errors = [errors[i] for i in bestofreps]
    projs = [projs[i] for i in bestofreps]

    oNMFres = {}
    oNMFres['onmf'] = {}
    oNMFres['onmf']['loadings'] = loadings
    oNMFres['onmf']['projs'] = projs
    oNMFres['onmf']['errors'] = errors
    oNMFres['onmf']['nfactors'] = nfactors

    if len(nfactors) > 1:  # if more than one factor numbers in nfactors, find the optimal k
        find_best_k(oNMFres, dataset_path, alpha=alpha, SS_weight=SS_weight)
    else:
        np.save(os.path.join(dataset_path, 'W_ini.npy'), oNMFres['onmf']['loadings'][0].T)


    print('Finish oNMF initialization. See the results in folder:{}'.format(dataset_path))


################# Part2: Adversarial training

def evaluate_r2(autoencoder, dataset):
    """
    Calculate R2 on all genes or top50 DEGs
    Parameters
    ----------
    autoencoder: CAPE class
    autoencoder model
    dataset: AnnData
    Returns
    -------
    R2 between true and reconstructed gene expression over all genes or top50 DEGs
    """
    mean_score, mean_score_de = [], []
    for pert_category in dataset.de_genes.keys():
        de_idx = np.where(dataset.var_names.isin(np.array(dataset.de_genes[pert_category])[0:50]))[0]
        idx = np.where(dataset.perts_fullname == pert_category)[0]
        if len(idx) > 30:
            genes_predict = autoencoder.forward(dataset.genes[idx, :], dataset.perts[idx, :])
            mean_predict = np.array((genes_predict).detach().cpu())
            y_true = dataset.genes[idx, :].numpy()
            yp_m = mean_predict.mean(0)
            yt_m = y_true.mean(axis=0)
            mean_score.append(r2_score(yt_m, yp_m))
            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de]
    ]


def evaluate(autoencoder, datasets):
    """
    Evaluate the reconstruction along training
    Parameters
    ----------
    autoencoder: CAPE class
    autoencoder model
    datasets: SubDataset class
    Returns
    -------
    A dict contains evaluated R2
    """
    autoencoder.eval()
    with torch.no_grad():
        stats_train = evaluate_r2(
            autoencoder,
            datasets["training"].subset_condition(control=False)
        )
        stats_test = evaluate_r2(
            autoencoder,
            datasets["test"].subset_condition(control=False)
        )
        evaluation_stats = {
            "training": stats_train,
            "test": stats_test}
    autoencoder.train()
    return evaluation_stats


def CAPE_train(data_path, dataset_name, model_index=0, seed=0, perturbation_key='condition', split_key=None,
               max_epochs=300, lambda_adv=1, lambda_ort=0.5, patience=5, hparams=None, verbose=True):
    """
    Disentangling perturbation effects from inherent cell-state variations with adversarial training.
    The embeddings (basal state, factor-level expression), the orthogonal loading matrix,
     the model file (.pt file), the loss and metric curves are stored in dataset_name/CAPE.
    Parameters
    ----------
    data_path: string
    The path where the dataset locates
    dataset_name: string
    The name of dataset to makedir (storing results)
    model_index: int
    The index of the model, which is used in storing results (e.g. 'stored_model_index={}.pt')
    seed: int
    The random seed
    perturbation_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    split_key: string
    The column name of the training/test label in AnnData, default  None
    (randomly split the data into training/test)
    max_epochs: int
    The maximum training epochs
    lambda_adv: float
    The weight of adversarial loss, default 1
    lambda_ort: float
    The weight of orthogonal loss, default 0.5
    patience: int
    Early stopping.
    The training stops if the reconstruction is not improving in the latest 5*patience epochs.
    hparams: dictionary
    The dictionary contains hyperparameters in the networks,
    like the dimension of basal state, layers of neural networks, etc.
    verbose: bool
    Printing the loss (True) or not (False) during training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    datasets = load_dataset_splits(data=data_path, perturbation_key=perturbation_key, split_key=split_key)

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)  # makedir
    dataset_path = os.path.join(dataset_name, 'CAPE')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)  # makedir to store adversarial training results

    num_positives = torch.sum(datasets["all"].perts, dim=0)
    num_negatives = len(datasets["all"].perts) - num_positives
    pos_weight = num_negatives / num_positives
    if not os.path.exists(os.path.join(dataset_name, 'oNMF', 'W_ini.npy')):
        raise NameError('Cannot find the file: W_ini.npy. Please run the oNMF to initialize W or check the filename.')
    W_ini = (torch.from_numpy(np.load(os.path.join(dataset_name, 'oNMF', 'W_ini.npy')))).to(torch.float32)

    autoencoder = CAPE(
        num_genes=datasets["training"].num_genes,
        num_perts=datasets["training"].num_perts,
        pos_weight=pos_weight,
        W_ini=W_ini,
        device=device,
        seed=seed,
        patience=patience,
        loss_ae="mse",
        hparams=hparams,
        lambda_ort=lambda_ort
    )

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=autoencoder.hparams["batch_size"],
                shuffle=True
            )
        }
    )

    lambda_schedule = np.concatenate([np.linspace(0.0, lambda_adv, num=round(max_epochs / 4)),
                                      np.repeat(lambda_adv, max_epochs - round(max_epochs / 4))])

    for epoch in range(max_epochs):
        epoch_training_stats = defaultdict(float)
        for data in datasets["loader_tr"]:
            genes, perts, cell_states = data[0], data[1], data[2]
            minibatch_training_stats = autoencoder.update(genes=genes, perts=perts, cell_states=cell_states,
                                                          lambda_val=lambda_schedule[epoch])
            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(epoch_training_stats[key])
        autoencoder.history["epoch"].append(epoch)
        stop = (epoch == max_epochs - 1)
        if ((epoch + 1) % 5) == 0 or stop:
            evaluation_stats = evaluate(autoencoder, datasets)
            if verbose:
                print('Finish epoch:{}'.format(epoch))
                print('loss:{}'.format(autoencoder.history["loss_reconstruction"][-1]))
                print('iteration:{}'.format(autoencoder.iteration))
                print('l2 loss: {}'.format(autoencoder.history["l2loss"][-1]))
            for key, val in evaluation_stats.items():
                if not (key in autoencoder.history.keys()):
                    autoencoder.history[key] = []
                autoencoder.history[key].append(val)
            autoencoder.history["stats_epoch"].append(epoch)
            stop = stop or autoencoder.early_stopping(evaluation_stats["test"][1])
            if stop:
                print('Training stop at epoch {}'.format(epoch))
                torch.save(
                    (autoencoder.state_dict(), autoencoder.history, autoencoder.W),
                    os.path.join(
                        dataset_path,
                        "stored_model_index={}.pt".format(model_index),
                    ),
                )
                break

    ## plot
    if verbose:
        print("Plotting the loss and metrics...")
    plot_loss(history=autoencoder.history, save_path=dataset_path, model_index=model_index)
    plot_metric(history=autoencoder.history, save_path=dataset_path, model_index=model_index, epoch_min=9)

    ## output basal, factor expression (treated) and W
    if verbose:
        print("Calculating the basal state and factor-level expression of each cell...")

    gene_loading = np.array((autoencoder.W[0]).detach().cpu())
    gene_loading[gene_loading < 0] = 0
    np.save(os.path.join(dataset_path, 'model_index={}_gene_loading.npy').format(model_index), gene_loading)

    genes = datasets["all"].genes
    perts = datasets["all"].perts

    _, latent_basal, latent_treated = autoencoder.forward(genes, perts, return_latent_basal=True,
                                                          return_latent_treated=True)
    basal = pd.DataFrame(latent_basal.detach().cpu().numpy(), index=datasets["all"].cell_names)
    treated = pd.DataFrame(latent_treated.detach().cpu().numpy(), index=datasets["all"].cell_names)

    basal_adata = sc.AnnData(basal, basal.index.to_frame(), basal.columns.to_frame())
    basal_adata.obs.columns = basal_adata.obs.columns.astype(str)
    basal_adata.var.columns = basal_adata.var.columns.astype(str)
    basal_adata.obs['condition'] = datasets["all"].perts_names
    basal_adata.obs['cell_type'] = datasets["all"].cell_state
    basal_adata.obs['control'] = datasets["all"].is_control
    # check the mixing performance
    if verbose:
        print('Checking the mixing performance...')
    perts_name_list = list(set(list(basal_adata.obs.condition.values)))
    perts_name_list.remove('control')
    basal_adata.obsm['X_pca'] = basal_adata.X
    asw_res1 = scib.metrics.metrics(adata=basal_adata, adata_int=basal_adata,
                                   batch_key='control', label_key='cell_type',
                                   silhouette_=True).loc['ASW_label/batch'][0]

    if len(perts_name_list) <= 20:
        all_pert_asw_list = []
        for pert in perts_name_list:
            res = scib.metrics.metrics(adata=basal_adata[basal_adata.obs.condition.isin([pert, 'control'])],
                                       adata_int=basal_adata[basal_adata.obs.condition.isin([pert, 'control'])],
                                       batch_key='condition', label_key='cell_type', silhouette_=True).loc[
                'ASW_label/batch'][0]
            all_pert_asw_list.append(res)
        all_pert_asw_mean = np.mean(all_pert_asw_list)
    else:
        all_pert_asw_mean = 1

    if asw_res1 < 0.83 or all_pert_asw_mean < 0.83:
        print('Warning: The mixing performance is not good. Try a higher lambda_adv, or set a lower gradient penalty'+
            ' for the discriminator by using hparams={\'penalty_adversary\':val}, val<10')

    tr_adata = sc.AnnData(treated, treated.index.to_frame(), treated.columns.to_frame())
    tr_adata.obs.columns = tr_adata.obs.columns.astype(str)
    tr_adata.var.columns = tr_adata.var.columns.astype(str)
    tr_adata.obs['condition'] = datasets["all"].perts_names
    tr_adata.obs['cell_type'] = datasets["all"].cell_state
    del basal_adata.obsm['X_pca']
    basal_adata.obs = basal_adata.obs.drop('control', axis=1)
    basal_adata.write(os.path.join(dataset_path, 'model_index={}_basal.h5ad'.format(model_index)))
    tr_adata.write(os.path.join(dataset_path, 'model_index={}_treated.h5ad'.format(model_index)))

    if verbose:
        print('Finish adversarial training. See the results in folder:{}'.format(dataset_path))


################# Part3: Growing random forests

def CF_single_target_single_factor(target, factor, basal, treated, adata, pert_key='condition', n_estimators=2000,
                                   min_samples_leaf=5, one_side=False, random_state=0, alpha=0.05):
    """
    Calculating the perturbation effect for a target-factor pair.
    Parameters
    ----------
    target: string
    Name of the target
    factor: int
    Index of the factor
    basal: AnnData
    Basal state
    treated: AnnData
    Factor-level expression
    adata: AnnData
    The original dataset
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    n_estimators: int
    Number of trees
    min_samples_leaf: int
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at least
    min_samples_leaf training samples in each of the left and right branches.
    one_side: bool
    Conducting one-side tests (True) or two-side tests (False), default False
    random_state: int
    Random seed to obtain a deterministic behaviour during fitting
    alpha: float
    Significance level
    Returns
    -------
    CATE, q-value and sig of each cell
    """
    # set variables for causal forest Y=outcome, T=treatment, X=covariates,
    X = basal[basal.obs.condition.isin([target, 'control'])].X
    Y = treated[treated.obs.condition.isin([target, 'control'])].X[:, factor]
    T = 1 - adata[adata.obs[pert_key].isin([target, 'control'])].obs.control

    # set parameters for causal forest
    causal_forest = CausalForest(criterion='het',
                                 n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=random_state
                                 )
    # fit train data to causal forest model
    causal_forest.fit(X, T, Y)

    # use causal forest model to estimate treatment effects
    theta, var = causal_forest.predict_and_var(X)

    tau_factor_mean = theta.reshape(theta.shape[0], theta.shape[1])
    tau_factor_var = var.reshape(var.shape[0], var.shape[1])
    tau_factor_sd = np.sqrt(tau_factor_var)
    # perform one-side test
    tau_p_val_factor = 2 * norm.sf(x=abs(tau_factor_mean[:, 0]) / (tau_factor_sd[:, 0] + 1e-6), loc=0, scale=1)
    if one_side:
        tau_p_val_factor = tau_p_val_factor / 2
    tau_q_val_factor = fdrcorrection(tau_p_val_factor)[1]
    sig_factor = (tau_q_val_factor <= alpha).astype(int)
    return tau_factor_mean[:, 0], tau_q_val_factor, sig_factor


def CF_single_target_all_factor(target, basal, treated, adata, pert_key='condition', n_estimators=2000,
                                min_samples_leaf=5, one_side=False, random_state=0, alpha=0.05):
    """
    Calculating the perturbation effects of a target to all factors.
    Parameters
    ----------
    target: string
    Name of the target
    basal: AnnData
    Basal state
    treated: AnnData
    Factor-level expression
    adata: AnnData
    The original dataset
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    n_estimators: int
    Number of trees
    min_samples_leaf: int
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at least
    min_samples_leaf training samples in each of the left and right branches.
    one_side: bool
    Conducting one-side tests (True) or two-side tests (False), default False
    random_state: int
    Random seed to obtain a deterministic behaviour during fitting
    alpha: float
    Significance level
    Returns
    -------
    CATE, q-value and sig on each factor of each cell (array: [num_cells, num_factors])
    """
    # set variables for causal forest Y=outcome, T=treatment, X=covariates, W=effect_modifiers
    X = basal[basal.obs.condition.isin([target, 'control'])].X
    Y = treated[treated.obs.condition.isin([target, 'control'])].X
    T = 1 - adata[adata.obs[pert_key].isin([target, 'control'])].obs.control

    # set parameters for causal forest
    causal_forest = CausalForest(criterion='het',
                                 n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=random_state
                                 )
    # fit train data to causal forest model
    multi_forest = MultiOutputGRF(causal_forest)
    multi_forest.fit(X, T, Y)

    # use causal forest model to estimate treatment effects
    theta, var = multi_forest.predict_and_var(X)

    tau_factor_mean = theta.reshape(theta.shape[0], theta.shape[1])
    tau_factor_var = var.reshape(var.shape[0], var.shape[1])
    tau_factor_sd = np.sqrt(tau_factor_var)
    # perform one-side test
    tau_p_val_factor = np.ones_like(tau_factor_mean) * 0.5
    tau_q_val_factor = np.ones_like(tau_factor_mean) * 0.5
    for j in range(tau_p_val_factor.shape[1]):
        tau_p_val_factor[:, j] = 2 * norm.sf(x=abs(tau_factor_mean[:, j]) / (tau_factor_sd[:, j] + 1e-6), loc=0,
                                             scale=1)
        if one_side:
            tau_p_val_factor[:, j] = tau_p_val_factor[:, j] / 2
        tau_q_val_factor[:, j] = fdrcorrection(tau_p_val_factor[:, j])[1]
    sig_factor = (tau_q_val_factor <= alpha).astype(int)
    return tau_factor_mean, tau_q_val_factor, sig_factor


def CF_all_target_all_factor(dataset_name, basal, treated, adata, pert_key='condition', n_estimators=2000,
                             min_samples_leaf=5, one_side=False, verbose=True, random_state=0, alpha=0.05):
    """
    Calculating the perturbation effects of all targets to all factors.
    CATE, q-value and sig for each target on each factor of each cell are stored in dataset_name/CausalForests
    (dictionary, key: target name, value: array [num_cells, num_factors])
    Parameters
    ----------
    dataset_name: string
    The name of dataset to makedir (storing results)
    basal: AnnData
    Basal state
    treated: AnnData
    Factor-level expression
    adata: AnnData
    The original dataset
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    n_estimators: int
    Number of trees
    min_samples_leaf: int
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at least
    min_samples_leaf training samples in each of the left and right branches.
    one_side: bool
    Conducting one-side tests (True) or two-side tests (False), default False
    verbose: bool
    Print progress (True) or not (False) during running
    random_state: int
    Random seed to obtain a deterministic behaviour during fitting
    alpha: float
    Significance level
    """
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)  # makedir
    dataset_path = os.path.join(dataset_name, 'CausalForests')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)  # makedir to store causal forests results
    tau_factor_mean_all_dict = {}
    tau_q_val_factor_all_dict = {}
    sig_factor_all_dict = {}
    for pert in adata.obs[pert_key].cat.categories:
        if pert != 'control':
            tau_factor_mean, tau_q_val_factor, sig_factor = CF_single_target_all_factor(target=pert, basal=basal,
                                                                                        treated=treated, adata=adata,
                                                                                        n_estimators=n_estimators,
                                                                                        min_samples_leaf=min_samples_leaf,
                                                                                        one_side=one_side,
                                                                                        random_state=random_state,
                                                                                        alpha=alpha)
            tau_factor_mean_all_dict[pert] = tau_factor_mean
            tau_q_val_factor_all_dict[pert] = tau_q_val_factor
            sig_factor_all_dict[pert] = sig_factor
            if verbose:
                print('Finish: {}'.format(pert))
    with open(os.path.join(dataset_path, 'tau_factor_mean_all_dict.pkl'), 'wb') as fp:
        pickle.dump(tau_factor_mean_all_dict, fp)
    with open(os.path.join(dataset_path, 'tau_q_val_factor_all_dict.pkl'), 'wb') as fp:
        pickle.dump(tau_q_val_factor_all_dict, fp)
    with open(os.path.join(dataset_path, 'sig_factor_all_dict.pkl'), 'wb') as fp:
        pickle.dump(sig_factor_all_dict, fp)
    if verbose:
        print('Finish growing causal forests. See the results in folder:{}'.format(dataset_path))

################# Part4: Downstream analysis
def cal_clust_mean(adata, tau_factor_mean_all_dict, sig_factor_all_dict, aggregate_key, pert_key='condition'):
    """
    Calculate the average CATE and proportion of significant cells aggregated on cell subpopulation.
    Parameters
    ----------
    adata: AnnData
    The original dataset
    tau_factor_mean_all_dict: dictionary
    The dictionary containing CATE of all targets to all factors
    (output by CF_all_target_all_factor stored in dataset_name/Causal Forests)
    sig_factor_all_dict: dictionary
    The dictionary containing sig of all targets to all factors
    (output by CF_all_target_all_factor stored in dataset_name/Causal Forests)
    aggregate_key: string
    The column name of the labels that wish to aggregate in adata
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    Returns
    -------
    The average CATE and proportion of significant cells
    aggregated on cell subpopulation defined in 'aggregate_key' and all cells
    """
    tau_factor_mean_ctrl_dict = {}
    sig_factor_ctrl_dict = {}
    for pert in tau_factor_mean_all_dict.keys():
        subset = adata[adata.obs[pert_key].isin([pert, 'control'])]
        ctrl_idx = np.where(subset.obs[pert_key] == 'control')[0]
        tau_factor_mean_ctrl_dict[pert] = tau_factor_mean_all_dict[pert][ctrl_idx, :]
        sig_factor_ctrl_dict[pert] = sig_factor_all_dict[pert][ctrl_idx, :]

    ctrl_cells_cell_type = subset.obs.iloc[np.where(subset.obs[pert_key] == 'control')[0], :][aggregate_key]
    clust_mean_df = pd.DataFrame([])
    clust_sig_df = pd.DataFrame([])
    for target in tau_factor_mean_ctrl_dict.keys():
        tau_factor_mean_ctrl_clust_df = pd.DataFrame(tau_factor_mean_ctrl_dict[target])
        overall_df = pd.DataFrame(tau_factor_mean_ctrl_clust_df.mean(axis=0).values.reshape(1, -1))
        overall_df['cell_state'] = 'All'
        overall_df['target'] = target
        tau_factor_mean_ctrl_clust_df['cell_state'] = ctrl_cells_cell_type.values

        # group by cellstate
        clust_df = tau_factor_mean_ctrl_clust_df.groupby('cell_state').mean()
        clust_df['cell_state'] = clust_df.index
        clust_df['target'] = [target] * clust_df.shape[0]
        clust_mean_df = pd.concat([clust_mean_df, clust_df, overall_df], ignore_index=True)

        sig_factor_ctrl_clust_df = pd.DataFrame(sig_factor_ctrl_dict[target])
        overall_df = pd.DataFrame(sig_factor_ctrl_clust_df.mean(axis=0).values.reshape(1, -1))
        overall_df['cell_state'] = 'All'
        overall_df['target'] = target
        sig_factor_ctrl_clust_df['cell_state'] = ctrl_cells_cell_type.values

        sig_df = sig_factor_ctrl_clust_df.groupby('cell_state').mean()
        sig_df['cell_state'] = sig_df.index
        sig_df['target'] = [target] * sig_df.shape[0]
        clust_sig_df = pd.concat([clust_sig_df, sig_df, overall_df], ignore_index=True)
    return clust_mean_df, clust_sig_df


def cal_factor_level_similarity(adata, tau_factor_mean_all_dict, factor, cluster_name, aggregate_key,
                                pert_key='condition'):
    """
    Calculating the relationship between perturbations on a specific factor over certain subpopulation.
    Parameters
    ----------
    adata: AnnData
    The original dataset
    tau_factor_mean_all_dict: Dictionary
    The dictionary containing CATE of all targets to all factors.
    (output by CF_all_target_all_factor stored in dataset_name/Causal Forests)
    factor: int
    Index of factor
    cluster_name: string
    The name of the subpopulation label
    aggregate_key: string
    The column name of the labels that wish to aggregate in adata
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    Returns
    -------
    The pairwise distance matrix between perturbations
    """
    tau_factor_mean_ctrl_dict = {}
    for pert in tau_factor_mean_all_dict.keys():
        subset = adata[adata.obs[pert_key].isin([pert, 'control'])]
        ctrl_idx = np.where(subset.obs[pert_key] == 'control')[0]
        tau_factor_mean_ctrl_dict[pert] = tau_factor_mean_all_dict[pert][ctrl_idx, :]
    if cluster_name == 'All':
        pass
    else:
        ctrl_cells_cell_type = subset.obs.iloc[np.where(subset.obs[pert_key] == 'control')[0], :][aggregate_key]
    factor_mean_df = pd.DataFrame([])
    for target in tau_factor_mean_ctrl_dict.keys():
        factor_mean_df = pd.concat([factor_mean_df, pd.DataFrame(tau_factor_mean_ctrl_dict[target][:, factor])], axis=1)
    factor_mean_df.columns = tau_factor_mean_ctrl_dict.keys()
    if cluster_name == 'All':
        clust_factor_mean_df = factor_mean_df
    else:
        clust_factor_mean_df = factor_mean_df.iloc[np.where(ctrl_cells_cell_type == cluster_name)[0], :]

    similarity = pd.DataFrame(pairwise_distances(clust_factor_mean_df.T),
                              index=tau_factor_mean_ctrl_dict.keys(),
                              columns=tau_factor_mean_ctrl_dict.keys())
    return similarity


def cal_overall_similarity(adata, tau_factor_mean_all_dict, factor_weights, cluster_name, aggregate_key,
                           pert_key='condition'):
    """
    Calculating the overall relationship between perturbations over certain subpopulation.
    Parameters
    ----------
    adata: AnnData
    The original dataset
    tau_factor_mean_all_dict: Dictionary
    The dictionary containing CATE of all targets to all factors.
    (output by CF_all_target_all_factor stored in dataset_name/Causal Forests)
    factor_weights: list
    The weights of factors
    cluster_name: string
    The name of the subpopulation label
    aggregate_key: string
    The column name of the labels that wish to aggregate in adata
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    Returns
    -------
    The pairwise distance matrix between perturbations
    """
    tau_factor_mean_ctrl_dict = {}
    for pert in tau_factor_mean_all_dict.keys():
        subset = adata[adata.obs[pert_key].isin([pert, 'control'])]
        ctrl_idx = np.where(subset.obs[pert_key] == 'control')[0]
        tau_factor_mean_ctrl_dict[pert] = tau_factor_mean_all_dict[pert][ctrl_idx, :]
    if cluster_name == 'All':
        pass
    else:
        ctrl_cells_cell_type = subset.obs.iloc[np.where(subset.obs[pert_key] == 'control')[0], :][aggregate_key]
    num_factors = tau_factor_mean_ctrl_dict[pert].shape[1]

    overall_dis = np.zeros([len(tau_factor_mean_ctrl_dict.keys()), len(tau_factor_mean_ctrl_dict.keys())])
    for factor in range(num_factors):
        factor_mean_df = pd.DataFrame([])
        for target in tau_factor_mean_ctrl_dict.keys():
            factor_mean_df = pd.concat([factor_mean_df, pd.DataFrame(tau_factor_mean_ctrl_dict[target][:, factor])],
                                       axis=1)
        factor_mean_df.columns = tau_factor_mean_ctrl_dict.keys()
        if cluster_name == 'All':
            clust_factor_mean_df = factor_mean_df
        else:
            clust_factor_mean_df = factor_mean_df.iloc[np.where(ctrl_cells_cell_type == cluster_name)[0], :]
        factor_dis = pairwise_distances(clust_factor_mean_df.T)
        overall_dis = overall_dis + factor_weights[factor] * factor_dis

    overall_dis = pd.DataFrame(overall_dis,
                               index=tau_factor_mean_ctrl_dict.keys(),
                               columns=tau_factor_mean_ctrl_dict.keys())
    return overall_dis


def cal_factor_level_rank(clust_mean_df, cluster_name, absolute_val=False):
    """
    Calculating the rankings of perturbations on the factor-level over certain population.
    Parameters
    ----------
    clust_mean_df: DataFrame
    The average CATE aggregated on cell subpopulation of interest
    (output by 'cal_clust_mean')
    cluster_name: string
    The name of the subpopulation label
    absolute_val: bool
    Calculating the ranks based on absolute perturbation effects (True) or not (False)
    Returns
    -------
    A dictionary contains the rankings of perturbations on each factor,
    where the keys are the indices of factors.
    """
    factor_nums = clust_mean_df.shape[1] - 2
    mean_df = clust_mean_df[clust_mean_df['cell_state'] == cluster_name]
    target_names = np.array(mean_df['target'])
    # mean_df.index = mean_df['target']
    # mean_df = mean_df.drop(columns=['cell_state', 'target'])

    factor_level_rank_dict = {}
    for factor in range(factor_nums):
        if absolute_val:
            factor_level_rank_dict[str(factor)] = target_names[np.argsort(np.array(abs(mean_df.iloc[:, factor])))[::-1]]
        else:
            factor_level_rank_dict[str(factor)] = target_names[np.argsort(np.array((mean_df.iloc[:, factor])))[::-1]]
    return factor_level_rank_dict


def cal_overall_rank(clust_mean_df, cluster_name, factor_weights, absolute_val=False):
    """
    Calculating the overall rankings of perturbations over certain population.
    Parameters
    ----------
    clust_mean_df: DataFrame
    The average CATE aggregated on cell subpopulation of interest
    (output by 'cal_clust_mean')
    cluster_name: string
    The name of the subpopulation label
    factor_weights: list
    The weights of factors
    absolute_val: bool
    Calculating the ranks based on absolute perturbation effects (True) or not (False)
    Returns
    -------
    Overall rankings of perturbation effects.
    """
    factor_nums = clust_mean_df.shape[1] - 2
    mean_df = clust_mean_df[clust_mean_df['cell_state'] == cluster_name]
    target_names = np.array(mean_df['target'])

    if absolute_val:
        ranklist = target_names[np.argsort((abs(np.array(mean_df.iloc[:, 0:factor_nums]))
                                            @ (factor_weights.reshape(-1, 1)))[:, 0])[::-1]]
    else:
        ranklist = target_names[np.argsort(((np.array(mean_df.iloc[:, 0:factor_nums])) @
                                            (factor_weights.reshape(-1, 1)))[:, 0])[::-1]]
    return ranklist


################# Part5: Growing multi-armed causal forests

def cal_EPI(signed_sig_factor, sig_factor_epi):
    """
    Obtaining epistasis types.
    Parameters
    ----------
    signed_sig_factor: array
    An array ([cell_nums,3]) contains the signed sig (+1,-1 or 0) of tau (tau_a, tau_b and tau_int) of each cell
    sig_factor_epi: array
    An array ([cell_nums,2]) contains the sig (1 or 0) of epistasis test (epistasis a, b) of each cell
    Returns
    -------
    An array ([cell_nums,4]) contains the sig (1 or 0) of
    epistasis test (epistasis a, b, redundant and no effects) of each cell
    """
    EPI_res = np.zeros([sig_factor_epi.shape[0], 4])  # epiA, epiB, redundant, no effects
    for i in range(EPI_res.shape[0]):
        if (signed_sig_factor[i, 0] == 0) and (signed_sig_factor[i, 1] == 0) and (signed_sig_factor[i, 2] == 0):
            EPI_res[i, 3] = 1  # no effects
        else:
            if (sig_factor_epi[i, 0] == 0) and (sig_factor_epi[i, 1] == 0):
                EPI_res[i, 2] = 1  # redundant
            else:
                if (sig_factor_epi[i, 0] == 0):
                    EPI_res[i, 0] = 1  # epiA
                if (sig_factor_epi[i, 1] == 0):
                    EPI_res[i, 1] = 1  # epiB
    return EPI_res


def CF_GI_single_target_single_factor(dataset_name, target, factor, basal, treated, adata,
                                      pert_key='condition', n_estimators=1000, min_samples_leaf=5, verbose=True,
                                      random_state=0, one_side=False, alpha=0.05, return_ctrl=True, save=True):
    """
    Calculating genetic interactions between a pair of genes on one factor.
    Parameters
    ----------
    dataset_name: string
    The name of dataset to makedir (storing results)
    target: string
    Gene pair of interest
    factor: int
    Index of the factor of interest
    basal: AnnData
    Basal state
    treated: AnnData
    Factor-level expression
    adata: AnnData
    The original data
    pert_key: string
    The column name of the perturbation labels in AnnData, default  'condition'
    n_estimators: int
    Number of trees
    min_samples_leaf: int
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at least
    min_samples_leaf training samples in each of the left and right branches.
    verbose: bool
    Printing progress during running (True) or not (False)
    random_state: int
    Random seed to obtain a deterministic behaviour during fitting
    one_side: bool
    Conducting one-side marginal tests (True) or two-side marginal tests (False), default False
    alpha: float
    Significance level
    return_ctrl: bool
    Only return results on control cells (True) or not (False)
    save: bool
    Saving the results or not. If saving, the results will be stored in dataset_name/CausalForests
    Returns
    -------
    A dataframe ([cell_nums, 21]) containing all the inference results of each cell.
    """
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)  # makedir
    dataset_path = os.path.join(dataset_name, 'CausalForests')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)  # makedir to store causal forests results

    t1 = target.split('+')[0]
    t2 = target.split('+')[1]
    X = basal[basal.obs.condition.isin([t1, t2, target, 'control'])].X
    Y = treated[treated.obs.condition.isin([t1, t2, target, 'control'])].X[:, factor]
    labels_list = (adata[adata.obs[pert_key].isin([t1, t2, target, 'control'])].obs[pert_key]).tolist()
    ctrl_idx = np.where(np.array(labels_list) == 'control')[0]
    T = []
    for label in labels_list:
        if label == t1:
            T.append([1, 0, 0])
        if label == t2:
            T.append([0, 1, 0])
        if label == target:
            T.append([1, 1, 1])
        if label == 'control':
            T.append([0, 0, 0])
    T = np.array(T)
    # set parameters for causal forest
    causal_forest = CausalForest(criterion='het',
                                 n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=random_state
                                 )
    # fit train data to causal forest model
    causal_forest.fit(X, T, Y)

    # use causal forest model to estimate treatment effects
    theta, var = causal_forest.predict_and_var(X)

    tau_factor_mean = theta
    tau_factor_sd = []
    for i in range(var.shape[0]):
        tau_factor_sd.append(np.sqrt(np.diag(var[i])))
    tau_factor_sd = np.array(tau_factor_sd)

    tau_p_val_factor = np.zeros_like(tau_factor_mean)
    tau_q_val_factor = np.zeros_like(tau_factor_mean)
    sig_factor = np.zeros_like(tau_factor_mean)

    for i in range(tau_factor_mean.shape[1]):
        tau_p_val_factor[:, i] = 2 * (
            norm.sf(x=abs(tau_factor_mean[:, i]) / (tau_factor_sd[:, i] + 1e-6), loc=0, scale=1))
        if one_side:
            tau_p_val_factor[:, i] = tau_p_val_factor[:, i] / 2
        tau_q_val_factor[:, i] = fdrcorrection(tau_p_val_factor[:, i])[1]
        sig_factor[:, i] = (tau_q_val_factor[:, i] <= alpha).astype(int)
    signed_sig_factor = (np.sign(tau_factor_mean) * sig_factor).astype(int)

    tau_factor_epi_A_mean = (tau_factor_mean[:, 1] + tau_factor_mean[:, 2]).reshape(-1, 1)
    tau_factor_epi_B_mean = (tau_factor_mean[:, 0] + tau_factor_mean[:, 2]).reshape(-1, 1)
    tau_factor_epi_A_sd = []
    tau_factor_epi_B_sd = []
    for i in range(var.shape[0]):
        varA = max(var[i, 1, 1] + var[i, 2, 2] + var[i, 1, 2] + var[i, 2, 1], 0)
        varB = max(var[i, 0, 0] + var[i, 2, 2] + var[i, 0, 2] + var[i, 2, 0], 0)
        tau_factor_epi_A_sd.append(np.sqrt(varA))
        tau_factor_epi_B_sd.append(np.sqrt(varB))
    tau_factor_epi_A_sd = (np.array(tau_factor_epi_A_sd)).reshape(-1, 1)
    tau_factor_epi_B_sd = (np.array(tau_factor_epi_B_sd)).reshape(-1, 1)

    tau_p_val_factor_epi_A = 2 * norm.sf(x=abs(tau_factor_epi_A_mean[:, 0]) / (tau_factor_epi_A_sd[:, 0] + 1e-6), loc=0,
                                         scale=1)
    tau_q_val_factor_epi_A = fdrcorrection(tau_p_val_factor_epi_A)[1]
    sig_factor_epi_A = (tau_q_val_factor_epi_A <= alpha).astype(int)

    tau_p_val_factor_epi_B = 2 * norm.sf(x=abs(tau_factor_epi_B_mean[:, 0]) / (tau_factor_epi_B_sd[:, 0] + 1e-6), loc=0,
                                         scale=1)
    tau_q_val_factor_epi_B = fdrcorrection(tau_p_val_factor_epi_B)[1]
    sig_factor_epi_B = (tau_q_val_factor_epi_B <= alpha).astype(int)

    pvals_res = np.ones([signed_sig_factor.shape[0], 3])  # [SY,BF,DM]

    # set1
    test_cells_1 = list(
        np.intersect1d(np.where(signed_sig_factor[:, 0] == 1)[0], np.where(signed_sig_factor[:, 1] == 1)[0]))
    test_cells_2 = list(
        np.intersect1d(np.where(signed_sig_factor[:, 0] == 1)[0], np.where(signed_sig_factor[:, 1] == 0)[0]))
    test_cells_3 = list(
        np.intersect1d(np.where(signed_sig_factor[:, 0] == 0)[0], np.where(signed_sig_factor[:, 1] == 1)[0]))
    test_cells = test_cells_1 + test_cells_2 + test_cells_3
    if len(test_cells) != 0:
        for i in test_cells:
            if tau_factor_mean[i, 2] >= 0:
                pvals_res[i, 1] = 1 - (tau_q_val_factor[i, 2] / 2)  # H0:>0,H1:<0, BF
                pvals_res[i, 0] = tau_q_val_factor[i, 2] / 2  # H0:<0,H1:>0, SY
            else:
                pvals_res[i, 1] = tau_q_val_factor[i, 2] / 2
                pvals_res[i, 0] = 1 - (tau_q_val_factor[i, 2] / 2)

    # set2
    test_cells_1 = list(
        np.intersect1d(np.where(signed_sig_factor[:, 0] == -1)[0], np.where(signed_sig_factor[:, 1] == -1)[0]))
    test_cells_2 = list(
        np.intersect1d(np.where(signed_sig_factor[:, 0] == -1)[0], np.where(signed_sig_factor[:, 1] == 0)[0]))
    test_cells_3 = list(
        np.intersect1d(np.where(signed_sig_factor[:, 0] == 0)[0], np.where(signed_sig_factor[:, 1] == -1)[0]))
    test_cells = test_cells_1 + test_cells_2 + test_cells_3
    if len(test_cells) != 0:
        for i in test_cells:
            if tau_factor_mean[i, 2] >= 0:
                pvals_res[i, 0] = 1 - (tau_q_val_factor[i, 2] / 2)  # H0:>0,H1:<0, SY
                pvals_res[i, 1] = tau_q_val_factor[i, 2] / 2  # H0:<0,H1:>0, BF
            else:
                pvals_res[i, 0] = tau_q_val_factor[i, 2] / 2
                pvals_res[i, 1] = 1 - (tau_q_val_factor[i, 2] / 2)

    # set3
    test_cells = list(np.where((signed_sig_factor[:, 0] * signed_sig_factor[:, 1]) == -1)[0])
    if len(test_cells) != 0:
        pvals_res[test_cells, 2] = tau_q_val_factor[test_cells, 2]

    test_sig_factor = np.zeros_like(pvals_res)
    for i in range(pvals_res.shape[1]):
        test_sig_factor[:, i] = (pvals_res[:, i] <= alpha).astype(int)

    sig_factor_epi = (np.array([sig_factor_epi_A, sig_factor_epi_B])).T
    EPI_res = cal_EPI(signed_sig_factor, sig_factor_epi)

    if verbose:
        print('Finish GI calculation in target={} factor={}'.format(target, factor))

    all_res = np.zeros([tau_factor_mean.shape[0], 21])
    all_res[:, 0:3] = tau_factor_mean
    all_res[:, 3:6] = tau_q_val_factor
    all_res[:, 6:9] = signed_sig_factor
    all_res[:, 9:12] = pvals_res
    all_res[:, 12] = tau_q_val_factor_epi_A
    all_res[:, 13] = tau_q_val_factor_epi_B
    all_res[:, 14:17] = test_sig_factor
    all_res[:, 17:21] = EPI_res
    all_res_df = pd.DataFrame(all_res, columns=['tau_a', 'tau_b', 'tau_int', 'pval_a', 'pval_b', 'pval_int',
                                                'sign_sig_a', 'sign_sig_b', 'sign_sig_int', 'pval_SY', 'pval_BF',
                                                'pval_DM',
                                                'pval_epia', 'pval_epib', 'sig_SY', 'sig_BF', 'sig_DM', 'sig_epia',
                                                'sig_epib',
                                                'sig_RD', 'sig_NE'])
    if return_ctrl:
        all_res_ctrl = all_res[ctrl_idx, :]
        all_res_ctrl_df = pd.DataFrame(all_res_ctrl,
                                       columns=['tau_a', 'tau_b', 'tau_int', 'pval_a', 'pval_b', 'pval_int',
                                                'sign_sig_a', 'sign_sig_b', 'sign_sig_int', 'pval_SY', 'pval_BF',
                                                'pval_DM',
                                                'pval_epia', 'pval_epib', 'sig_SY', 'sig_BF', 'sig_DM', 'sig_epia',
                                                'sig_epib',
                                                'sig_RD', 'sig_NE'])
        if save:
            np.save(
                os.path.join(dataset_name, 'CausalForests', 'GI_res_target={}_factor={}.npy'.format(target, factor)),
                all_res_ctrl)
        return all_res_ctrl_df
    else:
        if save:
            np.save(
                os.path.join(dataset_name, 'CausalForests', 'GI_res_target={}_factor={}.npy'.format(target, factor)),
                all_res)
        return all_res_df


def acat_test(pvalues, weights=None):
    """
    Aggregated Cauchy Association Test
    A p-value combination method using the Cauchy distribution.
    Revised from: https://gist.github.com/ryananeff/c66cdf086979b13e855f2c3d0f3e54e1
    Parameters
    ----------
    pvalues: list
    The p-values aim to combine.
    weights: list
    The weights for each of the p-values. If None, equal weights are used.
    Returns
    -------
    The ACAT combined p-value
    """
    if any(np.isnan(pvalues)):
        raise Exception("Cannot have NAs in the p-values.")
    if any([(i > 1) | (i < 0) for i in pvalues]):
        raise Exception("P-values must be between 0 and 1.")
    if any([i == 1 for i in pvalues]) & any([i == 0 for i in pvalues]):
        raise Exception("Cannot have both 0 and 1 p-values.")
    if any([i == 0 for i in pvalues]):
        print("Warn: p-values are exactly 0.")
        return 0
    if any([i == 1 for i in pvalues]):
        print("Warn: p-values are exactly 1.")
        return 1
    if weights == None:
        weights = [1 / len(pvalues) for i in pvalues]
    elif len(weights) != len(pvalues):
        raise Exception("Length of weights and p-values differs.")
    elif any([i < 0 for i in weights]):
        raise Exception("All weights must be positive.")
    else:
        weights = [i / len(weights) for i in weights]

    pvalues = np.array(pvalues)
    weights = np.array(weights)

    if any([i < 1e-16 for i in pvalues]) == False:
        cct_stat = sum(weights * np.tan((0.5 - pvalues) * np.pi))
    else:
        is_small = [i < (1e-16) for i in pvalues]
        is_large = [i >= (1e-16) for i in pvalues]
        cct_stat = sum((weights[is_small] / pvalues[is_small]) / np.pi)
        cct_stat += sum(weights[is_large] * np.tan((0.5 - pvalues[is_large]) * np.pi))

    if cct_stat > 1e15:
        pval = (1 / cct_stat) / np.pi
    else:
        pval = 1 - sp.stats.cauchy.cdf(cct_stat)

    return pval


def cal_ACAT_pvals(dataset_name, target, factor_nums,
                   alpha=0.05, weights=None, return_res=True, save=True, verbose=True):
    """
    Combining p-values to infer GIs at the transciptome level.
    Parameters
    ----------
    dataset_name: string
    The name of dataset to makedir (storing results)
    target: string
    Gene pair of interest
    factor_nums: int
    The number of factors
    alpha: float
    Significance level
    weights: list
    The weights of factors
    return_res: bool
    Return results or not
    save: bool
    Saving results or not. If saving, the results will be stored in dataset_name/CausalForests
    verbose: bool
    Printing progress during running or not
    Returns
    -------
    A dataframe ([cell_nums, 11]) containing all results of each cell.
    """
    if weights is not None:
        if len(weights) != factor_nums:
            raise Exception("The length of weight list does not match factor_nums.")
    SYBFDM_qvals_all_factors = []
    epi_qvals_all_factors = []
    for factor in range(factor_nums):
        res = np.load(
            os.path.join(dataset_name, 'CausalForests', 'GI_res_target={}_factor={}.npy'.format(target, factor)))
        SYBFDM_qvals = res[:, [9, 10, 11]]
        epi_qvals = res[:, [12, 13]]
        SYBFDM_qvals_all_factors.append(SYBFDM_qvals)
        epi_qvals_all_factors.append(epi_qvals)
    SYBFDM_qvals_all_factors = np.transpose(np.array(SYBFDM_qvals_all_factors), (1, 0, 2))
    epi_qvals_all_factors = np.transpose(np.array(epi_qvals_all_factors), (1, 0, 2))

    SYBFDM_qvals_all_factors_res = np.zeros([SYBFDM_qvals_all_factors.shape[0], SYBFDM_qvals_all_factors.shape[2]])
    epi_qvals_all_factors_res = np.zeros([epi_qvals_all_factors.shape[0], epi_qvals_all_factors.shape[2]])

    for cell in range(SYBFDM_qvals_all_factors_res.shape[0]):
        for col in range(SYBFDM_qvals_all_factors_res.shape[1]):
            pvalues = list(SYBFDM_qvals_all_factors[cell, :, col])
            pvalues = [1 / (SYBFDM_qvals_all_factors_res.shape[0] * SYBFDM_qvals_all_factors_res.shape[1])
                       if pval == 0 else pval for pval in pvalues]
            pvalues = [1 - 1 / (SYBFDM_qvals_all_factors_res.shape[0] * SYBFDM_qvals_all_factors_res.shape[1])
                       if pval == 1 else pval for pval in pvalues]
            SYBFDM_qvals_all_factors_res[cell, col] = acat_test(pvalues=pvalues, weights=weights)
        for col2 in range(epi_qvals_all_factors_res.shape[1]):
            pvalues = list(epi_qvals_all_factors[cell, :, col2])
            pvalues = [1 / (epi_qvals_all_factors_res.shape[0] * epi_qvals_all_factors_res.shape[1])
                       if pval == 0 else pval for pval in pvalues]
            pvalues = [1 - 1 / (epi_qvals_all_factors_res.shape[0] * epi_qvals_all_factors_res.shape[1])
                       if pval == 1 else pval for pval in pvalues]
            epi_qvals_all_factors_res[cell, col2] = acat_test(pvalues=pvalues, weights=weights)

    SYBFDM_sig_all_factors_res = (SYBFDM_qvals_all_factors_res <= alpha).astype(int)
    epi_sig_all_factors_res = (epi_qvals_all_factors_res <= alpha).astype(int)

    EPI_res = np.zeros([epi_sig_all_factors_res.shape[0], 3])  # epiA, epiB, redundant
    for i in range(EPI_res.shape[0]):
        if (epi_sig_all_factors_res[i, 0] == 0) and (epi_sig_all_factors_res[i, 1] == 0):
            EPI_res[i, 2] = 1  # redundant
        else:
            if (epi_sig_all_factors_res[i, 0] == 0):
                EPI_res[i, 0] = 1  # epiA
            if (epi_sig_all_factors_res[i, 1] == 0):
                EPI_res[i, 1] = 1  # epiB
    if verbose:
        print('Finish ACAT calculation in target={}'.format(target))

    all_res = np.zeros([SYBFDM_qvals_all_factors_res.shape[0], 11])
    all_res[:, 0:3] = SYBFDM_qvals_all_factors_res
    all_res[:, 3:5] = epi_qvals_all_factors_res
    all_res[:, 5:8] = SYBFDM_sig_all_factors_res
    all_res[:, 8:11] = EPI_res
    all_res_df = pd.DataFrame(all_res, columns=['pval_SY', 'pval_BF', 'pval_DM', 'pval_epia', 'pval_epib',
                                                'sig_SY', 'sig_BF', 'sig_DM', 'sig_epia', 'sig_epib', 'sig_RD'])
    if save:
        np.save(os.path.join(dataset_name, 'CausalForests', 'GI_res_target={}_ACAT.npy'.format(target)), all_res)
    if return_res:
        return all_res_df
