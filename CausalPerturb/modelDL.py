# -*- coding:utf-8 -*-
"""
name:DL Model
function:Autoencoder model
"""

import torch

class MLP(torch.nn.Module):
    """
    A fully connected neural network with ReLU activations and BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]
        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CAPE(torch.nn.Module):
    """
    the CAPE autoencoder
    """

    def __init__(
            self,
            num_genes,
            num_perts,
            pos_weight,
            W_ini,
            device="cpu",
            seed=0,
            patience=5,
            loss_ae="mse",
            hparams=None,
            lambda_ort=0.5
    ):
        super(CAPE, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_perts = num_perts
        self.pos_weight = pos_weight
        self.num_factors = W_ini.shape[0]
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.lambda_ort=lambda_ort

        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        self.relu = torch.nn.ReLU()
        self.set_hparams_(hparams)

        self.encoder = MLP(
            [num_genes]
            + self.hparams["autoencoder_dim"]
            + [self.hparams["dim"]]
        )
        self.trcoder = MLP(
            [self.hparams["dim"]+self.num_perts]
            + [self.hparams["dim"]] * self.hparams["trcoder_depth"]
            + [self.num_factors]
        )

        self.adversary_perts = MLP(
            [self.hparams["dim"]]
            + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
            + [num_perts]
        )
        self.loss_adversary_perts = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.loss_autoencoder = torch.nn.MSELoss(reduction="mean")

        self.W = []
        self.W.append(W_ini.requires_grad_())

        self.iteration = 0
        self.to(self.device)

        # optimizers
        has_perts = self.num_perts > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []

        _parameters = (get_params(self.encoder, True)
                       + get_params(self.trcoder,True)+self.W)
        self.optimizer_autoencoder = torch.optim.Adam(_parameters,
                                                      lr=self.hparams["autoencoder_lr"],
                                                      weight_decay=self.hparams["autoencoder_wd"])
        _parameters = get_params(self.adversary_perts, has_perts)

        self.optimizer_adversaries = torch.optim.Adam(_parameters,
                                                      lr=self.hparams["adversary_lr"],
                                                      weight_decay=self.hparams["adversary_wd"])
        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])
        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, hparams):
        """
            Set hyperparameters to default values or values specified by the dictionary.
        """

        self.hparams = {
            "dim": 20,
            "trcoder_depth": 3,
            "autoencoder_dim":[512,128,64],
            "decoder_dim":[64,128],
            "adversary_width": 64,
            "adversary_depth": 2,
            "autoencoder_lr": 3e-4,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "adversary_wd": 4e-7,
            "adversary_steps": 3,
            "batch_size": 256,
            "step_size_lr": 45,
            "penalty_adversary":10
        }
        if hparams is not None:
            for key in hparams.keys():
                self.hparams[key]=hparams[key]
        return self.hparams

    # set hyperparameters

    def move_inputs_(self, genes, perts, cell_states=None):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            if perts is not None:
                perts = perts.to(self.device)
                if cell_states is not None:
                    cell_states = cell_states.to(self.device)
        return genes, perts, cell_states

    def compute_pert_embeddings_(self, z, perts):
        """
        Simulating the perturbations
        """
        treated_basal=self.trcoder(torch.cat((z, perts), dim=1))
        return treated_basal

    def mse_loss(self, weights):
        loss = torch.nn.MSELoss(reduction="mean")
        return loss(weights, torch.eye(weights.size()[0]))

    def forward(
            self,
            genes,
            perts,
            return_latent_basal=False,
            return_latent_treated=False,
    ):

        genes, perts, _ = self.move_inputs_(genes, perts)

        latent_basal = self.encoder(genes)

        if self.num_perts > 0:
            latent_treated = self.compute_pert_embeddings_(z=latent_basal, perts=perts)
        gene_reconstructions_mean = latent_treated @ self.relu(self.W[0])

        gene_reconstructions = gene_reconstructions_mean

        if return_latent_basal:
            if return_latent_treated:
                return gene_reconstructions, latent_basal, latent_treated
            else:
                return gene_reconstructions, latent_basal
        if return_latent_treated:
            return gene_reconstructions, latent_treated
        return gene_reconstructions

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def compute_gradients(self, output, input):
        grads = torch.autograd.grad(output, input, create_graph=True)
        grads = grads[0].pow(2).mean()
        return grads

    def update(self, genes, perts, cell_states, lambda_val=None):
        """
        Update parameters in the autoencoder.
        """
        genes, perts, cell_states = self.move_inputs_(genes, perts, cell_states)
        gene_reconstructions, latent_basal = self.forward(
            genes,
            perts,
            return_latent_basal=True,
        )

        gene_means = gene_reconstructions
        reconstruction_loss = self.loss_autoencoder(gene_means, genes)
        adversary_perts_loss = torch.tensor([0.0], device=self.device)
        l2loss = self.mse_loss(torch.matmul(self.relu(self.W[0]), self.relu(self.W[0]).t()))

        if self.num_perts > 0:
            adversary_perts_predictions = self.adversary_perts(latent_basal)
            adversary_perts_loss = self.loss_adversary_perts(adversary_perts_predictions, perts.gt(0).float())

        if self.iteration % self.hparams["adversary_steps"]:
            if self.num_perts > 0:
                adversary_perts_penalty = self.compute_gradients(
                    adversary_perts_predictions.sum(), latent_basal
                )
                self.optimizer_adversaries.zero_grad()
                (adversary_perts_loss + self.hparams["penalty_adversary"] * adversary_perts_penalty).backward()
                self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            (reconstruction_loss - lambda_val * adversary_perts_loss +
             self.lambda_ort*l2loss).backward()
            self.optimizer_autoencoder.step()
        self.iteration += 1
        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_perts": adversary_perts_loss.item(),
            "l2loss": l2loss.item()
        }