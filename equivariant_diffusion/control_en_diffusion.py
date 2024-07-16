from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from global_registry import PARAM_REGISTRY
from equivariant_diffusion.en_diffusion import *


class ControlEnLatentDiffusion(EnLatentDiffusion):
    """
    The E(n) Latent Diffusion Module.
    """
    def __init__(self, **kwargs):
        vae                     = kwargs.get('vae')
        dynamics                = kwargs.get('dynamics')
        trainable_ae_encoder    = kwargs.get('trainable_ae_encoder')
        trainable_ae_decoder    = kwargs.get('trainable_ae_decoder')
        trainable_ldm           = kwargs.pop('trainable_ldm')           # pop away, not required in Enlatentdiffusion.__init__()
        trainable_controlnet    = kwargs.pop('trainable_controlnet')
        trainable_fusion_blocks = kwargs.pop('trainable_fusion_blocks')
        
        super().__init__(**kwargs)
        
        assert isinstance(vae, EnHierarchicalVAE), f"required VAE class of EnHierarchicalVAE but {vae.__class__.__name__} given"
        assert isinstance(dynamics, models.ControlNet_Module_Wrapper), f"required controlled_ldm class of ControlNet_Module_Wrapper but {dynamics} given"

        # Create self.vae as the first stage model.
        self.trainable_ae_encoder = trainable_ae_encoder
        self.trainable_ae_decoder = trainable_ae_decoder
        self.trainable_ldm = trainable_ldm
        self.trainable_controlnet = trainable_controlnet
        self.trainable_fusion_blocks = trainable_fusion_blocks
        
        self.instantiate_first_second_stage(vae, dynamics)
    
    # def unnormalize_z(self, z, node_mask):
    #     # Overwrite the unnormalize_z function to do nothing (for sample_chain). 

    #     # Parse from z
    #     x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
    #     h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
    #     assert h_int.size(2) == self.include_charges

    #     # Unnormalize
    #     # x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
    #     output = torch.cat([x, h_cat, h_int], dim=2)
    #     return output
    
    def log_constants_p_h_given_z0(self, h, node_mask):
        """Computes p(h|z0)."""
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_h = n_nodes * self.n_dims

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_h * (- log_sigma_x - 0.5 * np.log(2 * np.pi))



    # ~!fp16
    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):

        # Computes the error for the distribution N(latent | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_pxh_given_z_without_constants = -0.5 * self.compute_error(net_out, gamma_0, eps)

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_pxh_given_z_without_constants

        return log_p_xh_given_z
    
    def forward(self, x1, h1, x2, h2, node_mask_1=None, node_mask_2=None, edge_mask_1=None, edge_mask_2=None, joint_edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """

        """ VAE Encoding """
        # Encode data to latent space.
        z_x_mu_1, z_x_sigma_1, z_h_mu_1, z_h_sigma_1 = self.vae.encode(x1, h1, node_mask_1, edge_mask_1, context)
        z_x_mu_2, z_x_sigma_2, z_h_mu_2, z_h_sigma_2 = self.vae.encode(x2, h2, node_mask_2, edge_mask_2, context)

        # ~!fp16
        # self.gamma = self.gamma.to(x.device)

        # Compute fixed sigma values for VAE.
        assert x1.size(0) == x2.size(0), f"Different batch_size encountered: x1={x1.size()} x2={x2.size()}"
        t_zeros = torch.zeros(size=(x1.size(0), 1), device=x1.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x1)
        sigma_0 = self.sigma(gamma_0, x1)

        # Infer latent z.
        z_xh_mean_1 = torch.cat([z_x_mu_1, z_h_mu_1], dim=2)
        z_xh_mean_2 = torch.cat([z_x_mu_2, z_h_mu_2], dim=2)
        
        # tmp
        # z_xh_mean = z_xh_mean * node_mask
        
        diffusion_utils.assert_correctly_masked(z_xh_mean_1, node_mask_1)
        diffusion_utils.assert_correctly_masked(z_xh_mean_2, node_mask_2)
        
        z_xh_sigma = sigma_0  # time-controlled VAE std, same for both ligand pocket
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        
        # Add noise to ligand only
        z_xh_1 = self.vae.sample_normal(z_xh_mean_1, z_xh_sigma, node_mask_1)  # z_xh = mu + sigma * eps
        if not self.trainable_ae_encoder:
            z_xh_1 = z_xh_1.detach()  # Always keep the VAE's Encoder fixed when training LDM and/or VAE's Decoder
            z_xh_mean_2 = z_xh_mean_2.detach()
        
        diffusion_utils.assert_correctly_masked(z_xh_1, node_mask_1)

        """ VAE Decoding - required if training VAE too """
        # Compute reconstruction loss - Ligand only
        if self.trainable_ae_decoder:
            # ground truth
            xh_1 = torch.cat([x1, h1['categorical'], h1['integer']], dim=2)
            # Decoder output (reconstruction).
            x_recon_1, h_recon_1 = self.vae.decoder._forward(z_xh_1, node_mask_1, edge_mask_1, context)
            xh_rec_1 = torch.cat([x_recon_1, h_recon_1], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec_1, xh_1)

        else:
            loss_recon = 0

        
        """ VAE Encoded features for LDM """
        # LDM
        assert z_xh_1.size(0) == z_xh_mean_2.size(0), f"Different batch_size encountered! z_xh_1={z_xh_1.size()} z_xh_mean_2={z_xh_mean_2.size()}"
        assert z_xh_1.size(2) == z_xh_mean_2.size(2), f"Different num embedding encountered! z_xh_1={z_xh_1.size()} z_xh_mean_2={z_xh_mean_2.size()}"

        z_x_1 = z_xh_1[:, :, :self.n_dims]
        z_h_1 = z_xh_1[:, :, self.n_dims:]
        z_x_2 = z_xh_mean_2[:, :, :self.n_dims]
        z_h_2 = z_xh_mean_2[:, :, self.n_dims:]
        diffusion_utils.assert_mean_zero_with_mask(z_x_1, node_mask_1)
        diffusion_utils.assert_mean_zero_with_mask(z_x_2, node_mask_2)
        
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h_1 = {'categorical': torch.zeros(0).to(z_h_1), 'integer': z_h_1}
        z_h_2 = {'categorical': torch.zeros(0).to(z_h_2), 'integer': z_h_2}

        # compute_loss() defined in EnVariationalDiffusion Above
        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss_ld, loss_dict = self.compute_loss(z_x_1, z_h_1, z_x_2, z_h_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss_ld, loss_dict = self.compute_loss(z_x_1, z_h_1, z_x_2, z_h_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, t0_always=True)
        
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_h_given_z0(
            torch.cat([h1['categorical'], h1['integer']], dim=2), node_mask_1)
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        neg_log_pxh = loss_ld + loss_recon + neg_log_constants

        return neg_log_pxh   # negatve log likelihood



    def compute_loss(self, x1, h1, x2, h2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        # False when training, true others
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        # ~!fp16
        assert x1.size(0) == x2.size(0), f"Different batch_size encountered: x1={x1.size()} x2={x2.size()}"
        t_int = torch.randint(lowest_t, self.T + 1, size=(x1.size(0), 1), device=x1.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x1)
        gamma_t = self.inflate_batch_array(self.gamma(t), x1)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x1)
        sigma_t = self.sigma(gamma_t, x1)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        # z_x = utils.sample_center_gravity_zero_gaussian_with_mask(..)
        # z_h = utils.sample_gaussian_with_mask(..)
        # z = torch.cat([z_x, z_h], dim=2)
        eps = self.sample_combined_position_feature_noise(n_samples=x1.size(0), n_nodes=x1.size(1), node_mask=node_mask_1)

        # Concatenate x, h[integer] and h[categorical].
        xh1 = torch.cat([x1, h1['categorical'], h1['integer']], dim=2)
        xh2 = torch.cat([x2, h2['categorical'], h2['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_1 = alpha_t * xh1 + sigma_t * eps
        z_t_2 = xh2

        diffusion_utils.assert_mean_zero_with_mask(z_t_1[:, :, :self.n_dims], node_mask_1)

        # Neural net prediction.
        net_out = self.phi(t, z_t_1, z_t_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x1, node_mask_1)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh1, node_mask_1)

        # Combining the terms
        # false when training, true others
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            assert x1.size(0) == x2.size(0), f"Different batch_size encountered: x1={x1.size()} x2={x2.size()}"
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x1)
            alpha_0 = self.alpha(gamma_0, x1)
            sigma_0 = self.sigma(gamma_0, x1)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x1.size(0), n_nodes=x1.size(1), node_mask=node_mask_1)
            z_0 = alpha_0 * xh1 + sigma_0 * eps_0

            net_out = self.phi(t_zeros, z_0, z_t_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x1, h1, z_0, gamma_0, eps_0, net_out, node_mask_1)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x1, h1, z_t_1, gamma_t, eps, net_out, node_mask_1)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, \
               {
                't': t_int.squeeze(),         # sampled random timesteps
                'loss_t': loss.squeeze(),     # final loss combined
                'error': error.squeeze()      # sum_except_batch((eps - eps_t) ** 2) / denom
                }


    def phi(self, t, xh1, xh2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context):
        net_out = self.dynamics._forward(t, xh1, xh2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context)

        return net_out


    @torch.no_grad()
    def sample(self, n_samples, n_nodes, x2, h2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        NOTE: full timesteps T.
        """
        
        """ VAE Encoding """
        # Encode data to latent space.
        x2 = diffusion_utils.remove_mean_with_mask(x2, node_mask_2)
        z_x_mu_2, z_x_sigma_2, z_h_mu_2, z_h_sigma_2 = self.vae.encode(x2, h2, node_mask_2, edge_mask_2, context)

        # Infer latent z.
        z_xh_mean_2 = torch.cat([z_x_mu_2, z_h_mu_2], dim=2)

        z_x_2 = z_xh_mean_2[:, :, :self.n_dims]
        z_h_2 = z_xh_mean_2[:, :, self.n_dims:]
        diffusion_utils.assert_mean_zero_with_mask(z_x_2, node_mask_2)
        
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h_2 = {'categorical': torch.zeros(0).to(z_h_2), 'integer': z_h_2}
        xh2 = torch.cat([z_x_2, z_h_2['categorical'], z_h_2['integer']], dim=2)  # from compute_loss, next line should be self.phi()
        zt_2 = xh2


        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask_1)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask_1)

        assert z.size(0) == zt_2.size(0), f"Different batch_size encountered! z={z.size()} zt_2={zt_2.size()}"
        assert z.size(2) == zt_2.size(2), f"Different num embedding encountered! z={z.size()}, zt_2={zt_2.size()}"

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask_1)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            # from T -> t=1
            z = self.sample_p_zs_given_zt(s_array, t_array, z, zt_2.clone(), node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, fix_noise=fix_noise)

        # Final sample z0, t=0
        # Finally sample p(x, h | z_0).
        z_x, z_h = self.sample_p_xh_given_z0(z, zt_2.clone(), node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask_1)

        # remove velocity
        max_cog = torch.sum(z_x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            z_x = diffusion_utils.remove_mean_with_mask(z_x, node_mask_1)

        z_xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh, node_mask_1)
        x, h = self.vae.decode(z_xh, node_mask_1, edge_mask_1, context)

        return x, h



    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        raise NotImplementedError()
        # """
        # Draw samples from the generative model, keep the intermediate states for visualization purposes.
        # """
        # chain_flat = super().sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames)

        # # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # # chain[0] = xh  # Overwrite last frame with the resulting x and h.

        # # chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        # chain = chain_flat.view(keep_frames, n_samples, *chain_flat.size()[1:])
        # chain_decoded = torch.zeros(
        #     size=(*chain.size()[:-1], self.vae.in_node_nf + self.vae.n_dims), device=chain.device)

        # for i in range(keep_frames):
        #     z_xh = chain[i]
        #     diffusion_utils.assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

        #     x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)
        #     xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        #     chain_decoded[i] = xh
        
        # chain_decoded_flat = chain_decoded.view(n_samples * keep_frames, *chain_decoded.size()[2:])

        # return chain_decoded_flat



    def sample_p_zs_given_zt(self, s, t, zt_1, zt_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, fix_noise=False):
        """
        Samples from zs ~ p(zs | zt). Only used during sampling. 
        NOTE: One sampling step. (NOT final step z0)
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_1)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_1)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_1)

        # Neural net prediction.
        eps_t = self.phi(t, zt_1, zt_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt_1[:, :, :self.n_dims], node_mask_1)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask_1)
        mu = zt_1 / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        # z_x = utils.sample_center_gravity_zero_gaussian_with_mask(..)
        # z_h = utils.sample_gaussian_with_mask(..)
        # eps = torch.cat([z_x, z_h], dim=2)
        # zs = mu + sigma * eps
        zs = self.sample_normal(mu, sigma, node_mask_1, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask_1),
             zs[:, :, self.n_dims:]], dim=2
        )
        return zs



    def sample_p_xh_given_z0(self, z0, zt_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(zeros, z0, zt_2, node_mask_1, node_mask_2, edge_mask_1, edge_mask_2, joint_edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        z_xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask_1, fix_noise=fix_noise)

        z_x = z_xh[:, :, :self.n_dims]

        # h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        # x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask

        # Make the data structure compatible with the EnVariationalDiffusion sample() and sample_chain().
        z_h = {'integer': z_xh[:, :, self.n_dims:], 'categorical': torch.zeros(0).to(z_xh)}
        
        return z_x, z_h


    def instantiate_first_second_stage(self, vae: EnHierarchicalVAE, dynamics: models.ControlNet_Module_Wrapper):
        # VAE
        self.vae = vae
        if not self.trainable_ae_encoder and not self.trainable_ae_decoder:
            self.vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
            print(">>> [VAE] (Encoder) (Decoder) requires_grad = False")
        else:
            # set whole model to trainable, but if self.trainable_ae_encoder=False,
            # will detach the VAE Encoder's outputs from the loss computational graph
            # hence weights not updated.

            # update: setting requires_grad part by part - more secure
            self.vae.train()
            # Encoder
            for param in self.vae.encoder.parameters():
                if self.trainable_ae_encoder:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f">>> [VAE] (Encoder) requires_grad = {self.trainable_ae_encoder}")
            # Decoder
            for param in self.vae.decoder.parameters():
                if self.trainable_ae_decoder:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f">>> [VAE] (Decoder) requires_grad = {self.trainable_ae_decoder}")

        # Controlled Diffusion Model
        self.dynamics = dynamics
        if (not self.trainable_ldm) and (not self.trainable_controlnet) and (not self.trainable_fusion_blocks):
            self.dynamics.eval()
        else: # one of them is trainable
            self.dynamics.train()

        # LDM
        for param in self.dynamics.controlnet_arch_wrapper.diffusion_net.parameters():
            if self.trainable_ldm:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f">>> [LDM] requires_grad = {self.trainable_ldm}")

        # ControlNet
        for param in self.dynamics.controlnet_arch_wrapper.control_net.parameters():
            if self.trainable_controlnet:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f">>> [ControlNet] requires_grad = {self.trainable_controlnet}")

        # Fusion Blocks
        for param in self.dynamics.controlnet_arch_wrapper.fusion_net.parameters():
            if self.trainable_fusion_blocks:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f">>> [FusionBlock] requires_grad = {self.trainable_fusion_blocks}")