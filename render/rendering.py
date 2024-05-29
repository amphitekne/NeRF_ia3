import torch
from nerfacc.volrend import rendering as rendering_nerfacc


def render_rays(
        model, estimator, render_step_size, rays_o, rays_d, tn, tf, device="cuda", ) -> tuple:
    def sigma_fn(t_starts, t_ends, ray_indices):
        """ Define how to query density for the estimator."""
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        with torch.no_grad():
            sigmas = model.density(positions)
        sigmas = torch.from_numpy(sigmas.cpu().detach().numpy()).to(sigmas.device)
        return sigmas.squeeze(-1)  # (n_samples,)

    def positions_fn(t_starts, t_ends, ray_indices):
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = (t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0)
        return positions, t_dirs

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        positions, t_dirs = positions_fn(t_starts, t_ends, ray_indices)
        rgbs, sigmas = model.forward(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    ray_indices, t_starts, t_ends = estimator.sampling(rays_o, rays_d, sigma_fn=sigma_fn, near_plane=tn,
                                                       far_plane=tf, render_step_size=render_step_size, alpha_thre=0.00)
    rgb, opacity, depth, extras = rendering_nerfacc(t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0],
                                                    rgb_sigma_fn=rgb_sigma_fn,
                                                    render_bkgd=torch.Tensor([0., 0., 0.]).to(device))
    return rgb, opacity, depth, extras, [ray_indices, t_starts, t_ends]
