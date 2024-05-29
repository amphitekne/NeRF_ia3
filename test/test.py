import torch
from render.rendering import render_rays
from tqdm import tqdm
@torch.no_grad()
def test(
        model,
        estimator,
        o,
        d,
        H,
        W,
        tn,
        tf,
        render_step_size,
        chunk_size=1000,
        device="cuda",
):
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)
    image = []
    for o_batch, d_batch in tqdm(zip(o, d)):
        o_batch.to(device)
        d_batch.to(device)
        with torch.no_grad():
            prediction, _, _, _, _ = render_rays(model, estimator, render_step_size, o_batch, d_batch, tn, tf, device=device)
        image.append(prediction)  # N, 3
        torch.cuda.empty_cache()
    image = torch.cat(image)
    image = image.reshape(H, W, 3)

    return image