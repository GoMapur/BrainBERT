import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import models
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import torch
from glob import glob
from datasets.masked_tf_dataset import *
from omegaconf import OmegaConf

def build_model(cfg):
    ckpt_path = cfg.upstream_ckpt
    init_state = torch.load(ckpt_path)
    upstream_cfg = init_state["model_cfg"]
    upstream = models.build_model(upstream_cfg)
    return upstream, upstream_cfg

def load_model_weights(model, states, multi_gpu):
    if multi_gpu:
        model.module.load_weights(states)
    else:
        model.load_weights(states)
        
from omegaconf import OmegaConf

device = "cuda:1"
ckpt_path = "../pretrained_weights/stft_large_pretrained.pth"
cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
model, upstream_cfg = build_model(cfg)
model.to(device)
init_state = torch.load(ckpt_path)
load_model_weights(model, init_state['model'], False)

for picked_channel in glob("/mnt/AI_Magic/projects/iEEG_data/spindles/np_2kHz_segmented_laplacian/Th_proj_Pt9/*"):
    # only go if it is a directory
    if not os.path.isdir(picked_channel):
        continue
    dataset_cfg = OmegaConf.load("../conf/data/masked_tf_dataset.yaml")
    task_cfg = OmegaConf.load("../conf/task/fixed_mask_pretrain.yaml")
    preprocessor_cfg = OmegaConf.load("../conf/preprocessor/stft.yaml")
    dataset_cfg["data"] = picked_channel
    dataset_cfg["use_mask"] = False

    dataset = MaskedTFDataset(dataset_cfg, task_cfg=task_cfg, preprocessor_cfg=preprocessor_cfg)

    from tqdm import tqdm
    stats = []
    samples = []

    for sample in tqdm(dataset):
        inputs = torch.FloatTensor(sample["masked_input"]).unsqueeze(0).to(device)
        mask = torch.zeros((inputs.shape[:2])).bool().to(device)
        
        with torch.no_grad():
            pred_out = model.forward(inputs, mask, intermediate_rep=False)
            embed_out = model.forward(inputs, mask, intermediate_rep=True)
            
        original = sample["target"].cpu().numpy()
        recovery = pred_out[0][0].cpu().numpy()
        embedding = embed_out[0][0].cpu().numpy()
        
        mean = sample["mean"]
        std = sample["std"]
        un_normalized_data = sample["un_normalized_target"]
        wav = sample["wav"]
        erased_Zxx = sample["erased_Zxx"]
        fn = sample["fn"]
        
        stats += [np.mean(np.abs(recovery - original)) / np.mean(np.abs(original))]
        samples += [[sample["masked_input"].cpu().numpy(), 
                    sample["target"].cpu().numpy(), 
                    recovery,
                    embedding,
                    sample["freq"], 
                    sample["time"], 
                    np.mean(np.abs(recovery - original)) / np.mean(np.abs(original)), 
                    mean, 
                    std, 
                    un_normalized_data, 
                    wav, 
                    erased_Zxx,
                    fn]]

    import pandas as pd
    samples_df = pd.DataFrame(samples, columns=["masked_input", "target", "recovery", "embedding", "freq", "time", "error", "mean", "std", "un_normalized_data", "wav", "erased_Zxx", "fn"])
    samples_df["label"] = samples_df["fn"].apply(lambda x: x.split("/")[0])

    data = np.array(samples_df["embedding"].tolist())
    labels = np.array(samples_df["label"].tolist()).astype(str)

    if data.shape[0] <= 30:
        continue

    import numpy as np
    import plotly.express as px
    from sklearn import datasets
    from sklearn.manifold import TSNE
    import umap
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X = data
    y = labels

    # Perform t-SNE and UMAP transformations
    tsne = TSNE(n_components=3, random_state=42)
    umap = umap.UMAP(n_components=3, random_state=42)

    X_tsne = tsne.fit_transform(X)
    X_umap = umap.fit_transform(X)

    # Create dataframes for Plotly
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2', 'Dim3'])
    df_tsne['Target'] = y
    df_umap = pd.DataFrame(X_umap, columns=['Dim1', 'Dim2', 'Dim3'])
    df_umap['Target'] = y

    # Create 3D scatter plots using Plotly
    fig_tsne = px.scatter_3d(df_tsne, x='Dim1', y='Dim2', z='Dim3', color='Target', 
                            labels={'Target': 'Class'}, title='t-SNE Projection', opacity=0.7)
    fig_umap = px.scatter_3d(df_umap, x='Dim1', y='Dim2', z='Dim3', color='Target', 
                            labels={'Target': 'Class'}, title='UMAP Projection', opacity=0.7)

    def gen_range_ticks(data):
        xrange = [np.min(data), np.max(data)]
        xrange_ticks = np.linspace(xrange[0], xrange[1], 11)
        return xrange, xrange_ticks

    x_range_tsne, x_range_ticks_tsne = gen_range_ticks(X_tsne[:,0])
    y_range_tsne, y_range_ticks_tsne = gen_range_ticks(X_tsne[:,1])
    z_range_tsne, z_range_ticks_tsne = gen_range_ticks(X_tsne[:,2])

    fig_tsne.update_layout(
        scene = dict(
            aspectmode  = 'cube',
            aspectratio = dict(x=1, y=1, z=1),
            xaxis = dict(nticks=11, tickvals=x_range_ticks_tsne, ticktext=np.round(x_range_ticks_tsne, 1), range=x_range_tsne,),
            yaxis = dict(nticks=11, tickvals=y_range_ticks_tsne, ticktext=np.round(y_range_ticks_tsne, 1), range=y_range_tsne,),
            zaxis = dict(nticks=11, tickvals=z_range_ticks_tsne, ticktext=np.round(z_range_ticks_tsne, 1), range=z_range_tsne,),
            ),
        width=1600,
    )
    fig_tsne.update_scenes(
        xaxis_autorange=False,
        yaxis_autorange=False,
        zaxis_autorange=False,
    )

    x_range_umap, x_range_ticks_umap = gen_range_ticks(X_umap[:,0])
    y_range_umap, y_range_ticks_umap = gen_range_ticks(X_umap[:,1])
    z_range_umap, z_range_ticks_umap = gen_range_ticks(X_umap[:,2])

    fig_umap.update_layout(
        scene = dict(
            aspectmode  = 'cube',
            aspectratio = dict(x=1, y=1, z=1),
            xaxis = dict(nticks=11, tickvals=x_range_ticks_umap, ticktext=np.round(x_range_ticks_umap, 1), range=x_range_umap,),
            yaxis = dict(nticks=11, tickvals=y_range_ticks_umap, ticktext=np.round(y_range_ticks_umap, 1), range=y_range_umap,),
            zaxis = dict(nticks=11, tickvals=z_range_ticks_umap, ticktext=np.round(z_range_ticks_umap, 1), range=z_range_umap,),
            ),
        width=1600,
    )
    fig_umap.update_scenes(
        xaxis_autorange=False,
        yaxis_autorange=False,
        zaxis_autorange=False,
    )

    # Show the plots
    html_save_fn = picked_channel.split("/")[-1]
    if not os.path.exists(f"./vis/{html_save_fn}"):
        os.makedirs(f"./vis/{html_save_fn}")
    fig_tsne.write_html(f"./vis/{html_save_fn}/tsne.html")
    fig_umap.write_html(f"./vis/{html_save_fn}/umap.html")