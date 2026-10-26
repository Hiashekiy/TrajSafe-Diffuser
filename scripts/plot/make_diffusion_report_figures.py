"""Generate self-contained educational figures for the diffusion report."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "diffusion_report"


def make_target(size=384):
    """Draw a small original scene so the report has no external image dependency."""
    im = Image.new("RGB", (size, size), "white")
    p = ImageDraw.Draw(im)
    # Sky gradient
    for y in range(size):
        u = y / (size - 1)
        c = tuple(int((1-u)*a + u*b) for a, b in zip((27, 57, 96), (166, 202, 224)))
        p.line((0, y, size, y), fill=c)
    # Moon, distant mountains and snow
    p.ellipse((size*0.72, size*0.10, size*0.87, size*0.25), fill=(247, 239, 202))
    p.polygon([(0, size*.60), (size*.22, size*.34), (size*.40, size*.60)], fill=(78, 105, 132))
    p.polygon([(size*.22, size*.60), (size*.52, size*.28), (size*.78, size*.60)], fill=(61, 88, 117))
    p.polygon([(size*.58, size*.60), (size*.80, size*.39), (size, size*.60)], fill=(85, 112, 137))
    p.rectangle((0, size*.58, size, size), fill=(225, 239, 244))
    # Evergreen silhouettes
    for x, y, scale in [(45, 235, .85), (320, 222, 1.05), (275, 250, .65), (92, 250, .60)]:
        p.rectangle((x-3, y, x+3, y+75*scale), fill=(45, 70, 68))
        for k in range(3):
            yy = y + k*22*scale
            half = (30-k*5)*scale
            p.polygon([(x, yy-36*scale), (x-half, yy+18*scale), (x+half, yy+18*scale)], fill=(28, 91, 78))
    # Fox body, head, ears, chest, tail and face
    orange = (218, 103, 42)
    dark = (54, 43, 42)
    cream = (245, 226, 196)
    p.ellipse((size*.39, size*.54, size*.62, size*.88), fill=orange)
    p.polygon([(size*.40, size*.50), (size*.45, size*.35), (size*.50, size*.49)], fill=orange)
    p.polygon([(size*.51, size*.49), (size*.57, size*.35), (size*.61, size*.51)], fill=orange)
    p.ellipse((size*.39, size*.42, size*.62, size*.67), fill=orange)
    p.polygon([(size*.45, size*.56), (size*.56, size*.56), (size*.51, size*.69)], fill=cream)
    p.ellipse((size*.47, size*.52, size*.49, size*.55), fill=dark)
    p.ellipse((size*.54, size*.52, size*.56, size*.55), fill=dark)
    p.ellipse((size*.505, size*.59, size*.535, size*.62), fill=dark)
    p.ellipse((size*.25, size*.67, size*.48, size*.86), fill=orange)
    p.ellipse((size*.22, size*.72, size*.34, size*.84), fill=cream)
    return np.asarray(im, dtype=np.float32) / 127.5 - 1.0


def noisy_states(x0, alphas, seed=20260927):
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=x0.shape).astype(np.float32)
    states = []
    for a in alphas:
        xt = np.sqrt(a) * x0 + np.sqrt(1.0-a) * eps
        states.append(np.clip((xt + 1.0) / 2.0, 0.0, 1.0))
    return states


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "font.size": 9, "pdf.fonttype": 42})
    x0 = make_target()
    alpha_reverse = [0.0, 0.03, 0.10, 0.25, 0.55, 1.0]
    reverse = noisy_states(x0, alpha_reverse)

    # Figure 1: the requested step-by-step denoising demonstration.
    fig, axes = plt.subplots(1, 6, figsize=(11.8, 2.25), constrained_layout=True)
    labels = [r"$x_T$", r"$x_{t_5}$", r"$x_{t_4}$", r"$x_{t_3}$", r"$x_{t_2}$", r"$x_0$"]
    subtitles = ["random noise", "global colors", "coarse layout",
                 "semantic shape", "fine details", "clean sample"]
    for ax, image, label, sub in zip(axes, reverse, labels, subtitles):
        ax.imshow(image)
        ax.set_title(label, fontsize=11, pad=3)
        ax.text(.5, -.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(.65); s.set_color("#7c8790")
    fig.savefig(OUT / "diffusion_denoising_sequence.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "diffusion_denoising_sequence.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: forward and reverse directions together.
    forward = list(reversed(reverse))
    fig, axes = plt.subplots(2, 6, figsize=(11.8, 4.35), constrained_layout=True)
    for j, image in enumerate(forward):
        axes[0, j].imshow(image)
        axes[0, j].set_title([r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_T$"][j], fontsize=10)
    for j, image in enumerate(reverse):
        axes[1, j].imshow(image)
        axes[1, j].set_title([r"$x_T$", r"$x_{T-1}$", r"$x_{T-2}$", r"$x_{T-3}$", r"$x_1$", r"$x_0$"][j], fontsize=10)
    axes[0, 0].set_ylabel("forward diffusion\n(add noise)", fontsize=9)
    axes[1, 0].set_ylabel("reverse process\n(remove noise)", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(.55); s.set_color("#8c969d")
    fig.savefig(OUT / "diffusion_forward_reverse.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "diffusion_forward_reverse.pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
