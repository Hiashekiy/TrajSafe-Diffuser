"""Visualize only the dataset counts and protocol stated in the manuscript.

No experiment logs, caches, inferred per-town counts or error bars are used.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures/experiments'
PAPER = ROOT / 'paper/TrajSafe-Diffuser：生成式安全约束下的扩散运动规划方法.md'


def main():
    text = PAPER.read_text(encoding='utf-8')
    for value in ('3850', '1550', '420', '1260', 'Town03', 'Town04',
                  'Town05', 'Town01', 'Town02', 'Town10HD'):
        assert value in text, f'Manuscript does not contain {value}'
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 8, 'svg.fonttype': 'path', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'text.color': '#263445', 'axes.labelcolor': '#263445',
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.25),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    specs = [
        ('(a) 城镇隔离的数据划分',
         ['训练集\nTown03 / Town04', '验证集\nTown05', '测试集\nTown01 / Town02\nTown10HD'],
         [3850, 1550, 420], 4500, [0, 2000, 4000], '样本数'),
        ('(b) 测试集评价协议',
         ['主对比\n420 样本 × 3 种子', '消融实验\n420 样本 × 1 种子',
          '效率实验\n420 样本 × 1 种子'],
         [1260, 420, 420], 1500, [0, 600, 1200], '每种方法或配置\n评价的轨迹数'),
    ]
    palette = ['#0072B2', '#E69F00', '#009E73']
    for ax, (title, labels, values, xmax, ticks, xlabel) in zip(axes, specs):
        bars = ax.barh(range(3), values, height=.5, color=palette,
                       edgecolor='white', linewidth=.5, zorder=3)
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_xlabel(xlabel, labelpad=5)
        ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', pad=10)
        ax.grid(axis='x', color='#D9DEE4', linewidth=.5, zorder=0)
        ax.tick_params(axis='both', length=0, pad=5)
        for bar, value in zip(bars, values):
            ax.text(value + xmax*.025, bar.get_y()+bar.get_height()/2,
                    f'{value:,}', va='center', fontsize=8)
    fig.subplots_adjust(left=.235, right=.985, bottom=.24, top=.83, wspace=1.08)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'fig4-1.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    (OUT / 'fig4-1-provenance.json').write_text(json.dumps({
        'source': str(PAPER), 'source_type': 'current manuscript only',
        'dataset_samples': {'train': 3850, 'validation': 1550, 'test': 420},
        'trajectories_per_configuration': {'main': 1260, 'ablation': 420, 'efficiency': 420},
        'no_per_town_counts_or_empirical_distributions_inferred': True,
        'generator': 'tools/plot_paper_dataset_protocol.py',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {OUT / "fig4-1.pdf"}')


if __name__ == '__main__':
    main()
