"""Render README figures from public numeric results only; no raw training data."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from verify_results import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures'
BLUE, TEAL, INK, MUTED, GRID = '#4f63d8', '#089b8b', '#18283b', '#52657b', '#e5ebf2'


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'text.color': INK, 'axes.labelcolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'axes.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
        'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.titleweight': 'bold', 'axes.titlepad': 17,
        'svg.fonttype': 'none', 'svg.hashsalt': 'roft-pytorch-figures-v1',
    })


def heading(fig, title, subtitle):
    fig.text(.075, .945, title, fontsize=22, fontweight='bold', va='top')
    fig.text(.075, .89, subtitle, fontsize=10.5, color=MUTED, va='top')


def save(fig, name):
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight', metadata={'Date': None})
    fig.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight', metadata={'Software': 'roft-pytorch / Matplotlib'})
    plt.close(fig)


def evaluation(summary):
    fig = plt.figure(figsize=(12.2, 7))
    heading(fig, 'Blender evaluation · Qwen3.5-4B', 'Fresh inference in both arms  •  identical tasks, seeds and verifier  •  34 paired episodes')
    ax = fig.add_axes([.075, .27, .38, .51]); ci = fig.add_axes([.615, .27, .345, .51])
    x = np.arange(2); labels = ['Transfer\n24 episodes / 6 tasks', 'Within-task\n10 episodes / 10 tasks']
    for offset, arm, color, label in [(-.17, 'base', BLUE, 'Base'), (.17, 'roft', TEAL, 'ROFT')]:
        heights = [summary['arms'][arm]['scores'][split]['mean_reward'] for split in ('transfer', 'within_task')]
        bars = ax.bar(x+offset, heights, .29, color=color, label=label, zorder=3)
        for b, value in zip(bars, heights):
            ax.text(b.get_x()+b.get_width()/2, value+.00045, f'{value:.5f}', ha='center', fontsize=10, color=color, fontweight='bold')
    ax.set_xticks(x, labels, fontsize=10); ax.set_ylim(0, .016); ax.set_ylabel('Mean actual verifier reward')
    ax.set_title('Observed mean reward', loc='left', fontsize=13); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, ncol=2, fontsize=10)
    ax.tick_params(axis='both', length=0, pad=9)
    for y, split in [(1, 'transfer'), (0, 'within_task')]:
        d = summary['deltas']['roft_minus_base_'+split]; mean = d['mean_reward_delta']; lo, hi = d['ci95_task_bootstrap']
        ci.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt='o', color=TEAL, markersize=8,
                    elinewidth=3, capsize=6, capthick=2, zorder=3)
        ci.text(mean, y+.16, f'{mean:+.5f}', ha='center', fontsize=11, color=TEAL, fontweight='bold')
    ci.axvline(0, color=MUTED, linestyle=(0, (4, 4)), linewidth=1.3)
    ci.set_yticks([1, 0], ['Transfer', 'Within-task'], fontsize=10); ci.set_ylim(-.47, 1.5)
    ci.set_xlim(-.0115, .017); ci.set_xticks([-.01, 0, .01]); ci.set_xlabel('ROFT − base reward')
    ci.set_title('Paired delta · 95% interval', loc='left', fontsize=13); ci.xaxis.grid(True)
    ci.tick_params(axis='both', length=0, pad=9)
    fig.text(.075, .10, 'Both intervals include zero.', fontsize=12, color=INK, fontweight='bold')
    fig.text(.075, .058, 'Continuous reward scale: 0–1; axes zoom in on low observed scores. Task-cluster bootstrap, 10,000 draws.\nBaseline histories resumed after interruption; one lost tool observation remains explicitly unknown.', fontsize=9, color=MUTED, va='top')
    save(fig, 'evaluation')


def training(checkpoints):
    steps = [a['step'] for a in checkpoints]; counts = [a['supervised_tokens'] for a in checkpoints]
    losses = [a['loss'] for a in checkpoints]; cumulative = np.cumsum(counts)
    assert cumulative[-1] == 256721 and steps == list(range(1, 8))
    fig = plt.figure(figsize=(12.2, 6.8))
    heading(fig, 'Seven fresh online training updates', '112 task attempts  •  448 fresh reflections  •  438 selected targets trained once  •  no replay')
    ax = fig.add_axes([.075, .28, .37, .48]); tok = fig.add_axes([.605, .28, .35, .48])
    ax.plot(steps, losses, '-o', color=TEAL, linewidth=2.8, markersize=7, zorder=3)
    ax.set_xlim(.7, 7.4); ax.set_ylim(0, 1.08); ax.set_xticks(steps); ax.set_xlabel('Saved optimizer update')
    ax.set_ylabel('Token-mean reflection cross-entropy'); ax.set_title('Loss on each fresh batch', loc='left', fontsize=13)
    ax.yaxis.grid(True); ax.tick_params(axis='both', length=0, pad=9)
    for i in (0, 6):
        ax.annotate(f'{losses[i]:.3f}', (steps[i], losses[i]), xytext=(0, 12), textcoords='offset points',
                    ha='center', color=TEAL, fontsize=10, fontweight='bold')
    tok.fill_between(steps, cumulative, color=BLUE, alpha=.10)
    tok.plot(steps, cumulative, '-o', color=BLUE, linewidth=2.8, markersize=7)
    tok.axhline(256721, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    tok.set_xlim(.7, 7.4); tok.set_ylim(0, 290000); tok.set_xticks(steps); tok.set_xlabel('Saved optimizer update')
    tok.set_ylabel('Cumulative supervised backward tokens'); tok.set_title('Exact token budget', loc='left', fontsize=13)
    tok.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
    tok.yaxis.grid(True); tok.tick_params(axis='both', length=0, pad=9)
    tok.annotate('256,721', (7, cumulative[-1]), xytext=(-4, 13), textcoords='offset points',
                 ha='right', color=BLUE, fontsize=12, fontweight='bold')
    fig.text(.075, .11, 'Falling reflection loss does not establish better task reward.', fontsize=12, fontweight='bold')
    fig.text(.075, .065, 'Each point uses different fresh targets; this is not a fixed validation-loss curve. Loss shown before the ×4 gradient scaling.\nOne boundary target omits its final 13 answer / end-of-turn tokens to meet the exact budget.', fontsize=9, color=MUTED, va='top')
    save(fig, 'training')


def per_task(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['split'], r['task']), {'base': [], 'roft': []})[r['arm']].append(r['reward'])
    order = sorted(groups, key=lambda k: (k[0] != 'transfer', k[1]))
    deltas = [np.mean(groups[k]['roft'])-np.mean(groups[k]['base']) for k in order]
    names = [k[1].removeprefix('blender-generated-').removesuffix('-model').replace('-', ' ').capitalize() for k in order]
    fig = plt.figure(figsize=(12.2, 9.2))
    heading(fig, 'Task-level changes are mixed', 'Mean paired reward delta by task  •  descriptive breakdown of the same 34 pairs')
    ax = fig.add_axes([.39, .18, .56, .63]); y = np.arange(len(order))
    colors = [TEAL if v > 0 else '#c65469' if v < 0 else '#9aa8b8' for v in deltas]
    ax.barh(y, deltas, color=colors, height=.62, zorder=3); ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, names, fontsize=9.5); ax.invert_yaxis(); ax.set_xlim(-.056, .047)
    ax.set_xlabel('ROFT − base · mean actual verifier reward'); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0, pad=8)
    for i, value in enumerate(deltas):
        ax.text(value + (.0014 if value >= 0 else -.0014), i, f'{value:+.5f}', va='center',
                ha='left' if value >= 0 else 'right', fontsize=9, color=colors[i])
    ax.axhline(5.5, color=GRID, linewidth=1.5)
    fig.text(.075, .105, 'Transfer: 4 seeds per task. Within-task: 1 seed per task.', fontsize=11, fontweight='bold')
    fig.text(.075, .068, 'Rows above the divider are transfer tasks. No additional sampling, ranking or task selection.\nWhite House baseline recovery includes one explicitly unknown tool observation. These rows add no independent evidence.', fontsize=9, color=MUTED, va='top')
    save(fig, 'per_task')


def main():
    verify(); style(); OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((ROOT/'results/results.json').read_text())
    rows = json.loads((ROOT/'results/episodes.json').read_text())['episodes']
    sources = [ROOT/'results/results.json', ROOT/'results/episodes.json']
    files = sorted((ROOT/'results/checkpoints').glob('checkpoint-*/audit.json')); sources += files
    checkpoints = [json.loads(p.read_text()) for p in files]
    evaluation(result); training(checkpoints); per_task(rows)
    manifest = {'generator': 'tools/plot_results.py', 'matplotlib_version': matplotlib.__version__,
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'figures': ['evaluation', 'training', 'per_task'],
                'formats': ['svg', 'png'], 'new_samples_generated': False,
                'training_data_exported': False, 'data': {'steps': [x['step'] for x in checkpoints],
                'reflection_loss': [x['loss'] for x in checkpoints],
                'cumulative_backward_tokens': np.cumsum([x['supervised_tokens'] for x in checkpoints]).tolist()}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Generated evaluation, training and per-task figures from public numeric records.')

if __name__ == '__main__':
    main()
