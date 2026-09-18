"""
The animation at the top of the README, plus a static version of its final frame.

Three fires on a 160 x 160 lattice, lit along the left edge, with rule B at
p = 0.9, whose critical density is D_c = 0.6396 (Tarasevich & van der Marck):
one below, one at and one above the threshold.

The runs are ordinary seeded runs. For the middle panel the script uses the
first seed whose fire reaches the right edge (at the threshold only about half
do), so the animation shows a spanning cluster.

Writes figures/fire_spread.gif and figures/burn_patterns(-dark).png.

    python scripts/make_animation.py
"""

from _common import FIGURES, parser

import numpy as np

from firesim import simulate, theory
from firesim.viz import STATE_LABELS, THEMES, save, state_cmap, themed

L = 160
P_BURN = 0.9
PANELS = [
    (0.58, "Below threshold", "D = 0.58: the fire dies out"),
    (0.64, "At threshold", "D = 0.64: it barely gets through"),
    (0.72, "Above threshold", "D = 0.72: it sweeps across"),
]


def run_panels(seed):
    runs = []
    for i, (density, _, _) in enumerate(PANELS):
        s = seed + 1000 * i
        while True:
            res, frames = simulate(L, density, P_BURN, "B", np.random.default_rng(s))
            if i != 1 or res["crossed"]:
                break
            s += 1
        runs.append((res, frames, s))
        print(f"D = {density}: seed {s}, {res['steps']} steps, crossed={res['crossed']}, "
              f"burned {res['burn_fraction']:.0%} of trees")
    return runs


def text_strip(width_px, height_px, items, theme, fontsize, weight="normal", color="ink"):
    """Render text items [(x_fraction, text)] into an RGB array of the given size."""
    import matplotlib.pyplot as plt

    dpi = 100
    t = THEMES[theme]
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(t["surface"])
    for x, text in items:
        fig.text(x, 0.5, text, ha="left", va="center", fontsize=fontsize,
                 color=t[color], weight=weight, family="DejaVu Sans")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def legend_strip(width_px, height_px, theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dpi = 100
    t = THEMES[theme]
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(t["surface"])
    x = 0.0
    for colour, label in zip(t["states"][1:], STATE_LABELS[1:]):
        fig.patches.append(Rectangle((x, 0.3), 0.012, 0.4, transform=fig.transFigure,
                                     facecolor=colour, edgecolor="none"))
        fig.text(x + 0.018, 0.5, label, va="center", fontsize=11, color=t["ink2"],
                 family="DejaVu Sans")
        x += 0.1
    fig.text(1.0, 0.5, f"rule B, spread probability p = {P_BURN}, critical density "
             f"{theory.TVDM_SITE_GIVEN_BOND[P_BURN]}", ha="right", va="center",
             fontsize=11, color=t["muted"], family="DejaVu Sans")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def write_gif(runs, path, theme="dark", scale=2, gap=24, margin=20, every=3,
              frame_ms=40, hold_ms=3000):
    from matplotlib.colors import to_rgb
    from PIL import Image

    from firesim.viz import lattice_rgb

    t = THEMES[theme]
    panel_w = L * scale
    width = 3 * panel_w + 2 * gap
    titles = text_strip(width, 34, [((panel_w + gap) * i / width, title)
                                    for i, (_, title, _) in enumerate(PANELS)],
                        theme, 15, weight="bold")
    subtitles = text_strip(width, 26, [((panel_w + gap) * i / width, sub)
                                       for i, (_, _, sub) in enumerate(PANELS)],
                           theme, 12, color="ink2")
    legend = legend_strip(width, 40, theme)
    bg = (np.array(to_rgb(t["surface"])) * 255).astype(np.uint8)

    def framed(img):
        h, w, _ = img.shape
        out = np.empty((h + 2 * margin, w + 2 * margin, 3), dtype=np.uint8)
        out[:] = bg
        out[margin:margin + h, margin:margin + w] = img
        return out

    # palette: exact lattice colours + grey ramps for anti-aliased text
    rgb = lambda c: tuple(int(round(255 * v)) for v in to_rgb(c))
    colours = [rgb(c) for c in t["states"]]
    for target in (t["ink"], t["ink2"], t["muted"]):
        a, b = np.array(rgb(t["surface"])), np.array(rgb(target))
        colours += [tuple(int(v) for v in a + (b - a) * f) for f in np.linspace(0.1, 1, 16)]
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for c in colours for v in c] + [0] * (768 - 3 * len(colours)))

    frames = [f for _, f, _ in runs]
    n = max(len(f) for f in frames)
    steps = list(range(0, n, every))
    if steps[-1] != n - 1:
        steps.append(n - 1)

    pad = np.broadcast_to(bg, (12, width, 3))
    images = []
    for s in steps:
        tiles = [lattice_rgb(f[min(s, len(f) - 1)], theme, scale) for f in frames]
        spacer = np.broadcast_to(bg, (panel_w, gap, 3))
        row = np.concatenate([tiles[0], spacer, tiles[1], spacer, tiles[2]], axis=1)
        canvas = framed(np.concatenate([titles, subtitles, pad, row, pad, legend], axis=0))
        im = Image.fromarray(np.ascontiguousarray(canvas))
        images.append(im.quantize(palette=pal, dither=getattr(Image, "Dither", Image).NONE))

    durations = [frame_ms] * (len(images) - 1) + [hold_ms]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True)
    return path, len(images)


def static_figure(runs, theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with themed(theme) as t:
        cmap, norm = state_cmap(theme)
        fig, axes = plt.subplots(1, 3, figsize=(10, 4.3))
        for ax, (density, title, _), (res, frames, _) in zip(axes, PANELS, runs):
            ax.imshow(frames[-1], cmap=cmap, norm=norm)
            ax.set_title(f"{title}, D = {density}", fontsize=10.5, pad=20)
            ax.text(0, 1.02, f"{res['burn_fraction']:.0%} of trees burned, "
                    f"{'crossed' if res['crossed'] else 'did not cross'}",
                    transform=ax.transAxes, fontsize=9, color=t["ink2"], va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
        handles = [Patch(facecolor=t["states"][1], label="tree (unburned)"),
                   Patch(facecolor=t["states"][3], label="ash (burned)"),
                   Patch(facecolor=t["surface"], edgecolor=t["axis"], label="empty")]
        fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 0.0))
        fig.text(0.0, 1.0, f"Burn scars left by fires lit along the left edge "
                 f"(rule B, p = {P_BURN}, {L} x {L} lattice)", ha="left", va="top",
                 fontsize=11.5, weight="semibold", color=t["ink"])
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        path = save(fig, FIGURES / "burn_patterns", theme)
        plt.close(fig)
        return path


def main():
    args = parser(__doc__).parse_args()
    FIGURES.mkdir(exist_ok=True)
    runs = run_panels(args.seed)
    path, n = write_gif(runs, FIGURES / "fire_spread.gif")
    print(f"wrote {path} ({n} frames, {path.stat().st_size / 1e6:.2f} MB)")
    for theme in ("light", "dark"):
        print("wrote", static_figure(runs, theme))


if __name__ == "__main__":
    main()
