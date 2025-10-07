import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Color palette for up to 38 joints (we use first 26)
color_detailed_description = {
    "green": "1", "dimgray": "2", "blue": "3", "brown": "4", "chartreuse": "5",
    "chocolate": "6", "coral": "7", "crimson": "8", "blueviolet": "9", "darkblue": "10",
    "darkgreen": "11", "firebrick": "12", "gold": "13", "teal": "14", "grey": "15",
    "indigo": "16", "steelblue": "17", "indianred": "18", "goldenrod": "19", "darkred": "20",
    "darkorange": "21", "magenta": "22", "maroon": "23", "navy": "24", "olive": "25",
    "orange": "26", "orchid": "27", "pink": "28", "plum": "29", "purple": "30",
    "red": "31", "cornflowerblue": "32", "sienna": "33", "darkkhaki": "34", "tan": "35",
    "dodgerblue": "36", "darkseagreen": "37", "cadetblue": "38"
}

def main():
    """Entry point used for generating train/test images for the full dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    window_size = 16
    stride = 1

    construct_image(
        Pdict_list_path=os.path.join(script_dir, f"../prepare_window/train_sequence_w{window_size}_s{stride}.npy"),
        labels_path=os.path.join(script_dir, f"../prepare_window/train_labels_w{window_size}_s{stride}.npy"),
        base_path=os.path.join(script_dir, f"train_imgs_w{window_size}_s{stride}"),
        bg_color="white",
        override=False,
        linestyle="-",
        linewidth=0.5,
        marker="",
        markersize=1,
        cell_size=(128, 384),
        num_workers=96
    )

    construct_image(
        Pdict_list_path=os.path.join(script_dir, f"../prepare_window/test_sequence_w{window_size}_s{stride}.npy"),
        labels_path=os.path.join(script_dir, f"../prepare_window/test_labels_w{window_size}_s{stride}.npy"),
        base_path=os.path.join(script_dir, f"test_imgs_w{window_size}_s{stride}"),
        bg_color="white",
        override=False,
        linestyle="-",
        linewidth=0.5,
        marker="",
        markersize=1,
        cell_size=(128, 384),
        num_workers=96
    )

def draw_image(
        pid,
        p_norm,  # (window, 3, 26)
        label,
        base_path,
        bg_color="white",
        override=True,
        linestyle="-",
        linewidth=1,
        marker="",
        markersize=2,
        cell_size=(128, 384)
):
    """Render a single temporal window as 3 stacked subplots (X/Y/Z).

    Parameters
    ----------
    pid : int
        Index of the window (used in filename, +1 to be 1-based).
    p_norm : ndarray (T,3,26)
        Normalized pose values in [0,1].
    label : int
        Majority label for the window.
    base_path : str
        Directory base where the image will be saved (subfolders auto-generated).
    bg_color : str
        Background color of the figure.
    override : bool
        If False and file exists, skip.
    linestyle, linewidth, marker, markersize : matplotlib style
        Styling for plotted joint sequences.
    cell_size : (int,int)
        Per subplot (height,width) pixels approximated via DPI.
    """
    grid_height, grid_width = 3, 1
    cell_height, cell_width = cell_size
    img_height = grid_height * cell_height
    img_width = cell_width

    base_path = (
        base_path + f"/" +
        f"{linestyle}*{linewidth}_{marker}*{markersize}_" +
        f"{grid_height}x{grid_width}_{img_height}x{img_width}"
    )
    base_path = base_path.replace("*", "_").replace("**", "__")

    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    img_path = os.path.join(base_path, f"{pid + 1}_label{label}.png")
    if os.path.exists(img_path) and (not override):
        return

    frame, coord, joint = p_norm.shape

    dpi = 200
    fig, axs = plt.subplots(3, 1, figsize=(img_width / dpi, img_height / dpi), dpi=dpi, facecolor=bg_color)
    time_axis = np.arange(1, frame + 1)

    num_joints = 26
    base_colors = list(color_detailed_description.keys())
    plt_colors = base_colors[:num_joints]

    for i, coord_name in enumerate(["X", "Y", "Z"]):
        ax = axs[i]
        for j in range(joint):
            y_val = p_norm[:, i, j]
            color = plt_colors[j % len(plt_colors)]
            ax.plot(
                time_axis, y_val,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                color=color
            )
        ax.set_xlim([1, frame])
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(coord_name, fontsize=12, fontweight="bold")

    for ax in axs[:-1]:
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(img_path, pad_inches=0, facecolor=bg_color)
    plt.close(fig)

def process_single_window(args):
    """Worker function: normalize a window and draw it.

    args is a tuple:
        (idx, p_window, label, base_path, bg_color, override,
         linestyle, linewidth, marker, markersize, cell_size)
    """
    (idx, p_window, label,
     base_path, bg_color, override,
     linestyle, linewidth, marker, markersize, cell_size) = args

    p_norm = p_window.copy()
    frame, coord, joint = p_window.shape
    for c in range(coord):
        data_2d = p_window[:, c, :]
        min_val = data_2d.min()
        max_val = data_2d.max()
        if max_val == min_val:
            p_norm[:, c, :] = 0.0
        else:
            p_norm[:, c, :] = (data_2d - min_val) / (max_val - min_val)

    draw_image(
        pid=idx,
        p_norm=p_norm,
        label=label,
        base_path=base_path,
        bg_color=bg_color,
        override=override,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        cell_size=cell_size
    )

def construct_image(
        Pdict_list_path,
        labels_path,
        base_path=None,
        bg_color="white",
        override=False,
        linestyle="-",
        linewidth=2,
        marker="",
        markersize=2,
        cell_size=(128, 384),
        num_workers=None
):
    """Generate images for every temporal window using multiprocessing.

    Parameters
    ----------
    Pdict_list_path : str
        Path to numpy file containing shape (N, T, 3, 26).
    labels_path : str
        Path to numpy file containing majority label per window shape (N,).
    base_path : str
        Output directory root (will be created if needed).
    num_workers : int or None
        Number of worker processes. If None, uses cpu_count()-1.
    """
    Pdict_list = np.load(Pdict_list_path)
    labels = np.load(labels_path)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"[INFO] Multiprocessing with {num_workers} workers...")

    args_list = []
    for idx, p_window in enumerate(Pdict_list):
        label = labels[idx]
        args_list.append((
            idx,
            p_window,
            label,
            base_path,
            bg_color,
            override,
            linestyle,
            linewidth,
            marker,
            markersize,
            cell_size
        ))

    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_single_window, args_list),
            total=len(args_list),
            desc="Generating images"
        ))

    print("[INFO] All image generation done.")

if __name__ == "__main__":
    main()
