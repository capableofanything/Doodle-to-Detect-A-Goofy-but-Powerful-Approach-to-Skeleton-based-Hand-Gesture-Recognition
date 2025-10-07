"""Demo script for generating a small sample of SHREC22 windows and images.

Workflow:
1. Build a tiny subset of windows from a few raw sequence files (training set).
2. Save the subset numpy arrays using the Dataset_shrec22 class (data_set="demo").
3. Render corresponding stacked coordinate images with draw.construct_image.

Usage:
python demo.py --num_files 2 --window 16 --stride 1

Requirements:
- Raw data present in shrec22/SHREC2022/shrec2022_training_set
- annotations.txt inside that folder
- The prepare_window and draw steps will create files inside their respective directories.
"""
from __future__ import annotations
import os
import argparse
from shrec22.prepare_window.shrec22_prepare_window import Dataset_shrec22
from shrec22.draw.shrec22_draw_3stack import construct_image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_files', type=int, default=2, help='How many training sequence files to sample')
    ap.add_argument('--file_names', type=str, nargs='*', default=None, help='Explicit file basenames (override num_files)')
    ap.add_argument('--window', type=int, default=16, help='Sliding window size')
    ap.add_argument('--stride', type=int, default=1, help='Sliding stride')
    return ap.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_raw_dir = os.path.join(script_dir, 'shrec22/SHREC2022/shrec2022_training_set')

    if not os.path.isdir(train_raw_dir):
        raise FileNotFoundError(f"Training raw directory not found: {train_raw_dir}")

    # If explicit file names not provided, pick first N (by file listing order)
    if args.file_names is None:
        all_files = [f for f in os.listdir(train_raw_dir) if f.endswith('.txt') and f != 'annotations.txt']
        # Remove extension & keep ordering
        all_files = sorted([os.path.splitext(f)[0] for f in all_files])
        file_subset = all_files[:args.num_files]
    else:
        file_subset = args.file_names

    print(f"Using files: {file_subset}")

    # Build subset dataset (saves arrays automatically)
    subset_dataset = Dataset_shrec22(
        data_dir=train_raw_dir,
        data_set='demo',
        w=args.window,
        stride=args.stride,
        file_names=file_subset,
        max_files=len(file_subset),
        save=True
    )
    print(f"Total windows (subset): {len(subset_dataset)}")

    prepare_dir = os.path.join(script_dir, 'shrec22/prepare_window')
    seq_path = os.path.join(prepare_dir, f"demo_sequence_w{args.window}_s{args.stride}.npy")
    label_path = os.path.join(prepare_dir, f"demo_labels_w{args.window}_s{args.stride}.npy")

    draw_base = os.path.join(script_dir, 'shrec22/draw/demo_imgs')
    os.makedirs(draw_base, exist_ok=True)

    print("Rendering images...")
    construct_image(
        Pdict_list_path=seq_path,
        labels_path=label_path,
        base_path=os.path.join(draw_base, f"w{args.window}_s{args.stride}"),
        num_workers=64,
        override=True,
        linewidth=0.5,
        markersize=1,
    )
    print("Demo complete. Check the draw/demo_imgs folder.")

if __name__ == '__main__':
    main()

