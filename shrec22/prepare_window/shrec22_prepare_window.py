import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
from tqdm import tqdm
import os.path as opt
from random import randint, shuffle
from typing import List, Optional

class Dataset_shrec22(Dataset):
    """
    SHREC2022 dataset window builder.

    This class reads raw per-frame skeleton data files and an annotations file
    ("annotations.txt") describing gesture segments, then constructs a sliding
    window dataset of shape (num_windows, w, 3, 26).

    Parameters
    ----------
    data_dir : str
        Directory containing raw sequence text files plus annotations.txt.
    data_set : str
        A label prefix used when saving generated numpy arrays (e.g. "train", "test", "demo").
    w : int
        Sliding window size (number of consecutive frames per window).
    stride : int, default 1
        Sliding stride.
    file_names : Optional[List[str]]
        If provided, only process these (basename without extension) files.
    max_files : Optional[int]
        If provided, stop after processing at most this many files (after any filtering).
    save : bool
        If True, automatically save generated arrays upon initialization.

    Generated Attributes
    --------------------
    sequence : List[np.ndarray]
        List of windows each shaped (w, 3, 26).
    labels_window : List[np.ndarray]
        Frame-level labels for each window (w,).
    label : List[int]
        Majority (mode) label for each window.
    all_file_poses : List[np.ndarray]
        Per-file full pose arrays (num_frames, 3, 26).
    window_source_file_indices : List[str]
        Source filename for each window.
    window_frame_indices : List[List[int,int]]
        Start/end (inclusive) frame indices (relative to file) for each window.
    """
    def __init__(self,
                 data_dir: str,
                 data_set: str,
                 w: int,
                 stride: int = 1,
                 file_names: Optional[List[str]] = None,
                 max_files: Optional[int] = None,
                 save: bool = True):
        self.path_to_data = data_dir
        self.w = w
        self.stride = stride
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Storage containers
        self.sequence = []
        self.labels_window = []
        self.label = []
        self.all_file_poses = []
        self.window_source_file_indices = []
        self.window_frame_indices = []

        # 17 labels (16 gestures + non-gesture (index 16))
        self.label_map = [
            "ONE", "TWO", "THREE", "FOUR", "OK", "MENU", "LEFT", "RIGHT",
            "CIRCLE", "V", "CROSS", "GRAB", "PINCH", "DENY", "WAVE", "KNOB",
            "nongesture"
        ]

        annotations_path = opt.join(self.path_to_data, "annotations.txt")
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"annotations.txt not found in {self.path_to_data}")

        # Normalize provided file name list (strip extensions)
        if file_names is not None:
            file_names = [os.path.splitext(f)[0] for f in file_names]
            file_names_set = set(file_names)
        else:
            file_names_set = None

        processed_files = 0
        with open(annotations_path, "r") as gt:
            for raw_line in gt.readlines():
                line = raw_line.strip('\n').split(";")
                if '' in line:
                    line.remove('')
                file_name = line[0]

                # Apply file subset filtering
                if file_names_set is not None and file_name not in file_names_set:
                    continue
                if max_files is not None and processed_files >= max_files:
                    break

                gesture_tokens = line[1:]

                # Initialize frame-level label array (default to nongesture index = 16)
                gt_window = np.full(780, 16)  # 780 assumed max frames per sequence
                for index in range(0, len(gesture_tokens), 3):
                    lab = gesture_tokens[index]
                    s = int(gesture_tokens[index + 1])
                    e = int(gesture_tokens[index + 2])
                    gt_window[s: e + 1] = [self.label_map.index(lab)] * (e - s + 1)

                file_path = opt.join(self.path_to_data, f"{file_name}.txt")
                if not os.path.isfile(file_path):
                    print(f"[WARN] Missing pose file: {file_path}; skipping.")
                    continue

                file_poses = []
                with open(file_path, "r") as fp:
                    for line_idx, pose_line in enumerate(fp.readlines()):
                        coords = pose_line.split(";")[2:-1]  # Skip first two tokens & last empty
                        coords = np.reshape(coords, (26, 3)).transpose().astype(np.float64)  # (3,26)
                        file_poses.append([coords[0], coords[1], coords[2]])
                file_poses = np.array(file_poses).astype(np.float64)  # (num_frames,3,26)
                self.all_file_poses.append(file_poses)

                # Sliding window
                for poses_index in range(0, file_poses.shape[0] - self.w, self.stride):
                    start_idx = poses_index
                    end_idx = poses_index + self.w - 1

                    self.sequence.append(file_poses[poses_index: poses_index + self.w, :, :])
                    self.window_source_file_indices.append(file_name)

                    label_window = gt_window[poses_index: poses_index + self.w]
                    label_count = list(np.bincount(label_window.astype("int64")))
                    self.label.append(label_count.index(max(label_count)))
                    self.labels_window.append(label_window)
                    self.window_frame_indices.append([start_idx, end_idx])

                processed_files += 1

        self.len_data = len(self.sequence)
        self.all_file_poses = np.array(self.all_file_poses, dtype=object)

        if save:
            self.save_windows_and_labels_no_norm(data_set, self.stride)

        print(f"Initialization complete. Total windows: {self.len_data}")

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.label[idx], self.labels_window[idx]

    def save_windows_and_labels_no_norm(self, data_set, stride):
        """Save windows, majority labels, and frame-level labels to .npy files."""
        save_path = os.path.join(self.script_dir, f"{data_set}_sequence_w{self.w}_s{stride}.npy")
        labels_save_path = os.path.join(self.script_dir, f"{data_set}_labels_w{self.w}_s{stride}.npy")
        labels_window_save_path = os.path.join(self.script_dir, f"{data_set}_labels_window_w{self.w}_s{stride}.npy")

        sequence_array = np.array(self.sequence, dtype=np.float64)
        labels_array = np.array(self.label, dtype=np.int64)
        labels_window_array = np.array(self.labels_window, dtype=np.int64)

        np.save(save_path, sequence_array)
        np.save(labels_save_path, labels_array)
        np.save(labels_window_save_path, labels_window_array)

        print(f"Sequence saved to {save_path}")
        print(f"Labels saved to {labels_save_path}")
        print(f"Labels_window saved to {labels_window_save_path}")

    def _save_window_source_file_indices(self, data_set, stride):
        """Save the source file name for each window to .npy."""
        save_path = os.path.join(self.script_dir, f"{data_set}_window_source_file_indices_s{stride}.npy")
        source_indices_array = np.array(self.window_source_file_indices)
        np.save(save_path, source_indices_array)
        print(f"Window source file indices saved to {save_path}")
        print(f"Shape: {source_indices_array.shape}")

    def _save_window_frame_indices(self, data_set, stride):
        """Save (start,end) frame indices for each window to .npy."""
        save_path = os.path.join(self.script_dir, f"{data_set}_window_frame_indices_s{stride}.npy")
        frame_indices_array = np.array(self.window_frame_indices, dtype=np.int64)
        np.save(save_path, frame_indices_array)
        print(f"Window frame indices saved to {save_path}")
        print(f"Shape: {frame_indices_array.shape}")


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_train = os.path.join(script_dir, "../SHREC2022/shrec2022_training_set")
    data_dir_test = os.path.join(script_dir, "../SHREC2022/shrec2022_test_set")
    dataset = Dataset_shrec22(data_dir=data_dir_train, data_set="train", w=16, stride=1)
    print("Accumulated file poses shape:", np.shape(dataset.all_file_poses))

    dataset = Dataset_shrec22(data_dir=data_dir_test, data_set="test", w=16, stride=1)
    print("Accumulated file poses shape:", np.shape(dataset.all_file_poses))
