import os
import numpy as np
import pandas as pd
from preprocess.datatool import DataTool

# ====== 全局常量 (根据新逻辑更新) ======
DEFAULT_FILENAME = "data/MSL/MSL_train.npy"
DEFAULT_TEST_FILE = "data/MSL/MSL_test.npy"
DEFAULT_TEST_LABEL_FILE = "data/MSL/MSL_test_label.npy"

# --- 以补丁为中心的设置 ---
SEQ_LEN = 9          # 输入/输出序列中包含的补丁数量。
PATCH_LEN = 20       # 每个补丁的时间步长度。
PATCH_STRIDE = 10    # 提取补丁时的时间步长。

# --- 数据生成设置 ---
WINDOW_STRIDE = 20   # 生成每个(输入, 目标)序列对时的滑动窗口步长。

# --- 数据集划分比例 ---
TRAIN_RATIO = 0.9      # 用于训练的总窗口比例。
VALID_RATIO = 0.1      # 用于验证的总窗口比例。
FINAL_TEST_RATIO = 0.2 # 从测试集中划分出的最终测试集比例。
TUNE_VAL_RATIO = 0.1   # 从调优集中划分出的验证集比例。


class AnomalyPatchExtractor:
    """
    修改后: 此类现在提取适用于自回归模型的重叠补丁序列
    (例如, 从补丁N预测补丁N+1)。
    """
    def __init__(
        self,
        filename: str = DEFAULT_FILENAME,
        seq_len: int = SEQ_LEN,
        patch_len: int = PATCH_LEN,
        patch_stride: int = PATCH_STRIDE,
        window_stride: int = WINDOW_STRIDE,
        train_ratio: float = TRAIN_RATIO,
        debug: bool = False
    ):
        """
        修改后: 移除了 `input_len` 和 `output_len`。
        现在的关键参数是 `seq_len` (序列中的补丁数量)。
        """
        self.data_tool = DataTool(filename, debug=debug)
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.window_stride = window_stride
        self.train_ratio = train_ratio
        self.debug = debug

    def _create_sequences_from_data(self, data: np.ndarray) -> tuple:
        """
        新增与重构: 这是核心的新函数。
        它在原始数据上滑动一个"超级窗口"，并在每个位置上生成一个
        (输入序列, 目标序列)的补丁对。目标序列是输入序列向右平移一个补丁的结果。

        Args:
            data: 原始时间序列数据，形状为 (T, C) 或 (T,)。

        Returns:
            一个元组 (x_patch_sequences, y_patch_sequences)。
        """
        # 如果数据是标签数组 (T,)，则重塑为 (T, 1) 以便统一处理。
        is_label_data = (data.ndim == 1)
        if is_label_data:
            data = data.reshape(-1, 1)

        T, C = data.shape

        # 1. 计算"超级窗口"的总时间步长度。
        #    这个窗口必须足够长，以包含 seq_len + 1 个补丁。
        total_ts_len = (self.seq_len * self.patch_stride) + self.patch_len

        # 2. 在时间序列上滑动超级窗口。
        x_sequences, y_sequences = [], []
        max_start = T - total_ts_len + 1

        for start in range(0, max_start, self.window_stride):
            super_window = data[start : start + total_ts_len] # 形状: (total_ts_len, C)

            # 3. 将这个超级窗口分割成其所有的构成补丁。
            #    这将产生 seq_len + 1 个补丁。
            all_patches = self._split_window_into_patches(super_window) # 形状: (seq_len + 1, patch_len, C)

            # 4. 创建输入(x)和目标(y)序列。
            #    x 是前 `seq_len` 个补丁。
            #    y 是后 `seq_len` 个补丁 (即 x 向右平移一个单位)。
            x_seq = all_patches[:-1]  # 形状: (seq_len, patch_len, C)
            y_seq = all_patches[1:]   # 形状: (seq_len, patch_len, C)

            x_sequences.append(x_seq)
            y_sequences.append(y_seq)
        
        if not x_sequences:
            raise ValueError("无法从数据中提取任何序列。请检查数据长度和窗口/补丁参数。")

        x_stacked = np.stack(x_sequences, axis=0) # 形状: (N_sequences, seq_len, patch_len, C)
        y_stacked = np.stack(y_sequences, axis=0) # 形状: (N_sequences, seq_len, patch_len, C)

        # 如果是标签数据，则将最后一个维度压缩掉。
        if is_label_data:
            x_stacked = np.squeeze(x_stacked, axis=-1)
            y_stacked = np.squeeze(y_stacked, axis=-1)

        return x_stacked, y_stacked

    def _split_window_into_patches(self, window: np.ndarray) -> np.ndarray:
        """
        保留: 这是一个通用的辅助函数，未作更改。
        给定单个窗口，将其分割成补丁。

        Args:
            window: np.ndarray，形状为 (window_len, C)。

        Returns:
            patches: np.ndarray，形状为 (num_patches, patch_len, C)。
        """
        window_len, C = window.shape
        num_patches = (window_len - self.patch_len) // self.patch_stride + 1
        
        patches = np.zeros((num_patches, self.patch_len, C), dtype=window.dtype)
        for i in range(num_patches):
            start = i * self.patch_stride
            patches[i] = window[start : start + self.patch_len]

        return patches

    def _split_indices(self, total_sequences: int) -> tuple:
        """
        保留: 用于分割索引的通用工具，未作更改。
        """
        n_train = int(total_sequences * self.train_ratio)
        
        train_idx = (0, n_train)
        valid_idx = (n_train, total_sequences)
        return train_idx, valid_idx

    def extract_and_store_all(self, save_dir: str = "data/MSL/patches"):
        """
        修改后: 提取、分割并存储训练集和验证集。
        整个流程现在基于新的 `_create_sequences_from_data` 函数。
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) 加载原始训练数据
        # MODIFIED: Now uses the real DataTool to load and standardize
        self.data_tool.load()
        data = self.data_tool.standardize()
        
        if self.debug:
            print(f"\n=== 正在处理训练数据: {self.data_tool.path} ===")
            print(f"原始数据形状: {data.shape}")

        # 2) 从数据中创建所有(输入, 目标)序列对
        x_patches_all, y_patches_all = self._create_sequences_from_data(data)
        N, _, _, _ = x_patches_all.shape
        if self.debug:
            print(f"已创建 {N} 个序列对。")
            print(f"x_patches_all 形状: {x_patches_all.shape}")
            print(f"y_patches_all 形状: {y_patches_all.shape}")
        
        # 3) 确定训练/验证集的索引
        (t0, t1), (v0, v1) = self._split_indices(N)

        # 4) 根据索引切分补丁序列
        x_train, y_train = x_patches_all[t0:t1], y_patches_all[t0:t1]
        x_valid, y_valid = x_patches_all[v0:v1], y_patches_all[v0:v1]
        if self.debug:
            print(f"训练集 x/y 形状: {x_train.shape} / {y_train.shape}")
            print(f"验证集 x/y 形状: {x_valid.shape} / {y_valid.shape}")

        # 5) 将每个分割保存为.npz文件
        train_path = os.path.join(save_dir, "msl_train.npz")
        valid_path = os.path.join(save_dir, "msl_val.npz")

        np.savez_compressed(train_path, x_patches=x_train, y_patches=y_train)
        np.savez_compressed(valid_path, x_patches=x_valid, y_patches=y_valid)

        if self.debug:
            print(f"\n已保存训练补丁到: {train_path}")
            print(f"已保存验证补丁到: {valid_path}")
        
        return train_path, valid_path
        
    def process_and_split_test_set(
        self, 
        test_filename: str = DEFAULT_TEST_FILE, 
        label_filename: str = DEFAULT_TEST_LABEL_FILE, 
        save_dir: str = "data/MSL/patches"
    ):
        """
        新增与重构: 一个统一的函数来处理完整的测试流程：
        加载测试数据和标签，创建序列，进行分割，并保存所有部分。
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- 1. 处理测试特征数据 ---
        # MODIFIED: Use the real DataTool and reuse the scaler from training data
        test_data_tool = DataTool(test_filename, debug=self.debug)
        test_data_tool.load()
        # IMPORTANT: Use the scaler from the training data to transform test data
        test_data = self.data_tool.scaler.transform(test_data_tool.data)
        
        if self.debug:
            print(f"\n=== 正在处理测试数据: {test_filename} ===")
        x_test_all, y_test_all = self._create_sequences_from_data(test_data)
        N_test = x_test_all.shape[0]
        if self.debug:
            print(f"已创建 {N_test} 个测试序列对。")

        # --- 2. 处理测试标签数据 ---
        # Label data does not need standardization
        label_data_tool = DataTool(label_filename, debug=self.debug)
        raw_labels = label_data_tool.load()
        if self.debug:
            print(f"\n=== 正在处理标签数据: {label_filename} ===")
        # 使用完全相同的方法创建标签序列
        x_label_patches, y_label_patches = self._create_sequences_from_data(raw_labels)
        
        # 将每个标签补丁转换为单个二进制标签 (如果补丁内有任何异常，则为1)
        x_labels = np.any(x_label_patches, axis=2).astype(np.int32)
        y_labels = np.any(y_label_patches, axis=2).astype(np.int32)
        
        if self.debug:
            print(f"处理后的标签形状 x/y: {x_labels.shape} / {y_labels.shape}")
            print(f"x_labels中异常比例: {np.mean(x_labels):.4f}")

        # --- 3. 分割特征和标签 ---
        # 计算分割点
        final_test_size = int(N_test * FINAL_TEST_RATIO)
        tune_size = N_test - final_test_size
        val_size = int(tune_size * TUNE_VAL_RATIO)
        train_size = tune_size - val_size

        # 分割最终测试集
        x_final_test, y_final_test = x_test_all[-final_test_size:], y_test_all[-final_test_size:]
        x_labels_final_test, y_labels_final_test = x_labels[-final_test_size:], y_labels[-final_test_size:]

        # 分割调优集
        x_tune_all, y_tune_all = x_test_all[:-final_test_size], y_test_all[:-final_test_size]
        x_labels_tune_all, y_labels_tune_all = x_labels[:-final_test_size], y_labels[:-final_test_size]

        x_tune_train, y_tune_train = x_tune_all[:train_size], y_tune_all[:train_size]
        x_labels_tune_train, y_labels_tune_train = x_labels_tune_all[:train_size], y_labels_tune_all[:train_size]

        x_tune_val, y_tune_val = x_tune_all[train_size:], y_tune_all[train_size:]
        x_labels_tune_val, y_labels_tune_val = x_labels_tune_all[train_size:], y_labels_tune_all[train_size:]
        
        if self.debug:
            print("\n=== 测试集分割结果 ===")
            print(f"调优训练集大小: {x_tune_train.shape[0]}")
            print(f"调优验证集大小: {x_tune_val.shape[0]}")
            print(f"最终测试集大小: {x_final_test.shape[0]}")

        # --- 4. 保存所有分割 ---
        paths = {}
        # 保存特征
        paths['tune_train'] = os.path.join(save_dir, "msl_tune_train.npz")
        np.savez_compressed(paths['tune_train'], x_patches=x_tune_train, y_patches=y_tune_train)
        paths['tune_val'] = os.path.join(save_dir, "msl_tune_val.npz")
        np.savez_compressed(paths['tune_val'], x_patches=x_tune_val, y_patches=y_tune_val)
        paths['final_test'] = os.path.join(save_dir, "msl_final_test.npz")
        np.savez_compressed(paths['final_test'], x_patches=x_final_test, y_patches=y_final_test)
        # 保存标签
        paths['tune_train_labels'] = os.path.join(save_dir, "msl_tune_train_labels.npz")
        np.savez_compressed(paths['tune_train_labels'], x_labels=x_labels_tune_train, y_labels=y_labels_tune_train)
        paths['tune_val_labels'] = os.path.join(save_dir, "msl_tune_val_labels.npz")
        np.savez_compressed(paths['tune_val_labels'], x_labels=x_labels_tune_val, y_labels=y_labels_tune_val)
        paths['final_test_labels'] = os.path.join(save_dir, "msl_final_test_labels.npz")
        np.savez_compressed(paths['final_test_labels'], x_labels=x_labels_final_test, y_labels=y_labels_final_test)

        if self.debug:
            print("\n已保存所有测试集分割文件:")
            for name, path in paths.items():
                print(f"  - {name}: {path}")
        
        return paths

    @staticmethod
    def load_patch_split(split_path: str) -> tuple:
        """
        加载预先保存的 .npz 分割文件并返回 (x_patches, y_patches) numpy数组。
        """
        data = np.load(split_path)
        # 根据文件内容返回 patches 或 labels
        if "x_patches" in data:
            return data["x_patches"], data["y_patches"]
        elif "x_labels" in data:
            return data["x_labels"], data["y_labels"]
        else:
            raise KeyError("在 .npz 文件中未找到 'x_patches' 或 'x_labels'。")

    @staticmethod
    def verify_saved_patch_split(split_path: str):
        """
        打印已保存的 .npz 分割文件中数组的形状。
        """
        if not os.path.exists(split_path):
            print(f"*** 文件不存在: {split_path} ***")
            return
            
        data = np.load(split_path)
        print(f"*** {os.path.basename(split_path)} 的内容 ***")
        for key in data:
            print(f"  - {key}: {data[key].shape}")

if __name__ == "__main__":
    # 创建保存目录路径
    save_dir = "data/MSL/patches"
    
    # 创建提取器实例并处理数据
    extractor = AnomalyPatchExtractor(debug=True)
    
    # 1) 处理训练集和验证集
    print("\n================== 1. 处理训练和验证集 ==================")
    train_path, val_path = extractor.extract_and_store_all(save_dir=save_dir)
    AnomalyPatchExtractor.verify_saved_patch_split(train_path)
    AnomalyPatchExtractor.verify_saved_patch_split(val_path)
    
    # 2) 统一处理测试集（包括数据和标签的分割与保存）
    print("\n================== 2. 处理测试集和标签 ==================")
    saved_paths = extractor.process_and_split_test_set(save_dir=save_dir)
    
    # 3) 验证所有生成的文件
    print("\n================== 3. 验证所有已保存的文件 ==================")
    for name, path in saved_paths.items():
        if os.path.exists(path):
            AnomalyPatchExtractor.verify_saved_patch_split(path)
        else:
            print(f"*** 文件不存在: {path} ***")
