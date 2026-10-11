import os
import glob
import numpy as np
import torch
import torch.utils.data as udata
import h5py
from PIL import Image

import logging
logger = logging.getLogger(__name__)

def get_icvl_hsi_shape(path):
    """Возвращает (C, W, H) без чтения самих данных."""
    with h5py.File(path, 'r') as f:
        if 'rad' not in f:
            raise KeyError(f"В {path} нет 'rad'. Есть: {list(f.keys())}")
        return f['rad'].shape


def load_icvl_hsi_patch(path, top, left, ps):
    """
    Читает только окно patch_size×patch_size из гиперкуба.
    Возвращает (C, ps, ps) float32.
    """
    with h5py.File(path, 'r') as f:
        patch = f['rad'][:, top:top+ps, left:left+ps]
    return np.asarray(patch, dtype=np.float32)


def load_raw_patch(path, top, left, ps):
    """
    Читает только окно из PNG и возвращает (3, ps, ps) float32.
    Соглашение осей: (C, W, H), поэтому кроп по (W, H).
    """
    img = Image.open(path)                         # ленивая загрузка
    arr = np.asarray(img)                          # (H, W, 3)
    arr = arr[left:left+ps, top:top+ps, :]         # (ps, ps, 3)
    arr = np.transpose(arr, (2, 0, 1))             # (3, ps, ps)
    return arr.astype(np.float32)

# ----------------------------------------------------------------------
# Датасет
# ----------------------------------------------------------------------
class ICVLDataset(udata.Dataset):
    """
    Пары (RAW, HSI).
    RAW (3, W, H)
    HSI (C, W, H)
    Ожидается, что имена файлов совпадают (без расширения) в двух директориях.
    """
    def __init__(self,
                 raw_dir,
                 hsi_dir,
                 raw_ext='.png',
                 train=True,
                 patch_size=None,
                 stride=None,
                 transform=None,          # для raw
                 target_transform=None):  # для hsi
        self.raw_dir = raw_dir
        self.hsi_dir = hsi_dir
        self.train = train
        
        self.patch_size = patch_size
        # если stride не задан — берём без перекрытия
        self.stride = stride if stride is not None else patch_size
        
        self.transform = transform
        self.target_transform = target_transform

        self.pairs = self._build_pairs(raw_dir, hsi_dir, raw_ext)
        self.patches = self._build_patches()

        logger.info(f"[ICVLDataset] пар: {len(self.pairs)}, "
                    f"патчей: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        pair_idx, top, left = self.patches[index]
        raw_path, hsi_path = self.pairs[pair_idx]
        ps = self.patch_size

        raw = load_raw_patch(raw_path, top, left, ps)         # (3, ps, ps)
        hsi = load_icvl_hsi_patch(hsi_path, top, left, ps)    # (C, ps, ps)

        if self.transform is not None:
            raw = self.transform(raw)
        if self.target_transform is not None:
            hsi = self.target_transform(hsi)

        return raw, hsi

    #
    # helpers
    #
    def _build_pairs(self, raw_dir, hsi_dir, raw_ext):
        """Собирает пары (raw_path, hsi_path) по совпадению stem имени."""
        raw_paths = sorted(glob.glob(os.path.join(raw_dir, f'*{raw_ext}')))
        pairs = []
        for rp in raw_paths:
            stem = os.path.splitext(os.path.basename(rp))[0]
            hp = os.path.join(hsi_dir, stem + ".mat")
            if os.path.exists(hp):
                pairs.append((rp, hp))
            else:
                logger.warning(f"нет пары для {rp}")

        if not pairs:
            raise RuntimeError("Не найдено ни одной пары RAW-HSI!")
        return pairs

    def _build_patches(self):
        """
        Строит плоский список (pair_idx, top, left) для всех патчей.
        Размеры (W, H) берутся из метаданных первого .mat, без чтения данных.
        """
        _, W, H = get_icvl_hsi_shape(self.pairs[0][1])
        ps, st = self.patch_size, self.stride
        self._check_image_size(W, H, ps)

        tops  = self._axis_offsets(W, ps, st)
        lefts = self._axis_offsets(H, ps, st)

        return [
            (pair_idx, top, left)
            for pair_idx in range(len(self.pairs))
            for top in tops
            for left in lefts
        ]

    @staticmethod
    def _check_image_size(W, H, ps):
        if W < ps or H < ps:
            raise ValueError(
                f"Изображение {W}×{H} меньше патча {ps}×{ps}. "
                f"Уменьшите patch_size или добавьте паддинг."
            )

    @staticmethod
    def _axis_offsets(size, ps, st):
        """
        Координаты начала патчей по одной оси:
        0, st, 2*st, ..., плюс гарантированно последняя позиция size - ps.
        """
        offsets = list(range(0, size - ps + 1, st))
        if offsets[-1] != size - ps:
            offsets.append(size - ps)
        return offsets
        

##
## Transforms
##

def normalize_rgb(x: np.ndarray) -> np.ndarray:
    return (x / 255).astype(np.float32)


def normalize_icvl_hsi(x: np.ndarray) -> np.ndarray:
    """
    (H, W, C) float32, значения ~[185, 3145] -> (H, W, C) float32 в [0, 1].
    Делим на глобальный max, чтобы шкала была одинаковой для всех кубов
    (per-image нормировка ломает сравнение сцен между собой).
    """
    ICVL_RAD_MAX = 3145.5
    return (x / ICVL_RAD_MAX).astype(np.float32)
#№# (H, W, C) -> (C, H, W)
#HWC_to_CHW_transform = transforms.Compose([
#    transforms.Lambda(lambda x: torch.from_numpy(np.ascontiguousarray(x))
#                                .permute(2, 0, 1)
#                                .float()),
#])

