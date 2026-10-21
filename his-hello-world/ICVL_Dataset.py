import os
import glob
import numpy as np
import torch
import torch.utils.data as udata
import h5py
from PIL import Image

from lazy_dataset import LazyDataset

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


def load_rgb_patch(path, top, left, ps):
    arr = np.load(path)                            # (H, W, 3) npy
    arr = arr[left:left+ps, top:top+ps, :]
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


# ----------------------------------------------------------------------
# Датасет
# ----------------------------------------------------------------------
class ICVLDataset(LazyDataset):
    """
        Пары (HSI .mat, RAW .npy).
    
        - file_list: удалённые пути на FTP, ['ICVL_our/4cam_0411-1640-1.mat', ...]
        - hsi_local_dir: директория с ГСИ файлами
        - rgb_local_dir: директория с RGB файлами

        Патчи:
        - patch_size=None  -> один сэмпл на файл (весь кадр);
        - patch_size=N     -> патчи N×N со страйдом stride (по умолчанию N),
                            включая гарантированный последний по каждой оси.
    """
    def __init__(self,
                 file_list,
                 hsi_local_dir='./downloaded/ICVL',
                 rgb_local_dir='./rgb',
                 rgb_ext='.npy',
                 patch_size=None,
                 train=True,
                 stride=None,
                 transform=None,
                 target_transform=None,
                 project_dir="."):
        logger.debug("Директория ГСИ: {hsi_local_dir}")
        logger.debug("Директория RGB: {hsi_local_dir}")

        self.hsi_local_dir = hsi_local_dir 
        self.rgb_local_dir = rgb_local_dir
        self.rgb_ext = rgb_ext
        self.hsi_ext = ".mat"
        self.train = train

        #Патчи
        self.patch_size = patch_size
        if patch_size is None:
            self.stride = None
        else:
            self.stride = stride if stride is not None else patch_size

        #Трансфореры
        self.transform = transform
        self.target_transform = target_transform
        self.project_dir = project_dir

        self.remote_paths = file_list

        # items — список единиц работы датасета
        self.items = self._build_items()

        logger.info(f"[ICVLDataset] файлов: {len(self.remote_paths)}, "
                    f"записей: {len(self.items)}")

    # ------------------------------------------------------------------
    # пути
    # ------------------------------------------------------------------
    def _stem(self, remote_path: str) -> str:
        """
            Возвращает «чистое» имя файла без директорий и без расширения.

            Пример:
                'ICVL_our/BGU_0522-1136.mat'  ->  'BGU_0522-1136'
            """
        return os.path.splitext(os.path.basename(remote_path))[0]

    def _local_hsi_path(self, remote_path: str) -> str:
        """
            По удалённому пути к HSI на FTP возвращает локальный путь,
            по которому этот .mat лежит (или должен лежать) после скачивания.

            Пример:
                remote: 'ICVL_our/BGU_0522-1136.mat' -> './downloaded/ICVL/BGU_0522-1136.mat'
        """
        return os.path.join(self.hsi_local_dir,
                            self._stem(remote_path) + self.hsi_ext)

    def _local_rgb_path(self, remote_path: str) -> str:
        """
            По удалённому пути к HSI возвращает локальный путь к соответствующему
            RGB-файлу (.npy). RGB берётся по тому же stem, что и .mat.

            Пример:
                remote: 'ICVL_our/BGU_0522-1136.mat' -> './rgb/BGU_0522-1136.npy'
        """
        return os.path.join(self.rgb_local_dir,
                            self._stem(remote_path) + self.rgb_ext)

    # ------------------------------------------------------------------
    # LazyDownloadDataset
    # ------------------------------------------------------------------
    def _required_files(self, idx):
        """
        Возвращает оба файла: .mat (HSI) и .npy (RGB).
        База проверит наличие и докачает отсутствующие.
        """
        item = self.items[idx]
        remote = self.remote_paths[item["file_idx"]]

        hsi_local = self._local_hsi_path(remote)
        rgb_local = self._local_rgb_path(remote)

        rgb_remote = self._stem(remote) + self.rgb_ext

        return [
            (remote,     hsi_local),
            (rgb_remote, rgb_local),
        ]

    def _load_sample(self, idx):
        """
            Читает и возвращает один сэмпл (rgb, hsi) для индекса idx.

            Вызывается из LazyDownloadDataset.__getitem__ 

            Что делает по шагам:
            1. По idx достаёт record и удалённый путь к .mat.
            2. Переводит удалённый путь в локальные пути:
                - .mat (HSI)  -> ./downloaded/ICVL/<stem>.mat
                - .npy (RGB)  -> ./rgb/<stem>.npy
            3. Берёт окно patch_size×patch_size (сейчас top=left=0 —
                заглушка под случай «один патч на файл»).
            4. Читает из .npy RGB-патч (3, ps, ps) и из .mat HSI-патч (C, ps, ps).
            5. Применяет transform / target_transform, если они заданы.
            6. Возвращает пару (raw, hsi) — то, что увидит DataLoader.

            Ожидаемые формы тензоров (до transform):
                raw: (3, ps, ps) float32
                hsi: (C, ps, ps) float32

        """
        item = self.items[idx]
        remote = self.remote_paths[item["file_idx"]]

        hsi_path = self._local_hsi_path(remote)
        rgb_path = self._local_rgb_path(remote)

        if self.patch_size is None:
            # без патчей: читаем весь кадр. ps = размер кадра по каждой оси —
            # берём из HSI-метаданных, чтобы не открывать .npy зря.
            _, W, H = get_icvl_hsi_shape(hsi_path)
            top, left, ps_w, ps_h = 0, 0, W, H
            rgb = load_rgb_patch(rgb_path, top, left, ps_w)  # (3, W, H)
            hsi = load_icvl_hsi_patch(hsi_path, top, left, ps_w)
        else:
            top  = item["top"]
            left = item["left"]
            ps   = self.patch_size
            rgb  = load_rgb_patch(rgb_path, top, left, ps)
            hsi  = load_icvl_hsi_patch(hsi_path, top, left, ps)

        if self.transform is not None:
            rgb = self.transform(rgb)
        if self.target_transform is not None:
            hsi = self.target_transform(hsi)
        return rgb, hsi


    def _build_items(self):
        """
            Строит self.items — плоский список «единиц работы» датасета.
            len(self.items) == len(self), поэтому DataLoader индексирует именно его.

            items — это лёгкие словари-адреса, а не данные. Каждый элемент говорит,
            ЧТО прочитать для сэмпла с данным idx, но не читает это сам:
                - {"file_idx": k}              — весь k-й файл (патчи выключены);
                - {"file_idx": k, "top": t, "left": l}    — окно patch_size×patch_size в k-м файле на позиции (t, l).

            Поведение зависит от self.patch_size:

            Раскладка патчей:
                tops  = _axis_offsets(W, ps, st)  — начала окон по первой оси;
                lefts = _axis_offsets(H, ps, st)  — по второй.

            Возвращаемое значение:
                Список словарей. Для каждого файла i и каждой пары (top, left)
                из tops×lefts — один элемент. Порядок детерминированный:
                сначала все патчи файла 0, затем файла 1 и т.д.
                Это позволяет воспроизводимо перемешивать индексы и,
                при желании, строить срезы вроде «только первые патчи каждого файла».

            Пример:
                files = 3, W=H=512, ps=64, st=64
                -> tops = lefts = [0, 64, ..., 448]  (8 значений)
                -> len(items) = 3 * 8 * 8 = 192 сэмпла.
        """
        if self.patch_size is None:
            # один сэмпл на файл
            return [{"file_idx": i} for i in range(len(self.remote_paths))]

        # для патчей нужны размеры куба: скачиваем метаданные первого файла.
        # Это единственное место, где мы «трогаем» файл в __init__.
        from pathlib import Path
        first_remote = self.remote_paths[0]
        first_local = Path(self._local_hsi_path(first_remote))
        self._ensure_local(first_remote, first_local)

        _, W, H = get_icvl_hsi_shape(str(first_local))
        ps, st = self.patch_size, self.stride
        self._check_image_size(W, H, ps)

        tops  = self._axis_offsets(W, ps, st)
        lefts = self._axis_offsets(H, ps, st)

        return [
            {"file_idx": i, "top": t, "left": l}
            for i in range(len(self.remote_paths))
            for t in tops
            for l in lefts
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

