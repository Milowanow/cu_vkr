# lazy_dataset.py
import os
import subprocess
from pathlib import Path
import logging

import torch.utils.data as udata

logger = logging.getLogger(__name__)


class LazyDataset(udata.Dataset):
    """
    Базовый torch Dataset с ленивой подгрузкой файлов с FTP.

    Наследник:
      - заполняет self.records  (список чего угодно, по одному на __getitem__);
      - реализует _required_files(idx) -> list[(remote_path, local_path)];
      - реализует _load_sample(idx) -> то, что должен вернуть __getitem__.

    __getitem__ сначала гарантирует, что все файлы для idx скачаны,
    затем вызывает _load_sample(idx).
    """

    #: как запускать get_data.py; переопределяется в наследнике при необходимости
    download_cmd = ("uv", "run", "python", "get_data.py")
    #: рабочая директория, откуда запускать команду (корень проекта uv)
    project_dir = "."

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------
    # то, что реализует наследник
    # ------------------------------------------------------------------
    def _required_files(self, idx):
        """
        Вернуть список пар (remote_path, local_path) — что должно быть
        на диске к моменту чтения сэмпла idx.
        """
        raise NotImplementedError

    def _load_sample(self, idx):
        """Прочитать и вернуть сэмпл. Файлы уже гарантированно на диске."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # то, что делает база
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        for remote_path, local_path in self._required_files(idx):
            self._ensure_local(remote_path, Path(local_path))
        return self._load_sample(idx)

    def _ensure_local(self, remote_path: str, local_path: Path):
        if local_path.exists():
            return
    
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[lazy] download {remote_path} -> {local_path}")
        cmd = list(self.download_cmd) + [
            "--remote-path", remote_path,
            "--save-path", str(local_path.parent),
        ]
        subprocess.run(cmd, check=True, cwd=self.project_dir)
        if not local_path.exists():
            raise FileNotFoundError(
                f"get_data.py не создал {local_path} "
                f"(remote={remote_path}). Проверьте --save-path."
            )