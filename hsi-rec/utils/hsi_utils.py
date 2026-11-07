"""
Утилиты для загрузки матриц камеры.

Модуль предоставляет функции для чтения:
- CCM (Color Correction Matrix) из camconst.json (формат JSONC);
- CSS (Camera Spectral Sensitivity) из camspec_database.txt.
"""

import json
import re
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np


# ==============================================================================
# Внутренние вспомогательные функции
# ==============================================================================

def _strip_json_comments(text: str) -> str:
    """
    Удаляет // комментарии и /* ... */ блочные комментарии из JSON-подобного текста.
    Нужно, потому что camconst.json — это JSON с комментариями (JSONC).
    """
    # Удаляем блочные комментарии /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Удаляем строчные комментарии // ... (но не внутри строк)
    text = re.sub(r'(?m)(?<![:\"])//.*$', "", text)
    # Убираем возможные висячие запятые перед } или ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _normalize(name: str) -> str:
    """Приводит имя камеры к нижнему регистру и убирает лишние пробелы."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ==============================================================================
# CCM: Color Correction Matrix из camconst.json
# ==============================================================================

def get_ccm(camera_name: str, camconst_path: str) -> Optional[np.ndarray]:
    """
    Извлекает CCM (dcraw_matrix) для указанной камеры из camconst.json.

    Параметры
    ----------
    camera_name : str
        Имя камеры, например "Canon EOS 6D Mark II".
    camconst_path : str
        Путь до файла camconst.json.

    Возвращает
    ----------
    np.ndarray формы (3, 3), dtype=float64
        Матрица цветовой коррекции (значения dcraw_matrix / 10000).
    None
        Если камера не найдена или у неё нет dcraw_matrix.
    """
    with open(camconst_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean = _strip_json_comments(raw_text)
    data = json.loads(clean)

    target = _normalize(camera_name)
    entries = data.get("camera_constants", [])

    for entry in entries:
        make_model = entry.get("make_model")
        if make_model is None:
            continue

        if isinstance(make_model, str):
            names = [make_model]
        else:
            names = make_model

        if any(_normalize(n) == target for n in names):
            matrix = entry.get("dcraw_matrix")
            if matrix is None:
                return None
            return np.array(matrix, dtype=np.float64).reshape(3, 3) / 10000.0

    return None


# ==============================================================================
# CSS: Camera Spectral Sensitivity из camspec_database.txt
# ==============================================================================

@lru_cache(maxsize=8)
def load_camspec(path: str) -> dict:
    """
    Читает camspec_database.txt.

    Возвращает
    ----------
    dict : name -> np.ndarray shape (3, 33)
        Каналы: [R, G, B]; 33 точки — 400..720 нм с шагом 10 нм.
    """
    cams = {}
    with open(path, "r") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        R = np.array([float(x) for x in lines[i + 1].split()])
        G = np.array([float(x) for x in lines[i + 2].split()])
        B = np.array([float(x) for x in lines[i + 3].split()])
        cams[name] = np.stack([R, G, B])   # (3, 33)
        i += 4
    return cams


def get_css(camspec_path: str, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает матрицу спектральной чувствительности камеры (CSS).

    Параметры
    ----------
    camspec_path : str
        Путь к файлу camspec_database.txt.
    camera_name : str
        Имя камеры, например 'Canon 60D' или 'Nikon D3X'.

    Возвращает
    ----------
    CSS : np.ndarray, shape (3, 33), dtype float64
        Спектральная чувствительность [R, G, B].
        Столбцы соответствуют длинам волн 400, 410, ..., 720 нм.
    waves_length : np.ndarray, shape (33,)
        Массив длин волн.
    """
    cams = load_camspec(camspec_path)
    if camera_name not in cams:
        available = ", ".join(sorted(cams))
        raise KeyError(
            f"Камера '{camera_name}' не найдена.\n"
            f"Доступные: {available}"
        )

    waves_length = np.arange(400, 721, 10)  # 33 точки
    return (cams[camera_name].copy(), waves_length)