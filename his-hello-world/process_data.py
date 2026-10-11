import os
import argparse

import numpy as np
from pymatreader import read_mat
from PIL import Image

from hsi_utils import get_ccm, get_css


# ==============================================================================
# Основная логика обработки
# ==============================================================================

def srgb_gamma(x: np.ndarray) -> np.ndarray:
    """Гамма-коррекция sRGB."""
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(np.clip(x, 0, None), 1/2.4) - 0.055
    )


def process_image(source_mat: str, save_dir: str) -> str:
    """
    Основной пайплайн обработки HSI-изображения.

    Параметры
    ----------
    source_mat : str
        Путь до исходного .mat файла.
    save_dir : str
        Директория для сохранения результата. Будет создана, если не существует.
        Имя выходного файла совпадает с именем входного, но с расширением .png.

    Возвращает
    ----------
    str
        Путь к сохранённому PNG-файлу.
    """
    # --- Параметры ---
    CAMERA_NAME_CSS = "Canon 1DMarkIII"
    CAMERA_NAME_CCM = "Canon EOS 6D Mark II"
    CAMCONST_PATH = "camconst.json"
    CAMSPEC_PATH = "camspec_database.txt"
    DELTA_LAMBDA = 10.0  # нм

    # --- Проверка входного .mat файла ---
    if not os.path.exists(source_mat):
        raise FileNotFoundError(f"Исходный .mat файл не найден: {source_mat}")
    if not os.path.isfile(source_mat):
        raise ValueError(f"Указанный путь не является файлом: {source_mat}")

    # --- Проверка наличия вспомогательных файлов ---
    if not os.path.exists(CAMCONST_PATH):
        raise FileNotFoundError(f"Файл не найден: {CAMCONST_PATH}")
    if not os.path.exists(CAMSPEC_PATH):
        raise FileNotFoundError(f"Файл не найден: {CAMSPEC_PATH}")

    # --- Формирование пути для сохранения ---
    # Имя выходного файла = имя входного файла с расширением .png
    base_name = os.path.splitext(os.path.basename(source_mat))[0]
    output_filename = f"{base_name}.png"

    # Создаём директорию, если её нет
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_filename)

    print(f"Загрузка данных из {source_mat}...")

    try:
        data = read_mat(source_mat)
    except Exception as e:
        raise IOError(f"Не удалось прочитать .mat файл: {e}")

    print("Ключи в mat файле")
    for k in data:
        print(k)

    if not all(k in data for k in ['rad', 'bands']):
        raise ValueError("В .mat файле отсутствуют необходимые ключи 'rad' и/или 'bands'")

    radiance = data['rad']
    bands = np.asarray(data['bands']).ravel()

    print(f"  - radiance: {radiance.shape}")
    print(f"  - bands: {bands.shape} (от {bands.min()} до {bands.max()} нм)")

    # --- Загрузка и подготовка матриц камеры ---
    print("Загрузка матриц камеры...")
    CSS, waves_length_css = get_css(CAMSPEC_PATH, CAMERA_NAME_CSS)
    CCM = get_ccm(CAMERA_NAME_CCM, CAMCONST_PATH)

    if CCM is None:
        raise ValueError(f"Не удалось получить CCM для камеры {CAMERA_NAME_CCM}")
    print(f"  - CCM: {CCM.shape}")
    print(f"  - CSS: {CSS.shape}")

    # Нормировка CCM по строкам
    row_sums = CCM.sum(axis=1, keepdims=True)
    CCM = CCM / row_sums

    # --- Согласование длин волн ---
    print("Согласование длин волн...")
    mask = (waves_length_css >= 400) & (waves_length_css <= 700)
    waves_length_cut = waves_length_css[mask]
    CSS_cut = CSS[:, mask]

    if not np.allclose(waves_length_cut, bands):
        raise ValueError("Длины волн HSI и CSS не совпадают после обрезки.")

    print(f"  - Согласованные длины волн: {waves_length_cut.shape} точек")

    # --- Свёртка radiance с CSS для получения RAW ---
    print("Свёртка radiance с CSS...")
    # (H, W, 31) x (31, 3) -> (H, W, 3)
    raw = np.tensordot(radiance, CSS_cut.T, axes=([2], [0]))
    raw = raw * DELTA_LAMBDA
    print(f"  - RAW изображение: {raw.shape}, диапазон [{raw.min():.2f}, {raw.max():.2f}]")

    # --- Нормировка RAW ---
    # Нормировка на максимум зелёного канала
    raw_norm = raw / (raw[..., 1].max() + 1e-12)
    print(f"  - Нормированный RAW: диапазон [{raw_norm.min():.4f}, {raw_norm.max():.4f}]")

    # --- Применение CCM для получения линейного sRGB ---
    print("Применение CCM...")
    rgb_lin = raw_norm @ CCM.T
    rgb_lin = np.clip(rgb_lin, 0, 1)
    print(f"  - Линейный sRGB: {rgb_lin.shape}, диапазон [{rgb_lin.min():.4f}, {rgb_lin.max():.4f}]")

    # --- Гамма-коррекция ---
    print("Гамма-коррекция sRGB...")
    rgb_srgb = srgb_gamma(rgb_lin)
    print(f"  - sRGB: диапазон [{rgb_srgb.min():.4f}, {rgb_srgb.max():.4f}]")

    # --- Сохранение результата ---
    print(f"Сохранение результата в {output_path}...")
    rgb_8bit = np.round(255 * rgb_srgb).astype(np.uint8)

    try:
        output_image = Image.fromarray(rgb_8bit)
        output_image.save(output_path)
        print("Сохранение успешно завершено.")
    except Exception as e:
        raise IOError(f"Не удалось сохранить изображение: {e}")

    return output_path


# ==============================================================================
# Точка входа
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Обработка HSI-изображения из .mat файла и сохранение в sRGB PNG. "
            "Имя выходного файла совпадает с именем входного (меняется только расширение)."
        )
    )
    parser.add_argument(
        "--source-mat",
        required=True,
        help="Путь до исходного .mat файла с данными."
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help=(
            "Директория для сохранения итогового .png файла. "
            "Будет создана, если не существует. "
            "Имя файла берётся из имени входного .mat файла."
        )
    )

    args = parser.parse_args()

    try:
        output_path = process_image(args.source_mat, args.save_path)
        print(f"\nГотово. Результат: {output_path}")
    except (FileNotFoundError, ValueError, IOError, KeyError) as e:
        print(f"\nОшибка: {e}")
        exit(1)


if __name__ == "__main__":
    main()