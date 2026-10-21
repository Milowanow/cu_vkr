import os
import argparse

import numpy as np
from pymatreader import read_mat
from PIL import Image

from hsi_utils import get_ccm, get_css

#-------Логгирование--------------------
from log_setup import setup_logging
setup_logging()

import logging
logger = logging.getLogger("create_rgb")
#---------------------------------------

# ==============================================================================
# глобальные пармаметры
CAMERA_NAME_CSS = "Canon 1DMarkIII"
CAMSPEC_PATH = "camspec_database.txt"
DELTA_LAMBDA = 10.0  # нм
# ==============================================================================

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
    logger.info(f"Start: {source_mat}")
    logger.debug(f"Имя камеры: {CAMERA_NAME_CSS}")

    # --- Проверка входного .mat файла ---
    if not os.path.exists(source_mat):
        raise FileNotFoundError(f"Исходный .mat файл не найден: {source_mat}")
    if not os.path.isfile(source_mat):
        raise ValueError(f"Указанный путь не является файлом: {source_mat}")

    # --- Проверка наличия вспомогательных файлов ---
    if not os.path.exists(CAMSPEC_PATH):
        raise FileNotFoundError(f"Не найден файл матриц CSS: {CAMSPEC_PATH}")

    logger.debug(f"Загрузка данных из {source_mat}")

    try:
        data = read_mat(source_mat)
    except Exception as e:
        raise IOError(f"Не удалось прочитать .mat файл: {e}")

    #logger.debug(f"Ключи в {data}")
       
    if not all(k in data for k in ['rad', 'bands']):
        raise ValueError("В .mat файле отсутствуют необходимые ключи 'rad' и/или 'bands'")

    radiance = data['rad'].astype(np.float32)
    bands = np.asarray(data['bands']).ravel()

    logger.debug(f"radiance shape: {radiance.shape}")
    logger.debug(f"bands shape: {bands.shape}")
    logger.debug(f"bands min: {bands.min()}, bands max: {bands.max()}")

    # --- Загрузка и подготовка матриц камеры ---
    logger.debug(f"Загрузка матрицы CSS: {CAMERA_NAME_CSS}")
    CSS, waves_length_css = get_css(CAMSPEC_PATH, CAMERA_NAME_CSS)
    CSS.astype(np.float32)

    # --- Нужно чтобы CSS считалась правильно и не были перепутаны ожидаем:
    # пик красного: ~600 нм
    # пик зеленого: ~540 нм
    # пик синего:   ~450 нм
    for ch, name in enumerate("RGB"):
        peak = waves_length_css[np.argmax(CSS[ch])]
        logger.debug(f"{name}: пик на {peak} нм")


    # --- Согласование длин волн ---
    logger.debug(f"Согласование длин волн")
    mask = (waves_length_css >= 400) & (waves_length_css <= 700)
    waves_length_cut = waves_length_css[mask]
    CSS_cut = CSS[:, mask]

    if not np.allclose(waves_length_cut, bands):
        raise ValueError("Длины волн HSI и CSS не совпадают после обрезки.")

    # --- Свёртка radiance с CSS для получения RGB ---
    logger.debug("Свёртка radiance с CSS.")
    # (H, W, 31) x (31, 3) -> (H, W, 3)
    rgb = np.tensordot(radiance, CSS_cut.T, axes=([2], [0]))
    rgb = rgb * DELTA_LAMBDA
    logger.debug(f"RGB shape: {rgb.shape}, диапазон [{rgb.min():.2f}, {rgb.max():.2f}]")

    # --- Сохранение результата ---
    # Имя выходного файла = имя входного файла с расширением .raw
    base_name = os.path.splitext(os.path.basename(source_mat))[0]
    output_filename = f"{base_name}"

    # Создаём директорию, если её нет
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_filename)

    np.save(output_path, rgb.astype(np.float32))
    logger.info(f"Линейный RGB сохранён: {output_path}.npy")

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
        process_image(args.source_mat, args.save_path)
    except (FileNotFoundError, ValueError, IOError, KeyError) as e:
        logger.critical(f"Ошибка: {e}")
        exit(1)


if __name__ == "__main__":
    main()