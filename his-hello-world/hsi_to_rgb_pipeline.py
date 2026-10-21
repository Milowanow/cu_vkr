#!/usr/bin/env python3
"""
читает список файлов из текстового файла,
для каждого скачивает его через get_data.py и запускает create_rgb.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path


#-------Логгирование--------------------
from log_setup import setup_logging
setup_logging()

import logging
logger = logging.getLogger("hsi_to_rgb_pipeline")
#---------------------------------------

def run(cmd: list[str]) -> None:
    """Запускает команду и проверяет код возврата."""
    logger.debug(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Команда завершилась с кодом {result.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Пакетная загрузка и обработка .mat файлов")
    parser.add_argument(
        "--list-file",
        required=True,
        help="Текстовый файл со списком путей (по одному в строке), например NTIRE/ARAD_1K_0001.mat",
    )
    parser.add_argument(
        "--download-dir",
        default="./downloaded",
        help="Куда сохранять скачанные файлы (по умолчанию ./downloaded)",
    )
    parser.add_argument(
        "--rgb-dir",
        default="./rgb",
        help="Куда сохранять обработанные файлы (по умолчанию ./rgb)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Пропускать файлы, если результат обработки уже существует",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Индекс строки в списке, с которой начать (для докачки)",
    )
    args = parser.parse_args()

    list_path = Path(args.list_file)
    if not list_path.is_file():
        print(f"Файл со списком не найден: {list_path}", file=sys.stderr)
        return 1

    download_dir = Path(args.download_dir)
    rgb_dir = Path(args.rgb_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # Читаем список, убираем пустые строки и пробелы
    with list_path.open("r", encoding="utf-8") as f:
        remote_paths = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    total = len(remote_paths)
    logger.info(f"Всего файлов в списке: {total}")

    for idx, remote_path in enumerate(remote_paths):
        if idx < args.start:
            continue

        filename = Path(remote_path).name  # например ulm_0328-1118.mat
        local_mat = download_dir / filename
        processed_name = Path(filename).stem + ".npy"  # подставьте своё расширение, если нужно
        processed_path = rgb_dir / processed_name

        logger.info(f"[{idx + 1}/{total}] {remote_path}")

        if args.skip_existing and processed_path.exists():
            logger.info(f"Пропуск — результат уже есть: {processed_path}")
            continue

        # 1) Скачивание
        try:
            run([
                "uv", "run", "python", "get_data.py",
                "--remote-path", remote_path,
                "--save-path", str(download_dir),
            ])
        except RuntimeError as e:
            logger.error(f"Ошибка при скачивании {remote_path}: {e}")
            continue

        if not local_mat.exists():
            logger.error(f"Внимание: скачанный файл не найден: {local_mat}")
            continue

        # 2) Обработка
        try:
            run([
                "uv", "run", "python", "create_rgb.py",
                "--source-mat", str(local_mat),
                "--save-path", str(rgb_dir),
            ])
        except RuntimeError as e:
            logger.error(f"Ошибка при обработке {local_mat}: {e}")
            continue

    logger.info(f"Готово")
    return 0


if __name__ == "__main__":
    sys.exit(main())


    