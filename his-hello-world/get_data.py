import argparse
import json
import os
import sys
from urllib.parse import quote

import requests


def load_credentials(cred_path: str) -> dict:
    """Загружает учётные данные из JSON-файла."""
    if not os.path.isfile(cred_path):
        print(f"Ошибка: файл с учётными данными не найден: {cred_path}")
        sys.exit(1)

    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Ошибка чтения {cred_path}: {e}")
        sys.exit(1)

    sftp = data.get("sftp")
    if not sftp:
        print(f"Ошибка: в {cred_path} отсутствует секция 'sftp'")
        sys.exit(1)

    for key in ("url", "username", "password"):
        if not sftp.get(key):
            print(f"Ошибка: в секции 'sftp' отсутствует поле '{key}'")
            sys.exit(1)

    return sftp


def get_user_token(base_url: str, username: str, password: str) -> str:
    """Получает JWT-токен пользователя через REST API SFTPGo."""
    token_url = f"{base_url}/api/v2/user/token"

    try:
        response = requests.get(token_url, auth=(username, password), timeout=30)
        response.raise_for_status()
        token = response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении токена: {e}")
        sys.exit(1)

    if not token:
        print("Ошибка: токен не найден в ответе сервера.")
        sys.exit(1)

    print("Токен успешно получен.")
    return token


def download_file(
    base_url: str,
    token: str,
    remote_path: str,
    save_dir: str,
) -> None:
    """Скачивает файл из SFTPGo и сохраняет его в указанную директорию."""

    # 3. Если директории нет — создаём её (включая промежуточные)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 4. Локальное имя файла = имя файла из remote_path (без ведущего слэша)
    local_filename = os.path.basename(remote_path.rstrip("/"))
    if not local_filename:
        print(f"Ошибка: не удалось определить имя файла из пути '{remote_path}'")
        sys.exit(1)

    local_path = os.path.join(save_dir, local_filename) if save_dir else local_filename

    # Кодируем путь (ведущий слэш лучше убрать — см. предыдущее обсуждение)
    normalized_remote = remote_path.lstrip("/")
    encoded_path = quote(normalized_remote, safe="")

    download_url = f"{base_url}/api/v2/user/files?path={encoded_path}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        print(f"Скачивание файла: {remote_path} -> {os.path.abspath(local_path)}")
        with requests.get(
            download_url, headers=headers, stream=True, timeout=60
        ) as file_response:
            file_response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"Файл успешно сохранён: {os.path.abspath(local_path)}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при скачивании файла: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Ошибка при записи файла: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Скачивание файла из SFTPGo WebClient через REST API."
    )
    parser.add_argument(
        "--cred",
        default="cred.json",
        help="Путь к JSON-файлу с учётными данными (по умолчанию: cred.json)",
    )
    parser.add_argument(
        "--remote-path",
        required=True,
        help="Путь к файлу на сервере SFTPGo (например: ICVM_our/ulm_0328-1118.mat)",
    )
    parser.add_argument(
        "--save-path",
        default=".",
        help="Локальная директория для сохранения файла (по умолчанию: текущая)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    creds = load_credentials(args.cred)
    base_url = creds["url"].rstrip("/")

    token = get_user_token(base_url, creds["username"], creds["password"])
    download_file(
        base_url=base_url,
        token=token,
        remote_path=args.remote_path,
        save_dir=args.save_path,
    )


if __name__ == "__main__":
    main()