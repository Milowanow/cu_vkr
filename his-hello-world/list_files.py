import argparse
import json
import os
import sys
from urllib.parse import quote

import requests


def load_credentials(cred_path: str) -> dict:
    """Загружает учётные данные из JSON-файла."""
    if not os.path.isfile(cred_path):
        print(f"Ошибка: файл с учётными данными не найден: {cred_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Ошибка чтения {cred_path}: {e}", file=sys.stderr)
        sys.exit(1)

    sftp = data.get("sftp")
    if not sftp:
        print(f"Ошибка: в {cred_path} отсутствует секция 'sftp'", file=sys.stderr)
        sys.exit(1)

    for key in ("url", "username", "password"):
        if not sftp.get(key):
            print(f"Ошибка: в секции 'sftp' отсутствует поле '{key}'", file=sys.stderr)
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
        print(f"Ошибка при получении токена: {e}", file=sys.stderr)
        sys.exit(1)

    if not token:
        print("Ошибка: токен не найден в ответе сервера.", file=sys.stderr)
        sys.exit(1)

    return token


def list_files(base_url: str, token: str, remote_dir: str) -> None:
    """Выводит полные пути только файлов (без поддиректорий) в указанной директории."""
    normalized_dir = remote_dir.strip("/")
    encoded_path = quote(normalized_dir, safe="")

    list_url = f"{base_url}/api/v2/user/dirs"
    if encoded_path:
        list_url += f"?path={encoded_path}"

    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(list_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении списка файлов: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("Ошибка: неожиданный формат ответа сервера.", file=sys.stderr)
        sys.exit(1)

    # Префикс для полного пути от корня
    prefix = f"{normalized_dir}/" if normalized_dir else ""

    files = []
    for item in data:
        mode = item.get("mode", 0)
        is_dir = (mode & 0o40000) != 0
        if is_dir:
            continue
        name = item.get("name", "")
        if name:
            files.append(prefix + name)

    files.sort()
    for path in files:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Список файлов в директории SFTPGo WebClient через REST API."
    )
    parser.add_argument(
        "--cred",
        default="cred.json",
        help="Путь к JSON-файлу с учётными данными (по умолчанию: cred.json)",
    )
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Путь к директории на сервере SFTPGo (например: ICVM_our)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    creds = load_credentials(args.cred)
    base_url = creds["url"].rstrip("/")

    token = get_user_token(base_url, creds["username"], creds["password"])
    list_files(base_url, token, args.remote_dir)


if __name__ == "__main__":
    main()