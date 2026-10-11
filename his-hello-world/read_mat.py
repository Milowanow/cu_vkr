"""
Скрипт для чтения .mat файлов и вывода основной информации о содержимом.
Поддерживает форматы MATLAB v4, v6, v7 (scipy) и v7.3 (HDF5).
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Красивый вывод заголовка."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def format_size(nbytes):
    """Форматирование размера в человекочитаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def describe_value(name, value, indent="  "):
    """Вывод описания одной переменной."""
    import numpy as np
    
    dtype = type(value).__name__
    print(f"{indent}├─ Имя:        {name}")
    print(f"{indent}├─ Тип:        {dtype}")
    
    if isinstance(value, np.ndarray):
        print(f"{indent}├─ Форма:      {value.shape}")
        print(f"{indent}├─ Размерность:{value.ndim}")
        print(f"{indent}├─ dtype:      {value.dtype}")
        print(f"{indent}├─ Кол-во эл.: {value.size}")
        print(f"{indent}├─ Память:     {format_size(value.nbytes)}")
        
        # Дополнительная информация для числовых массивов
        if np.issubdtype(value.dtype, np.number) and value.size > 0:
            try:
                print(f"{indent}├─ Мин:        {np.nanmin(value)}")
                print(f"{indent}├─ Макс:       {np.nanmax(value)}")
                print(f"{indent}├─ Среднее:    {np.nanmean(value)}")
            except Exception:
                pass
        # Для строк / объектов
        elif value.dtype == object and value.size > 0:
            sample = value.flat[0]
            print(f"{indent}├─ Пример:     {repr(sample)[:80]}")
    
    elif isinstance(value, (str, bytes)):
        preview = value[:80] if len(value) > 80 else value
        print(f"{indent}├─ Длина:      {len(value)}")
        print(f"{indent}├─ Значение:   {repr(preview)}")
    
    elif isinstance(value, (int, float, complex)):
        print(f"{indent}├─ Значение:   {value}")
    
    elif isinstance(value, dict):
        print(f"{indent}├─ Кол-во ключей: {len(value)}")
        print(f"{indent}├─ Ключи:      {list(value.keys())[:20]}")
    
    elif isinstance(value, (list, tuple)):
        print(f"{indent}├─ Длина:      {len(value)}")
    
    # Атрибуты (для mat_struct из scipy)
    if hasattr(value, '_fieldnames'):
        print(f"{indent}├─ Поля:       {value._fieldnames}")
    
    print(f"{indent}└─")
    print()


def read_mat_scipy(filepath):
    """Чтение через scipy.io.loadmat (v4, v6, v7)."""
    from scipy.io import loadmat
    
    # struct_as_record=False — структуры как объекты с атрибутами
    # squeeze_me=True — убирает лишние размерности
    data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    return data


def read_mat_hdf5(filepath):
    """Чтение v7.3 через h5py."""
    import h5py
    import numpy as np
    
    data = {}
    with h5py.File(filepath, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    # Пытаемся прочитать как массив
                    arr = obj[()]
                    if isinstance(arr, np.ndarray):
                        # Транспонируем (HDF5 хранит в обратном порядке)
                        if arr.ndim >= 2:
                            arr = arr.T
                    data[name] = arr
                except Exception as e:
                    data[name] = f"<не прочитано: {e}>"
        f.visititems(visitor)
    return data


def inspect_mat(filepath):
    """Основная функция: определяет формат и выводит информацию."""
    
    if not os.path.exists(filepath):
        print(f"❌ Файл не найден: {filepath}")
        return False
    
    if not filepath.lower().endswith('.mat'):
        print(f"⚠️  Файл не имеет расширения .mat, но попробуем прочитать...")
    
    # Информация о файле
    print_header("ИНФОРМАЦИЯ О ФАЙЛЕ")
    stat = os.stat(filepath)
    print(f"  Путь:         {os.path.abspath(filepath)}")
    print(f"  Размер:       {format_size(stat.st_size)}")
    print(f"  Изменён:      {__import__('datetime').datetime.fromtimestamp(stat.st_mtime)}")
    
    # Определение версии по заголовку
    with open(filepath, 'rb') as f:
        header = f.read(128)
    
    version = "неизвестно"
    is_hdf5 = header[:8] == b'\x89HDF\r\n\x1a\n'
    
    if is_hdf5:
        version = "MATLAB v7.3 (HDF5)"
    elif header[:20].startswith(b'MATLAB 5.0 MAT-file'):
        # Проверяем на текст в конце заголовка
        header_str = header.decode('latin-1', errors='ignore')
        if 'MATLAB 5.0 MAT-file' in header_str:
            # Приблизительная версия
            version = "MATLAB v5/v6/v7 (Level 5)"
    
    print(f"  Формат:       {version}")
    print(f"  HDF5:         {'да' if is_hdf5 else 'нет'}")
    
    # Чтение данных
    print_header("ЧТЕНИЕ ДАННЫХ")
    
    try:
        if is_hdf5:
            try:
                data = read_mat_hdf5(filepath)
            except ImportError:
                print("❌ Для чтения v7.3 нужен пакет h5py:")
                print("   pip install h5py")
                return False
        else:
            data = read_mat_scipy(filepath)
    except NotImplementedError:
        print("⚠️  scipy не поддерживает этот формат, пробуем h5py...")
        try:
            data = read_mat_hdf5(filepath)
        except ImportError:
            print("❌ Требуется h5py: pip install h5py")
            return False
    except Exception as e:
        print(f"❌ Ошибка чтения: {e}")
        return False
    
    # Фильтрация служебных ключей scipy
    service_keys = {'__header__', '__version__', '__globals__'}
    user_vars = {k: v for k, v in data.items() if k not in service_keys}
    
    # Информация о MAT-заголовке
    if '__header__' in data:
        hdr = data['__header__']
        if isinstance(hdr, bytes):
            hdr = hdr.decode('latin-1', errors='ignore')
        print(f"  Заголовок:    {hdr.strip()}")
    if '__version__' in data:
        print(f"  Версия:       {data['__version__']}")
    if '__globals__' in data:
        print(f"  Globals:      {data['__globals__']}")
    
    # Переменные
    print_header(f"ПЕРЕМЕННЫЕ ({len(user_vars)} шт.)")
    
    if not user_vars:
        print("  (переменных не найдено)")
        return True
    
    # Сводная таблица
    print(f"\n  {'Имя':<25} {'Тип':<20} {'Форма/Размер'}")
    print(f"  {'-'*25} {'-'*20} {'-'*30}")
    
    import numpy as np
    for name, value in user_vars.items():
        vtype = type(value).__name__
        if isinstance(value, np.ndarray):
            shape = str(value.shape) + f" ({value.dtype})"
        elif hasattr(value, '_fieldnames'):
            shape = f"struct, поля: {list(value._fieldnames)}"
        else:
            shape = str(value)[:40]
        print(f"  {name:<25} {vtype:<20} {shape}")
    
    # Детально по каждой переменной
    print_header("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ")
    for name, value in user_vars.items():
        describe_value(name, value)
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Использование: python read_mat.py <файл.mat>")
        print("\nПример:")
        print("  python read_mat.py data.mat")
        print("  python read_mat.py /path/to/experiment.mat")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = inspect_mat(filepath)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()