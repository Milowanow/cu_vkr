import logging.config
from pathlib import Path

# Папка для логов рядом со скриптом (или можно взять из env)
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": "DEBUG",
            "filename": str(LOG_FILE),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
            "delay": True,
        },
    },
    "loggers": {
        # Всё, что начинается с PIL.*, будет иметь эффективный уровень WARNING
        "PIL": {"level": "WARNING", "propagate": True},

        # Заодно — другие болтливые библиотеки
        "matplotlib": {"level": "WARNING"},
        "matplotlib.font_manager": {"level": "WARNING"},
        "urllib3": {"level": "WARNING"},
        "h5py": {"level": "WARNING"},
        "tensorboard": {"level": "WARNING"},
        "torch.utils.tensorboard": {"level": "WARNING"},
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING)