import psutil, os, time, threading

from log_setup import setup_logging
setup_logging()


import logging
logger = logging.getLogger(__name__)

def _monitor(interval: float = 1.0) -> None:
    p = psutil.Process(os.getpid())
    while True:
        print(f"[mem] {p.memory_info().rss / 1024**2:.1f} MB", flush=True)
        logger.info(f"[mem] {p.memory_info().rss / 1024**2:.1f} MB")
        time.sleep(interval)



def start_memory_monitor(interval: float = 1.0) -> threading.Thread:
    """Запускает фоновый поток, логирующий RSS-память текущего процесса."""
    thread = threading.Thread(
        target=_monitor,
        kwargs={"interval": interval},
        daemon=True,
        name="memory-monitor",
    )
    thread.start()
    return thread