import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import mlflow
import mlflow.pytorch

from ICVL_Dataset import ICVLDataset, normalize_icvl_hsi, normalize_rgb

from AWAN.AWAN import AWAN
from loss import MRAELoss

#логирование
from log_setup import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("s3transfer").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


from memory_logger import start_memory_monitor
#----------------------------------------------------------------------------




os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_access"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_secret"

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#tracking_uri = f"http://localhost:{5000}"
tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"device:, {device}")


hsi_transform = transforms.Compose([
    transforms.Lambda(normalize_icvl_hsi),
])

rgb_transform = transforms.Compose([
    transforms.Lambda(normalize_rgb),
])


def read_lines_from_file(filename: str) -> list[str]:
    """
    Читает текстовый файл и возвращает список строк (без символов перевода строки).
    
    Args:
        filename: путь к текстовому файлу.
    
    Returns:
        Список строк из файла.
    """
    with open(filename, "r", encoding="utf-8") as f:
        # strip() убирает '\n' и '\r' в конце строки
        return [line.strip() for line in f if line.strip()]
    
file_list = read_lines_from_file("ICVL_our_files.txt")
logger.info(f"Dataset ICVL: {len(file_list)} файлов")

train_ds = ICVLDataset(
    file_list=file_list,
    hsi_local_dir="./downloaded/ICVL",
    rgb_local_dir="../rgb",
    patch_size=64,
    train=True,
    stride=32,
    transform=rgb_transform,
    target_transform=hsi_transform
)
raw, hsi = train_ds[0]
logger.info(f"RAW: {raw.shape}, {raw.dtype}, {raw.min().item()}, {raw.max().item()}")
logger.info(f"HSI: {hsi.shape}, {hsi.dtype}, {hsi.min().item()}, {hsi.max().item()}")

train_loader = DataLoader(dataset = train_ds, batch_size=2, shuffle=False)
model = AWAN(inplanes=3, planes=31, channels=200, n_DRBs=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
criterion = MRAELoss()
writer = SummaryWriter(log_dir="./runs/awan_exp1")

RESUME_RUN_ID = None  # Например: "a1b2c3d4e5f6..."
RESUME_CHECKPOINT_PATH = None



if RESUME_RUN_ID:
    # Пытаемся скачать последний чекпоинт из артефактов MLflow
    try:
        # Загружаем артефакт с именем "checkpoint.pth" из указанного run
        local_path = mlflow.artifacts.download_artifacts(
            run_id=RESUME_RUN_ID,
            artifact_path="checkpoints/checkpoint.pth",
            dst_path=CHECKPOINT_DIR
        )
        RESUME_CHECKPOINT_PATH = local_path
        logger.info(f"Чекпоинт найден в MLflow: {local_path}")
    except Exception as e:
        logger.error(f"Не удалось загрузить чекпоинт из MLflow: {e}")

start_epoch = 0
if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
    checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Обучение будет продолжено с эпохи {start_epoch}")



# ==========================================
# 4. Цикл обучения с логированием в MLflow
# ==========================================
# Открываем run. Если возобновляем — используем тот же run_id, чтобы графики
# продолжились, а не начались с нуля.
start_memory_monitor()

with mlflow.start_run(run_id=RESUME_RUN_ID) as run:
    run_id = run.info.run_id
    logger.info(f"MLflow Run ID: {run_id}")

    # Логируем параметры только если это новый запуск
    if start_epoch == 0:
        mlflow.log_params({
            "learning_rate": 1e-4,
            "batch_size": 2,
            "patch_size": 64,
            "n_DRBs": 8,
            "optimizer": "Adam",
            "criterion": "MRAELoss"
        })

    num_epochs = 50

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (rgb, hsi) in enumerate(train_loader):
            logger.info(f"Epoch {epoch+1}/{num_epochs}, batch {batch_idx}/{len(train_loader)}")

            rgb, hsi = rgb.to(device), hsi.to(device)
            
            optimizer.zero_grad()
            output = model(rgb)
            loss = criterion(output, hsi)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Логируем loss на каждом шаге (опционально, можно раз в N батчей)
            # step помогает строить график по времени/итерациям
            global_step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

        # Средний loss за эпоху
        avg_loss = running_loss / len(train_loader)
        
        # Логируем метрики эпохи
        mlflow.log_metrics({
            "train_loss_epoch": avg_loss,
            "epoch": epoch
        }, step=epoch)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # ==========================================
        # Сохранение чекпоинта (каждую эпоху)
        # ==========================================
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        
        # Сохраняем локально
        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
        local_ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
        torch.save(checkpoint, local_ckpt_path)
        
        # Загружаем в MLflow как артефакт.
        # Используем одинаковое имя "checkpoint.pth", чтобы при resume
        # всегда скачивать последний (MLflow перезаписывает артефакт).
        mlflow.log_artifact(local_ckpt_path, artifact_path="checkpoints")
        
    
    # В конце обучения логируем финальную модель (для инференса)
    # Переводим в eval и на CPU перед сохранением, чтобы избежать проблем с CUDA
    model.eval()
    model.cpu()
    mlflow.pytorch.log_model(model, "final_model")
    
    # Возвращаем на GPU, если нужно
    model.to(device)
    
    logger.info(f"Обучение завершено. Run ID: {run_id}")
    logger.info(f"Чтобы продолжить, запустите скрипт с RESUME_RUN_ID='{run_id}'")