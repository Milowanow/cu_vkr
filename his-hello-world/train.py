
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ICVL_Dataset import ICVLDataset, normalize_icvl_hsi, normalize_rgb

from AWAN.AWAN import AWAN
from loss import MRAELoss

#логирование
from log_setup import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)
#

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

def train_epoch(
    network,
    train_loader,
    criterion,
    optimizer,
    writer=None,
    epoch=0,
    global_step=0,
    logging_interval=50,
):
    network.train()
    device = next(network.parameters()).device

    running_loss = 0.0
    n_batches = 0

    for step, (raw, hsi) in enumerate(train_loader):
        logger.debug(f"step: {step}")
        raw = raw.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        pred = network(raw)
        loss = criterion(pred, hsi)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss, global_step

### ---------------
### memory logger
import psutil, os, time, threading

def monitor():
    p = psutil.Process(os.getpid())
    while True:
        print(f"[mem] {p.memory_info().rss / 1024**2:.1f} MB", flush=True)
        time.sleep(1)

threading.Thread(target=monitor, daemon=True).start()
### ---------------

logger.debug(f"Start")

hsi_transform = transforms.Compose([
    transforms.Lambda(normalize_icvl_hsi),
])

rgb_transform = transforms.Compose([
    transforms.Lambda(normalize_rgb),
])

train_ds = ICVLDataset(
    raw_dir='./raw/ICVL_our/',
    hsi_dir='./downloaded/ICVL_our/',
    raw_ext='.png',
    train=True,
    patch_size=64,
    transform=rgb_transform,
    target_transform=hsi_transform
)

raw, hsi = train_ds[0]
# print("RAW:", raw.shape, raw.dtype, raw.min().item(), raw.max().item())
# print("HSI:", hsi.shape, hsi.dtype, hsi.min().item(), hsi.max().item())


train_loader = DataLoader(dataset = train_ds, batch_size=2, shuffle=False)

model = AWAN(inplanes=3, planes=31, channels=200, n_DRBs=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
criterion = MRAELoss()   # см. определение ниже
writer = SummaryWriter(log_dir="./runs/awan_exp1")


avg_loss, global_step = train_epoch(
     network = model,
     train_loader = train_loader,
     criterion = criterion,
     optimizer = optimizer,
     writer=None,
     epoch=0,
     global_step=0
 )

# print(avg_loss)
# print(global_step)

#global_step = 0
#for epoch in range(1, 101):
#     avg_loss, global_step = train_epoch(
#         model, train_loader, criterion, optimizer,
#         writer=writer, epoch=epoch, global_step=global_step,
#         logging_interval=50,
#     )
#     print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

    # чекпоинт
    #torch.save({
    #    "epoch": epoch,
    #    "model": model.state_dict(),
    #    "optimizer": optimizer.state_dict(),
    #    "loss": avg_loss,
    #}, f"./checkpoints/awan_epoch{epoch}.pth")