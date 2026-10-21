import torch

# 1. Проверка доступности GPU
print(f"PyTorch version: {torch.__version__}")
print(f"HIP/ROCm version: {torch.version.hip}")  # покажет версию ROCm, например '6.2.41134'
print(f"CUDA available (для ROCm тоже True): {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device properties: {torch.cuda.get_device_properties(0)}")

# 2. Выбор устройства (тот же API, что и для CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Для явного указания конкретной карты: torch.device("cuda:0")

# 3. Простое вычисление на GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = x @ y  # матричное умножение на GPU

print(f"Result device: {z.device}")
print(f"Result sum: {z.sum().item()}")