import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np



# --- Конфигурация ---
IMAGE_PATH = "eve_0331-1656.png"          # <-- путь к вашему изображению
OUTPUT_DIR = "generated/masks"          # <-- папка для сохранения масок
CONFIDENCE_THRESHOLD = 0.8              # порог из статьи (раздел 3)

# Материалы, которые ищем (список можно расширять)
MATERIAL_PROMPTS = [
    "wood",
    "fabric",
    "ceramic",
    "metal",
    "plastic",
    "glass",
    "paper",
    "grass"
]


def save_mask_as_png(mask: np.ndarray, path: str) -> None:
    """Сохраняет бинарную маску как PNG (белое — объект, чёрное — фон)."""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(path)


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    """Накладывает цветную маску на исходное изображение для визуализации."""
    img_array = np.array(image).astype(np.float32)
    overlay = img_array.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask > 0, img_array[:, :, c] * (1 - alpha) + color[c] * alpha, img_array[:, :, c])
    return Image.fromarray(overlay.astype(np.uint8))


def main():
    # 1. Загрузка модели (веса скачаются при первом запуске, ~1.7 ГБ)
    print("Загрузка SAM 3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, device="cpu")

    # 2. Загрузка изображения
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Изображение не найдено: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Изображение загружено: {image.size}")

    # 3. Препроцессинг (один раз для изображения)
    inference_state = processor.set_image(image)

    # 4. Создаём выходную папку
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. Проходим по всем материалам
    all_results = {}

    for material in MATERIAL_PROMPTS:
        print(f"\nПоиск материала: '{material}'")

        # Запуск инференса с текстовым промптом
        output = processor.set_text_prompt(state=inference_state, prompt=material)

        masks = output["masks"]      # список бинарных масок (numpy или torch)
        boxes = output["boxes"]      # список bounding box'ов
        scores = output["scores"]    # список уверенностей

        # Фильтрация по порогу уверенности (как в статье: 0.8)
        filtered = [
            (mask, box, score)
            for mask, box, score in zip(masks, boxes, scores)
            if float(score) >= CONFIDENCE_THRESHOLD
        ]

        print(f"  Найдено всего: {len(masks)}, после фильтра (≥{CONFIDENCE_THRESHOLD}): {len(filtered)}")

        all_results[material] = filtered

        # Сохраняем каждую маску
        for idx, (mask, box, score) in enumerate(filtered):
            # Приводим маску к numpy (на случай, если это torch.Tensor)
            if hasattr(mask, "cpu"):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.asarray(mask)

            # Бинаризуем (на всякий случай)
            mask_np = (mask_np > 0.5).astype(np.uint8)

            # Имя файла: material_001_score0.85.png
            fname = f"{material}_{idx:03d}_score{float(score):.2f}.png"
            save_path = os.path.join(OUTPUT_DIR, fname)
            save_mask_as_png(mask_np, save_path)

            # Дополнительно: визуализация с наложением
            overlay = overlay_mask_on_image(image, mask_np)
            overlay_path = os.path.join(OUTPUT_DIR, f"{material}_{idx:03d}_overlay.png")
            overlay.save(overlay_path)

    # 6. Итоговая статистика
    print("\n=== Итог ===")
    for material, results in all_results.items():
        print(f"{material}: {len(results)} масок с уверенностью ≥{CONFIDENCE_THRESHOLD}")

    print(f"\nВсе маски сохранены в: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()