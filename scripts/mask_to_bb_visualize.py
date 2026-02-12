from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def load_mask(path: Path):
    """
    Load mask as uint8 2D array (0/1). Return (mask, (h, w)).
    """
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        im = Image.open(str(path)).convert("L")
        arr = np.asarray(im)

    if arr.ndim == 3:
        arr = arr[..., 0]

    mask = (arr > 0).astype(np.uint8)
    if mask.size == 0:
        raise RuntimeError("Mask is empty")

    h, w = mask.shape
    return mask, (h, w)

def mask_to_bbox(mask):
    """
    mask: 2D uint8 (0/1) -> [xmin, ymin, xmax, ymax] in pixels
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None

    ymin = int(ys.min())
    ymax = int(ys.max())
    xmin = int(xs.min())
    xmax = int(xs.max())
    return [xmin, ymin, xmax, ymax]

def draw_bbox_on_image(image_path: Path, bbox, out_path: Path,
                       color=(255, 255, 0), width=1):
    """
    Draw bbox [xmin, ymin, xmax, ymax] on image and save.
    """
    img = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)

    xmin, ymin, xmax, ymax = bbox
    for i in range(width):
        draw.rectangle(
            [xmin - i, ymin - i, xmax + i, ymax + i],
            outline=color
        )

    img.save(str(out_path))
    return out_path

def main():
    # ====== MODIFICA QUI ======
    mask_path  = Path("/media/TBData/marco/Projects/HangCon/outputs/SAM3_HangCon_Prompt/hanging_object_with_rope/score_0_5/mask/D3_230415_1052_1.png")   # <-- tua maschera
    image_path = Path("/media/TBData/marco/D3_230415_1052_1.png")   # <-- immagine RGB
    out_path   = Path("/media/TBData/marco/D3_230415_1052_1.png")

    # --- load mask ---
    mask, shape = load_mask(mask_path)

    # --- bbox from mask ---
    bbox = mask_to_bbox(mask)
    if bbox is None:
        raise RuntimeError("Predicted mask has no foreground → no bbox")

    print("Predicted bbox:", bbox)

    # --- draw and save ---
    draw_bbox_on_image(image_path, bbox, out_path,
                       color=(255, 255, 0), width=1)

    print(f"Saved image with bbox to: {out_path}")

if __name__ == "__main__":
    main()
