import cv2
import os

# -----------------------------
# CONFIG
# -----------------------------
CLASS_NAMES = [
    "Hanging object",
    "Rope",
    "Hanging object with rope"
]

# Colori BGR (uno per classe)
CLASS_COLORS = {
    0: (0, 255, 0),     # verde
    1: (255, 0, 255),     # blu
    2: (255, 255, 0),     # rosso
}

# Ordine di disegno desiderato (dal basso verso l'alto)
DRAW_ORDER = [2, 0, 1]

# Spessore bbox (più sottile)
BBOX_THICKNESS = 1

# -----------------------------
# FUNZIONE PRINCIPALE
# -----------------------------
def visualize_yolo_bboxes(image_path, yolo_txt_path, output_path):
    # Carica immagine
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile caricare immagine: {image_path}")

    h, w, _ = img.shape

    # Se il file txt non esiste o è vuoto → salva immagine così com'è
    if not os.path.exists(yolo_txt_path) or os.path.getsize(yolo_txt_path) == 0:
        print("Nessuna annotazione trovata.")
        cv2.imwrite(output_path, img)
        return

    # Leggi annotazioni YOLO
    with open(yolo_txt_path, "r") as f:
        lines = f.readlines()

    boxes = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_w = float(parts[3])
        box_h = float(parts[4])

        # Converti da normalizzato a pixel
        x_center_px = x_center * w
        y_center_px = y_center * h
        box_w_px = box_w * w
        box_h_px = box_h * h

        x1 = int(x_center_px - box_w_px / 2)
        y1 = int(y_center_px - box_h_px / 2)
        x2 = int(x_center_px + box_w_px / 2)
        y2 = int(y_center_px + box_h_px / 2)

        # Clipping ai bordi immagine
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        boxes.append({
            "class_id": class_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    # Ordina i box secondo l'ordine desiderato: 2 → 0 → 1
    boxes_sorted = sorted(
        boxes,
        key=lambda b: DRAW_ORDER.index(b["class_id"]) if b["class_id"] in DRAW_ORDER else len(DRAW_ORDER)
    )

    # Disegna i bbox in ordine
    for b in boxes_sorted:
        class_id = b["class_id"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        color = CLASS_COLORS.get(class_id, (0, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

    # Salva risultato
    cv2.imwrite(output_path, img)
    print(f"Immagine salvata in: {output_path}")


# -----------------------------
# ESEMPIO DI USO
# -----------------------------
if __name__ == "__main__":
    image_path = "/media/TBData/marco/D3_230415_1052_1.png"
    yolo_txt_path = "/media/TBData/marco/Projects/HangCon/Dataset/HangCon/test/labels/YOLO/labels_yolo_exp3/D3_230415_1052_1.txt"
    output_path = "/home/marco/Desktop/SAM3-exp/HangCon_qualitative/D3_230415_1052_1.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    visualize_yolo_bboxes(image_path, yolo_txt_path, output_path)
