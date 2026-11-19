from pathlib import Path
from PIL import Image

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle")
SOURCE = BASE / "dalle"
OUTPUT = BASE / "jpeg_experiments" / "quality_100"

for cls in ["0_real", "1_fake"]:
    src = SOURCE / cls
    dst = OUTPUT / cls
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    for p in files:
        with Image.open(p) as im:
            im.convert("RGB").save(dst / (p.stem + ".jpg"), "JPEG", quality=100, subsampling=0)

    print(f"{cls}: converted {len(files)} images")

print("Baseline creation complete.")
