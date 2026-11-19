from pathlib import Path
from PIL import Image

SOURCE = Path(r"C:\Users\LENOVO\Downloads\dalle\dalle")
OUTPUT = Path(r"C:\Users\LENOVO\Downloads\dalle\controlled_jpeg")

QUALITIES = [100, 95, 75, 50, 25]
CLASSES = ["0_real", "1_fake"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

for quality in QUALITIES:
    for cls in CLASSES:
        src = SOURCE / cls
        dst = OUTPUT / f"quality_{quality}" / cls
        dst.mkdir(parents=True, exist_ok=True)

        files = [
            p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        for p in files:
            with Image.open(p) as img:
                img.convert("RGB").save(
                    dst / f"{p.stem}.jpg",
                    format="JPEG",
                    quality=quality,
                    subsampling=0
                )

        print(f"Q{quality} | {cls}: {len(files)} images")

print("Controlled dataset creation complete.")