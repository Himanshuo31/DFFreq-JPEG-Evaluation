from pathlib import Path
from PIL import Image

SOURCE = Path(r"C:\Users\LENOVO\Downloads\dalle\dalle")
OUTPUT = Path(r"C:\Users\LENOVO\Downloads\dalle\jpeg_experiments")

QUALITIES = [95, 75, 50, 25]
CLASSES = ["0_real", "1_fake"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

for quality in QUALITIES:
    for class_name in CLASSES:
        source_folder = SOURCE / class_name
        output_folder = OUTPUT / f"quality_{quality}" / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        files = [
            f for f in source_folder.iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS
        ]

        for file_path in files:
            with Image.open(file_path) as image:
                image = image.convert("RGB")
                output_path = output_folder / f"{file_path.stem}.jpg"
                image.save(
                    output_path,
                    format="JPEG",
                    quality=quality
                )

        print(
            f"Quality {quality}, {class_name}: "
            f"{len(files)} images saved"
        )

print("All JPEG compression copies created successfully.")