# scripts/split_dataset.py
import os, shutil
from sklearn.model_selection import train_test_split

SRC = "data_merged/train"
OUT = "data_final"
TRAIN_RATIO = 0.8  # 80% train, 10% val, 10% test

imgs_dir = os.path.join(SRC, "images")
if not os.path.exists(imgs_dir):
    raise SystemExit(f"Missing source images folder: {imgs_dir}")

all_imgs = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
print("Total merged images:", len(all_imgs))
if len(all_imgs) == 0:
    raise SystemExit("No images found to split.")

train, rem = train_test_split(all_imgs, test_size=1-TRAIN_RATIO, random_state=42)
val, test = train_test_split(rem, test_size=0.5, random_state=42)

splits = {"train": train, "val": val, "test": test}
for split, files in splits.items():
    img_out = os.path.join(OUT, split, "images")
    lbl_out = os.path.join(OUT, split, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img in files:
        base = os.path.splitext(img)[0]
        shutil.copy2(os.path.join(SRC, "images", img), os.path.join(img_out, img))
        lbl_src = os.path.join(SRC, "labels", base + ".txt")
        lbl_dst = os.path.join(lbl_out, base + ".txt")
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)
        else:
            open(lbl_dst, "w").close()

print("Split sizes:", {k: len(v) for k,v in splits.items()})
