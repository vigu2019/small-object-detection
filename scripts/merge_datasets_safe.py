import os, shutil
from tqdm import tqdm

SOURCES = {
    "vis": "data_yolo/visdrone/train",
    "nwpu": "data_yolo/nwpu/train"  # safe: will skip if missing
}

DEST = "data_merged/train"
IMG_DEST = os.path.join(DEST, "images")
LBL_DEST = os.path.join(DEST, "labels")

os.makedirs(IMG_DEST, exist_ok=True)
os.makedirs(LBL_DEST, exist_ok=True)

def copy_with_prefix(src_root, tag):
    img_src = os.path.join(src_root, "images")
    lbl_src = os.path.join(src_root, "labels")

    if not os.path.exists(img_src):
        print(f"SKIP: No dataset found for '{tag}' at: {img_src}")
        return

    print(f"Merging {tag} from {src_root}...")
    for img in tqdm(os.listdir(img_src), desc=tag):
        if not img.lower().endswith(('.jpg','.jpeg','.png')):
            continue

        base = os.path.splitext(img)[0]
        new_img = f"{tag}__{base}.jpg"

        shutil.copy2(os.path.join(img_src, img), os.path.join(IMG_DEST, new_img))

        lbl_in = os.path.join(lbl_src, base + ".txt")
        lbl_out = os.path.join(LBL_DEST, new_img.replace(".jpg", ".txt"))

        if os.path.exists(lbl_in):
            shutil.copy2(lbl_in, lbl_out)
        else:
            open(lbl_out, "w").close()

if __name__ == "__main__":
    for tag, path in SOURCES.items():
        copy_with_prefix(path, tag)

    print("Merge completed.")
