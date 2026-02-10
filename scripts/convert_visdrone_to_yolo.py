import os, cv2
from tqdm import tqdm

VISDRONE_TO_YOLO = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9
}

def convert_ann(vis_ann_path, img_path, out_label):
    img = cv2.imread(img_path)
    if img is None:
        open(out_label, "w").close()
        return

    h, w = img.shape[:2]
    out = []

    with open(vis_ann_path) as f:
        for line in f:
            vals = line.strip().split(',')
            x, y, bw, bh = map(float, vals[:4])
            cat = int(vals[5])

            if cat not in VISDRONE_TO_YOLO:
                continue

            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            bw /= w
            bh /= h
            cls = VISDRONE_TO_YOLO[cat]
            out.append(f"{cls} {xc} {yc} {bw} {bh}")

    with open(out_label, "w") as f:
        f.write("\n".join(out))

def convert_folder(src, dst):
    img_in = os.path.join(src, "images")
    ann_in = os.path.join(src, "annotations")

    img_out = os.path.join(dst, "images")
    lbl_out = os.path.join(dst, "labels")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img in tqdm(os.listdir(img_in)):
        if not img.lower().endswith(".jpg"):
            continue

        base = img[:-4]
        src_img = os.path.join(img_in, img)
        src_ann = os.path.join(ann_in, base + ".txt")

        import shutil
        shutil.copy2(src_img, os.path.join(img_out, img))

        out_lbl = os.path.join(lbl_out, base + ".txt")

        if os.path.exists(src_ann):
            convert_ann(src_ann, src_img, out_lbl)
        else:
            open(out_lbl, "w").close()

if __name__ == "__main__":
    mapping = {
        "data_raw/visdrone_train": "data_yolo/visdrone/train",
        "data_raw/visdrone_val":   "data_yolo/visdrone/val",
        "data_raw/visdrone_test":  "data_yolo/visdrone/test"
    }

    for src, dst in mapping.items():
        print("Processing:", src)
        convert_folder(src, dst)
