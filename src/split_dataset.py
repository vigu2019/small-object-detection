# src/split_dataset.py
import os, glob, random, shutil, sys
random.seed(42)

# edit these ratios if you want
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

src_img_folder = sys.argv[1] if len(sys.argv)>1 else "data/images"
src_lbl_folder = sys.argv[2] if len(sys.argv)>2 else "data/labels"
out_img = "data/images"
out_lbl = "data/labels"

# collect all image files from src_img_folder root (non-recursive assumption; adapt if needed)
images = []
for ext in ('*.jpg','*.jpeg','*.png'):
    images.extend(glob.glob(os.path.join(src_img_folder, ext)))

images = sorted(images)
random.shuffle(images)

n = len(images)
n_train = int(train_ratio * n)
n_val = int(val_ratio * n)
train_imgs = images[:n_train]
val_imgs = images[n_train:n_train+n_val]
test_imgs = images[n_train+n_val:]

def copy_set(img_list, subset):
    os.makedirs(os.path.join(out_img, subset), exist_ok=True)
    os.makedirs(os.path.join(out_lbl, subset), exist_ok=True)
    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # copy image
        shutil.copy(img_path, os.path.join(out_img, subset, os.path.basename(img_path)))
        # copy corresponding label if exists
        lbl_path = os.path.join(src_lbl_folder, base + ".txt")
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(out_lbl, subset, base + ".txt"))
        else:
            # create empty file if no label (YOLO training can skip images w/out labels, but better to know)
            open(os.path.join(out_lbl, subset, base + ".txt"), 'a').close()

copy_set(train_imgs, "train")
copy_set(val_imgs, "val")
copy_set(test_imgs, "test")
print("Split done:", len(train_imgs), len(val_imgs), len(test_imgs))
