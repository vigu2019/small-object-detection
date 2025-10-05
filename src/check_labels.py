# src/check_labels.py
import os, glob, random
import cv2
import sys

img_folder = sys.argv[1] if len(sys.argv)>1 else "data/images/train"
lbl_folder = sys.argv[2] if len(sys.argv)>2 else "data/labels/train"

imgs = []
for ext in ('*.jpg','*.jpeg','*.png'):
    imgs.extend(glob.glob(os.path.join(img_folder, ext)))
imgs = sorted(imgs)
print("Images found:", len(imgs))
no_label = 0
for img in imgs:
    base = os.path.splitext(os.path.basename(img))[0]
    if not os.path.exists(os.path.join(lbl_folder, base + ".txt")):
        no_label += 1
print("Images without label:", no_label)

# show a random sample with boxes
if len(imgs)>0:
    sample = random.choice(imgs)
    img = cv2.imread(sample)
    h,w = img.shape[:2]
    base = os.path.splitext(os.path.basename(sample))[0]
    lbl = os.path.join(lbl_folder, base + ".txt")
    if os.path.exists(lbl):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)<5: continue
                _, xc,yc,ww,hh = map(float, parts[:5])
                x1 = int((xc - ww/2)*w)
                y1 = int((yc - hh/2)*h)
                x2 = int((xc + ww/2)*w)
                y2 = int((yc + hh/2)*h)
                cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
    print("Showing sample:", sample)
    cv2.imshow("sample", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
