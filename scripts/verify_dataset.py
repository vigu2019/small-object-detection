# scripts/verify_dataset.py
import os, glob, sys

splits = ["train", "val", "test"]
img_exts = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

errors = False
for s in splits:
    img_dir = os.path.join('data','images',s)
    lbl_dir = os.path.join('data','labels',s)
    imgs = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(img_dir,'*')) if p.lower().endswith(img_exts)]
    lbls = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(lbl_dir,'*.txt'))]
    print(f"\n== split: {s} ==")
    print(f"images: {len(imgs)}, labels: {len(lbls)}")
    missing_labels = sorted(set(imgs) - set(lbls))
    missing_images = sorted(set(lbls) - set(imgs))
    if missing_labels:
        errors = True
        print("Images with NO label files (missing .txt):", missing_labels)
    if missing_images:
        errors = True
        print("Label files with NO matching images:", missing_images)
    # quick content check of labels
    for lblname in glob.glob(os.path.join(lbl_dir,'*.txt')):
        with open(lblname,'r') as f:
            for i,line in enumerate(f,1):
                parts = line.strip().split()
                if not parts: 
                    continue
                if len(parts) != 5:
                    print(f"Bad format: {lblname} line {i}: {line.strip()}")
                    errors = True
                else:
                    try:
                        cls = int(parts[0])
                        floats = list(map(float, parts[1:]))
                        if not all(0.0 <= v <= 1.0 for v in floats):
                            print(f"Out of range [0..1] in {lblname} line {i}: {line.strip()}")
                            errors = True
                    except Exception as e:
                        print(f"Parse error {lblname} line {i}: {line.strip()} -> {e}")
                        errors = True

if errors:
    print("\nDataset verification FAILED. Fix issues above.")
    sys.exit(2)
else:
    print("\nAll checks passed. Dataset looks OK.")
    