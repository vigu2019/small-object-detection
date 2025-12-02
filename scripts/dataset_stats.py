# scripts/dataset_stats.py
import os
from collections import Counter

CLASS_NAMES = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]

def stats(split_folder):
    imgs = [f for f in os.listdir(os.path.join(split_folder,"images")) if f.lower().endswith(('.jpg','.png'))]
    label_dir = os.path.join(split_folder,"labels")
    class_counts = Counter()
    for img in imgs:
        base = os.path.splitext(img)[0]
        lbl = os.path.join(label_dir, base + ".txt")
        if not os.path.exists(lbl): continue
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls = int(parts[0])
                class_counts[cls] += 1
    return len(imgs), class_counts

for split in ["data_final/train","data_final/val","data_final/test"]:
    if not os.path.exists(split): 
        print("Skip missing:", split)
        continue
    total, counts = stats(split)
    print("----", split, "----")
    print("Images:", total)
    for cls_idx, cnt in sorted(counts.items()):
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
        print(f"  {cls_idx} ({name}): {cnt}")
