# src/xml_to_yolo.py
import os, sys, glob
import xml.etree.ElementTree as ET

# EDIT this to your class list (in correct order -> class_id)
CLASS_LIST = ['airplane','ship','storage-tank','baseball-diamond','tennis-court',
              'basketball-court','ground-track-field','harbor','bridge','vehicle']


def xml_to_yolo(xml_file, img_w, img_h):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in CLASS_LIST:
            continue
        cls_id = CLASS_LIST.index(cls)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/xml_to_yolo.py <xml_folder> <img_folder> <output_label_folder>")
        sys.exit(1)
    xml_folder, img_folder, out_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_folder, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_folder, "*.xml"))
    for xf in xml_files:
        base = os.path.splitext(os.path.basename(xf))[0]
        # find image to get size
        possible_imgs = [os.path.join(img_folder, base + ext) for ext in ['.jpg','.jpeg','.png']]
        img_path = None
        for p in possible_imgs:
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            print(f"Image not found for XML {xf}, skipping")
            continue
        # get image size without heavy deps
        from PIL import Image
        img = Image.open(img_path)
        w,h = img.size
        lines = xml_to_yolo(xf, w, h)
        out_file = os.path.join(out_folder, base + ".txt")
        with open(out_file, "w") as f:
            f.write("\n".join(lines))
    print("Conversion done.")
