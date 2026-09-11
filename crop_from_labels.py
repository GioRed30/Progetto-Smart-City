"""
Da dataset di object detection (formato YOLO) a struttura ImageFolder di ritagli.

A differenza di detect_and_crop.py, qui NON si usa alcun detector: i ritagli sono
generati dalle bounding box di ground truth, e la classe di ogni ritaglio si legge
direttamente dal file di label. E' il percorso corretto per costruire il dataset
di addestramento del classificatore federato.

Input atteso (export Roboflow in formato YOLOv8/YOLO11/YOLO26):

    <root>/data.yaml            nc + names
    <root>/train/images/*.jpg
    <root>/train/labels/*.txt   righe: <class_id> <cx> <cy> <w> <h>  (normalizzati)
    <root>/valid/...  <root>/test/...

Output prodotto:

    <out>/damaged/*.jpg
    <out>/healthy/*.jpg

Uso tipico, in due passi.

  1) Ispeziona quali classi contiene il dataset e con quante box ciascuna:

     python crop_from_labels.py --root dataset/raw_capstone --list-classes

  2) Genera i ritagli mappando le classi sulle due categorie:

     python crop_from_labels.py --root dataset/raw_capstone \
         --out dataset/road_signs_binary \
         --damaged Broken,Vandalize --healthy Clean \
         --min-size 32

Le classi non elencate in --damaged o --healthy vengono ignorate: utile per
scartare categorie ambigue senza doverle rimuovere dal dataset.
"""

import argparse
import collections
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image

ALL_SPLITS = ('train', 'valid', 'val', 'test')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True,
                        help="Cartella radice dell'export (quella che contiene data.yaml).")
    parser.add_argument('--out', default=None,
                        help="Cartella di destinazione dei ritagli. Non serve con --list-classes.")
    parser.add_argument('--damaged', default='',
                        help="Nomi delle classi da mappare su 'damaged', separati da virgola.")
    parser.add_argument('--healthy', default='',
                        help="Nomi delle classi da mappare su 'healthy', separati da virgola.")
    parser.add_argument('--list-classes', action='store_true',
                        help="Stampa le classi presenti con il conteggio delle box ed esce.")
    parser.add_argument('--pad', type=float, default=0.05,
                        help="Margine proporzionale attorno alla box (0.05 = 5%%).")
    parser.add_argument('--min-size', type=int, default=32,
                        help="Scarta i ritagli con lato inferiore a questo valore in pixel.")
    parser.add_argument('--min-aspect', type=float, default=0.5,
                        help="Aspect ratio minimo (larghezza/altezza) della box. I cartelli "
                             "hanno AR ~1.0; box molto verticali sono in genere sfondo o "
                             "annotazioni che includono il palo. 0 disattiva il filtro.")
    parser.add_argument('--max-aspect', type=float, default=2.0,
                        help="Aspect ratio massimo della box. 0 disattiva il filtro.")
    parser.add_argument('--splits', default=','.join(ALL_SPLITS),
                        help="Split dell'export da processare, separati da virgola. "
                             "Default: tutti. Usa '--splits train,valid' per il pool di "
                             "addestramento e '--splits test' per un test set esterno.")
    return parser.parse_args()


def load_class_names(root: Path) -> List[str]:
    """Legge la lista dei nomi di classe da data.yaml."""
    yaml_path = root / 'data.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml non trovato in {root}")
    with open(yaml_path, 'r', encoding='utf-8') as handle:
        content = yaml.safe_load(handle)
    names = content.get('names')
    if isinstance(names, dict):  # formato {0: 'a', 1: 'b'}
        names = [names[key] for key in sorted(names)]
    if not names:
        raise ValueError(f"Nessuna classe trovata in {yaml_path}")
    return list(names)


def iter_label_files(root: Path, splits=ALL_SPLITS):
    """Produce coppie (file_immagine, file_label) per gli split richiesti."""
    for split in splits:
        labels_dir = root / split / 'labels'
        images_dir = root / split / 'images'
        if not labels_dir.is_dir() or not images_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob('*.txt')):
            for extension in IMAGE_EXTENSIONS:
                image_path = images_dir / (label_path.stem + extension)
                if image_path.is_file():
                    yield image_path, label_path
                    break


def read_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Legge le box da un file di label YOLO, ignorando righe malformate."""
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            boxes.append((class_id, cx, cy, bw, bh))
    return boxes


def count_classes(root: Path, class_names: List[str], splits) -> collections.Counter:
    counter = collections.Counter()
    for _, label_path in iter_label_files(root, splits):
        for class_id, *_ in read_boxes(label_path):
            name = class_names[class_id] if 0 <= class_id < len(class_names) else f"<id {class_id}>"
            counter[name] += 1
    return counter


def build_mapping(args: argparse.Namespace) -> Dict[str, str]:
    """Costruisce il dizionario nome_classe -> 'damaged'/'healthy'."""
    mapping = {}
    for name in (item.strip() for item in args.damaged.split(',')):
        if name:
            mapping[name] = 'damaged'
    for name in (item.strip() for item in args.healthy.split(',')):
        if name:
            mapping[name] = 'healthy'
    return mapping


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    class_names = load_class_names(root)
    splits = tuple(s.strip() for s in args.splits.split(',') if s.strip())

    if args.list_classes:
        counter = count_classes(root, class_names, splits)
        total = sum(counter.values())
        print(f"data.yaml dichiara {len(class_names)} classi: {class_names}\n")
        print(f"{'classe':<32s} {'box':>8s}  {'%':>6s}")
        print("-" * 50)
        for name, count in counter.most_common():
            print(f"{name:<32s} {count:8d}  {100 * count / total:5.1f}%")
        print("-" * 50)
        print(f"{'TOTALE':<32s} {total:8d}")
        print("\nOra rilancia indicando la mappatura, per esempio:")
        print("  --damaged <classi,danneggiate> --healthy <classi,integre>")
        return

    if not args.out:
        raise SystemExit("--out e' obbligatorio quando non si usa --list-classes")

    mapping = build_mapping(args)
    if not mapping:
        raise SystemExit("Specifica almeno --damaged o --healthy (vedi --list-classes)")

    out_root = Path(args.out)
    for label in ('damaged', 'healthy'):
        (out_root / label).mkdir(parents=True, exist_ok=True)

    saved = collections.Counter()
    skipped_small = skipped_unmapped = skipped_aspect = 0

    print(f"Split processati: {', '.join(splits)}")
    for image_path, label_path in iter_label_files(root, splits):
        boxes = read_boxes(label_path)
        if not boxes:
            continue
        with Image.open(image_path) as handle:
            image = handle.convert('RGB')
            width, height = image.size

            for index, (class_id, cx, cy, bw, bh) in enumerate(boxes):
                name = class_names[class_id] if 0 <= class_id < len(class_names) else None
                target = mapping.get(name)
                if target is None:
                    skipped_unmapped += 1
                    continue

                # Il filtro sull'aspect ratio va applicato alla box originale.
                # Il margine e' proporzionale su entrambi i lati, quindi non lo altera.
                if bh * height > 0:
                    aspect = (bw * width) / (bh * height)
                    if (args.min_aspect and aspect < args.min_aspect) or \
                       (args.max_aspect and aspect > args.max_aspect):
                        skipped_aspect += 1
                        continue

                # da coordinate normalizzate a pixel, con margine
                box_w, box_h = bw * width * (1 + 2 * args.pad), bh * height * (1 + 2 * args.pad)
                x1 = max(0.0, cx * width - box_w / 2)
                y1 = max(0.0, cy * height - box_h / 2)
                x2 = min(float(width), cx * width + box_w / 2)
                y2 = min(float(height), cy * height + box_h / 2)

                if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                    skipped_small += 1
                    continue

                crop = image.crop((round(x1), round(y1), round(x2), round(y2)))
                filename = f"{image_path.stem}_{index:02d}.jpg"
                crop.save(out_root / target / filename, quality=95)
                saved[target] += 1

    print("-" * 50)
    for label in ('damaged', 'healthy'):
        print(f"{label:<12s}: {saved[label]:6d} ritagli")
    print(f"{'scartati':<12s}: {skipped_small:6d} troppo piccoli (< {args.min_size}px), "
          f"{skipped_aspect} fuori AR [{args.min_aspect}-{args.max_aspect}], "
          f"{skipped_unmapped} di classi non mappate")
    total = saved['damaged'] + saved['healthy']
    if total:
        ratio = saved['damaged'] / total
        print(f"\nTotale {total} ritagli in '{out_root}' - bilanciamento damaged {100 * ratio:.1f}%")


if __name__ == '__main__':
    main()
