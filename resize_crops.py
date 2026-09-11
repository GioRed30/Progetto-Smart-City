"""
Pre-ridimensiona i ritagli alla risoluzione di input della rete.

Motivazione. Il DataLoader del framework usa num_workers=0: decodifica JPEG,
resize e normalizzazione avvengono in modo sincrono nel thread del client. Con
ritagli di mediana ~511px ridotti a 224px, e cinque passate per round su venti
round, la CPU satura e la GPU resta ferma all'1% di utilizzo.

Ridimensionare una volta sola su disco elimina quel lavoro ripetuto. Il risultato
e' identico: transforms.Resize((224, 224)) stira l'immagine alle dimensioni esatte
richieste, esattamente come fa questo script, con la stessa interpolazione
bilineare. Applicarlo prima invece che a ogni epoca non cambia i dati visti dalla
rete, cambia solo quante volte il calcolo viene eseguito.

Le immagini gia' alla dimensione richiesta vengono saltate, quindi rilanciare lo
script e' innocuo. I ritagli restano comunque rigenerabili da dataset/raw_capstone
tramite crop_from_labels.py.

Uso:
    python resize_crops.py --root dataset/road_signs_binary --size 224
    python resize_crops.py --root dataset/road_signs_test   --size 224
"""

import argparse
from pathlib import Path

from PIL import Image

CLASSES = ('damaged', 'healthy')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True,
                        help="Cartella con le sottocartelle damaged/ e healthy/.")
    parser.add_argument('--size', type=int, default=224,
                        help="Lato in pixel dell'immagine risultante (quadrata).")
    parser.add_argument('--quality', type=int, default=95,
                        help="Qualita' JPEG in salvataggio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    target = (args.size, args.size)

    resized = skipped = failed = 0
    bytes_before = bytes_after = 0

    for class_name in CLASSES:
        folder = root / class_name
        if not folder.is_dir():
            print(f"  cartella assente, salto: {folder}")
            continue

        paths = sorted(folder.glob('*.jpg'))
        print(f"{class_name}: {len(paths)} immagini")

        for path in paths:
            try:
                size_before = path.stat().st_size
                with Image.open(path) as handle:
                    if handle.size == target:
                        skipped += 1
                        continue
                    image = handle.convert('RGB').resize(target, Image.Resampling.BILINEAR)
                image.save(path, 'JPEG', quality=args.quality)
                bytes_before += size_before
                bytes_after += path.stat().st_size
                resized += 1
            except OSError as error:
                print(f"  ! errore su {path.name}: {error}")
                failed += 1

    print("-" * 52)
    print(f"ridimensionate : {resized}")
    print(f"gia' a {args.size}px  : {skipped}")
    if failed:
        print(f"fallite        : {failed}")
    if bytes_before:
        print(f"spazio         : {bytes_before / 1e6:.0f} MB -> {bytes_after / 1e6:.0f} MB "
              f"({100 * bytes_after / bytes_before:.0f}%)")


if __name__ == '__main__':
    main()
