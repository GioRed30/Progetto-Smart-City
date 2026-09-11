"""
Deduplica dei ritagli, consapevole delle copie specchiate.

Il dataset sorgente contiene immagini generate per flip orizzontale. Se un ritaglio
finisce nel pool di addestramento e la sua copia specchiata nel test set, le metriche
risultano gonfiate: il modello ha gia' visto quella scena. Lo stesso vale fra train e
valid interni.

Metodo. Per ogni immagine si calcola il dHash (64 bit) dell'immagine e quello della
sua immagine speculare, e si usa come chiave canonica il minimo dei due:

    canonical(x) = min(dhash(x), dhash(mirror(x)))

Due immagini identiche, o l'una lo specchio dell'altra, condividono la stessa chiave.
Raggruppando per chiave si individuano i duplicati senza confronti a coppie.

Politica di rimozione:
  * duplicati interni al pool di training  -> se ne tiene uno solo
  * collisioni fra training e test set     -> si rimuove dal TRAINING, mai dal test,
                                              per preservare l'integrita' del test set
  * collisioni fra damaged e healthy       -> segnalate, mai rimosse in automatico:
                                              sono conflitti di etichetta da ispezionare

Di default lo script non cancella nulla: riporta soltanto. Usare --apply per agire.

Esempio:
    python dedup_crops.py --train dataset/road_signs_binary --test dataset/road_signs_test
    python dedup_crops.py --train dataset/road_signs_binary --test dataset/road_signs_test --apply
"""

import argparse
import collections
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

CLASSES = ('damaged', 'healthy')
HASH_SIDE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', required=True,
                        help="Cartella del pool di addestramento (con damaged/ e healthy/).")
    parser.add_argument('--test', default=None,
                        help="Cartella del test set, per rilevare le collisioni incrociate.")
    parser.add_argument('--apply', action='store_true',
                        help="Cancella davvero i file. Senza questo flag lo script riporta soltanto.")
    return parser.parse_args()


def dhash(image: Image.Image) -> int:
    """dHash a 64 bit: confronta ogni pixel con il successivo sulla stessa riga."""
    small = image.convert('L').resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def canonical_hash(path: Path) -> int:
    """Chiave invariante rispetto al ribaltamento orizzontale."""
    with Image.open(path) as handle:
        image = handle.convert('RGB')
        direct = dhash(image)
        mirrored = dhash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    return min(direct, mirrored)


def index_folder(root: Path) -> Dict[int, List[Tuple[str, Path]]]:
    """Mappa chiave canonica -> lista di (classe, percorso)."""
    index = collections.defaultdict(list)
    for class_name in CLASSES:
        folder = root / class_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.jpg')):
            try:
                index[canonical_hash(path)].append((class_name, path))
            except OSError as error:
                print(f"  ! illeggibile, saltato: {path.name} ({error})")
    return index


def main() -> None:
    args = parse_args()
    train_root = Path(args.train)

    print(f"Indicizzo il pool di addestramento: {train_root}")
    train_index = index_folder(train_root)
    train_total = sum(len(items) for items in train_index.values())
    print(f"  {train_total} immagini, {len(train_index)} chiavi distinte")

    test_index = {}
    if args.test:
        test_root = Path(args.test)
        print(f"Indicizzo il test set: {test_root}")
        test_index = index_folder(test_root)
        test_total = sum(len(items) for items in test_index.values())
        print(f"  {test_total} immagini, {len(test_index)} chiavi distinte")

    to_delete: List[Path] = []
    label_conflicts: List[Tuple[Path, Path]] = []

    # 1. Duplicati interni al pool di addestramento
    intra_dupes = 0
    for key, items in train_index.items():
        if len(items) < 2:
            continue
        classes = {class_name for class_name, _ in items}
        if len(classes) > 1:
            # mostra un rappresentante per ciascuna delle due classi in conflitto
            first_of_class = {}
            for class_name, path in items:
                first_of_class.setdefault(class_name, path)
            pair = list(first_of_class.values())
            label_conflicts.append((pair[0], pair[1]))
            continue  # conflitto di etichetta: non decidere in automatico
        for _, path in items[1:]:
            to_delete.append(path)
            intra_dupes += 1

    # 2. Collisioni fra training e test: si rimuove sempre dal training
    cross_dupes = 0
    for key, items in train_index.items():
        if key in test_index:
            for _, path in items:
                if path not in to_delete:
                    to_delete.append(path)
                    cross_dupes += 1

    print("\n" + "=" * 58)
    print(f"{'duplicati interni al training':<42s} {intra_dupes:6d}")
    print(f"{'collisioni training/test (rimosse dal train)':<42s} {cross_dupes:6d}")
    print(f"{'conflitti di etichetta (damaged vs healthy)':<42s} {len(label_conflicts):6d}")
    print("-" * 58)
    print(f"{'da rimuovere in totale':<42s} {len(to_delete):6d}")
    print(f"{'restano nel pool di addestramento':<42s} {train_total - len(to_delete):6d}")
    print("=" * 58)

    if label_conflicts:
        print("\nConflitti di etichetta: stessa immagine presente in entrambe le classi.")
        print("Non vengono rimossi automaticamente, vanno ispezionati. Primi esempi:")
        for first, second in label_conflicts[:5]:
            print(f"  {first.parent.name}/{first.name}  <->  {second.parent.name}/{second.name}")

    if not args.apply:
        print("\nNessun file cancellato. Rilancia con --apply per applicare le rimozioni.")
        return

    removed = 0
    for path in to_delete:
        try:
            path.unlink()
            removed += 1
        except OSError as error:
            print(f"  ! impossibile rimuovere {path}: {error}")

    print(f"\nRimossi {removed} file.")
    for class_name in CLASSES:
        folder = train_root / class_name
        if folder.is_dir():
            print(f"  {class_name:<10s}: {len(list(folder.glob('*.jpg'))):6d} rimasti")


if __name__ == '__main__':
    main()
