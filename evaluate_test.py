"""
Valutazione del modello globale sul test set indipendente.

I 957 ritagli di dataset/road_signs_test provengono dallo split di test dell'export
originale, che Roboflow costruisce per immagine e non per bounding box: ritagli dello
stesso scatto restano quindi dalla stessa parte. Sono inoltre stati deduplicati
rispetto al pool di addestramento, comprese le copie speculari.

Il framework federato non li usa: valuta soltanto sui validation set locali dei
client. Questo script colma quel vuoto, ed e' la differenza fra poter affermare che
il modello impara e poter affermare che generalizza.

Richiede un file di pesi prodotto dall'aggregatore (*_weights.npz). Se non ne esiste
nessuno, rigenerali eseguendo una singola configurazione:

    FL_CONFIG=best_config.json FL_WORKERS=1 python federated_grid_search.py

Uso:
    python evaluate_test.py                      # usa i pesi col miglior F1 trovati
    python evaluate_test.py --weights <file>     # usa un file specifico
    python evaluate_test.py --tutti              # valuta tutti i pesi disponibili
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from model_manager import ModelManager

TEST_DIR = "road_signs_test"
TEST_ROOT = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', default=None,
                        help="File .npz specifico. Senza, usa quello col miglior F1 di validazione.")
    parser.add_argument('--tutti', action='store_true',
                        help="Valuta tutti i file di pesi trovati e li mette a confronto.")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def trova_pesi() -> List[Path]:
    trovati = []
    for schema in ("csv*/runs/*_weights.npz", "csv*/*/runs/*_weights.npz"):
        trovati.extend(Path(p) for p in glob.glob(schema))
    return sorted(set(trovati))


def carica(percorso: Path):
    dati = np.load(percorso, allow_pickle=False)
    n = int(dati['n_tensori'])
    pesi = [dati[f'w{i}'] for i in range(n)]
    meta = {
        'model_name': str(dati['model_name']),
        'num_custom_layers': int(dati['num_custom_layers']),
        'num_classes': int(dati['num_classes']),
        'image_size': int(dati['image_size']),
        'best_round': int(dati['best_round']),
        'best_f1': float(dati['best_f1']),
        'aggregation_algorithm': str(dati['aggregation_algorithm']),
        'learning_rate': float(dati['learning_rate']),
        'num_clients': int(dati['num_clients']),
    }
    return pesi, meta


def valuta(pesi, meta: Dict, batch_size: int, device: str) -> Dict:
    config = {
        'model_name': meta['model_name'],
        'num_custom_layers': meta['num_custom_layers'],
        'num_classes': meta['num_classes'],
        'image_size': meta['image_size'],
        'device': device,
        'worker_id': 0,
        'cache_dataset': False,
    }
    # dataset_path punta alla cartella che contiene road_signs_test, cosi' lo "split"
    # richiesto al dataloader coincide con la cartella del test set.
    gestore = ModelManager(config=config, dataset_path=TEST_ROOT)
    gestore.set_weights(pesi)
    caricatore = gestore._get_dataloader(TEST_DIR, batch_size)

    gestore.model.eval()
    previsti, reali = [], []
    with torch.no_grad():
        for immagini, etichette in caricatore:
            immagini = immagini.to(gestore.device)
            uscite = gestore.model(immagini)
            _, p = torch.max(uscite, 1)
            previsti.extend(p.cpu().numpy())
            reali.extend(etichette.numpy())

    return {
        'f1': f1_score(reali, previsti, average='weighted', zero_division=0),
        'accuracy': accuracy_score(reali, previsti),
        'precision': precision_score(reali, previsti, average='weighted', zero_division=0),
        'recall': recall_score(reali, previsti, average='weighted', zero_division=0),
        'matrice': confusion_matrix(reali, previsti),
        'classi': caricatore.dataset.classes,
        'n': len(reali),
    }


def stampa(meta: Dict, ris: Dict) -> None:
    print(f"  modello        : {meta['model_name']}, head {meta['num_custom_layers']} layer")
    print(f"  configurazione : {meta['aggregation_algorithm']}, lr {meta['learning_rate']}, "
          f"{meta['num_clients']} client, round migliore {meta['best_round']}")
    print()
    print(f"  {'':16s}{'validazione':>13s}{'test':>11s}{'scarto':>10s}")
    print("  " + "-" * 50)
    print(f"  {'F1':16s}{meta['best_f1']:>13.4f}{ris['f1']:>11.4f}"
          f"{ris['f1'] - meta['best_f1']:>+10.4f}")
    for nome in ('accuracy', 'precision', 'recall'):
        print(f"  {nome:16s}{'':>13s}{ris[nome]:>11.4f}")
    print()
    print(f"  matrice di confusione ({ris['n']} campioni)")
    etichette = ris['classi']
    print(f"  {'':14s}" + "".join(f"{'-> ' + c:>14s}" for c in etichette))
    for i, riga in enumerate(ris['matrice']):
        print(f"  {etichette[i]:<14s}" + "".join(f"{v:>14d}" for v in riga))
    print()
    for i, c in enumerate(etichette):
        tot = ris['matrice'][i].sum()
        ok = ris['matrice'][i][i]
        print(f"  {c:<14s} {ok}/{tot} corretti ({100 * ok / tot:.1f}%)")


def main() -> None:
    args = parse_args()

    if not (Path(TEST_ROOT) / TEST_DIR).is_dir():
        raise SystemExit(f"Test set non trovato in {TEST_ROOT}/{TEST_DIR}")

    if args.weights:
        candidati = [Path(args.weights)]
    else:
        candidati = trova_pesi()
        if not candidati:
            raise SystemExit(
                "Nessun file di pesi trovato.\n\n"
                "Le esecuzioni gia' svolte non li avevano salvati. Rigenerali con:\n"
                "    FL_CONFIG=best_config.json FL_WORKERS=1 python federated_grid_search.py"
            )

    caricati = [(p, *carica(p)) for p in candidati]
    if not args.tutti and len(caricati) > 1:
        caricati = [max(caricati, key=lambda t: t[2]['best_f1'])]

    for percorso, pesi, meta in caricati:
        print("=" * 62)
        print(f"  {percorso.name}")
        print("=" * 62)
        ris = valuta(pesi, meta, args.batch_size, args.device)
        stampa(meta, ris)
        print()


if __name__ == '__main__':
    main()
