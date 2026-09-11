# Vision Transformer compatti nel federated learning cifrato

Classificazione binaria di segnaletica stradale danneggiata in un sistema di
**Federated Learning** protetto da **crittografia omomorfica parziale** (schema di
Paillier), con confronto fra due backbone — **ResNet18** e **DeiT-Tiny** — e fra due
profondità della testa di classificazione.

Il lavoro estende il framework **HELM** del gruppo di ricerca, che è integralmente
contenuto in questa cartella e non è stato modificato nelle sue parti di protocollo.


## 1. Contenuto

### Framework federato (ereditato, invariato nel protocollo)

| File | Ruolo |
|---|---|
| `federated_server.py` | Server: coordina i round, aggrega, valuta |
| `federated_client.py` | Client: addestramento locale, cifratura, decifratura |
| `trusted_authority.py` | Genera la coppia di chiavi Paillier e distribuisce la privata ai soli client |
| `run_multiple_clients.py` | Avvia gli N client come thread separati |
| `data_splitter.py` | Partiziona il dataset fra i client con `StratifiedKFold` |
| `aggregator.py` | Algoritmi di aggregazione, arresto anticipato, riepilogo della run |
| `model_manager.py` | Costruzione dei modelli, addestramento locale, valutazione |
| `federated_grid_search.py` | Orchestratore della grid search |
| `utils.py` | Serializzazione e primitive omomorfe |
| `PCNAME.py` | Nome della macchina, usato nei percorsi di output |

### Modifiche apportate al framework

Sono documentate nel codice con commenti in italiano nel punto in cui intervengono.

- `model_manager.py` — supporto ai backbone **timm** (DeiT-Tiny), cache del dataset in
  memoria, `pin_memory` e numero di worker del DataLoader configurabili
- `aggregator.py` — **persistenza dei pesi** del modello migliore in `.npz`,
  parametro opzionale `init_seed`, esclusione dei parametri infrastrutturali dal CSV
- `federated_grid_search.py` — numero di processi e file di configurazione da
  variabile d'ambiente, `torch_threads`, pausa termica fra le run (`FL_COOLDOWN`),
  correzione della risoluzione sovrascritta

### Script per la costruzione del dataset

| File | Ruolo |
|---|---|
| `crop_from_labels.py` | Da annotazioni YOLO a ritagli in struttura ImageFolder, con filtri geometrici |
| `dedup_crops.py` | Deduplica invariante al ribaltamento orizzontale |
| `resize_crops.py` | Pre-ridimensionamento a 224×224 sul disco |

### Analisi e strumenti

| File | Ruolo |
|---|---|
| `pannello_v2.ipynb` | Pannello di controllo: stato, profilo termico, avvio, cruscotto, arresto |
| `analisi_finale.ipynb` | Tabelle e figure dai risultati |
| `figure.py` | Genera le quattro figure dal CSV dei risultati |
| `evaluate_test.py` | Valuta i pesi salvati sul test set indipendente |
| `stop.ps1` | Arresto forzato della grid search |

### Dati

```
dataset/raw_capstone/          sorgente Roboflow, annotata in formato YOLO
dataset/road_signs_binary/     7.100 ritagli di addestramento (damaged / healthy)
dataset/road_signs_test/         957 ritagli di test indipendente
```

### Risultati

```
csv_DESKTOP-APC2VEU/
  DESKTOP-APC2VEU/federated_grid_search_results_*.csv   le 216 esecuzioni, una riga ciascuna
  runs/*.csv                                            metriche per round di ogni esecuzione

csv_best_DESKTOP-APC2VEU/
  runs/*_weights.npz                                    pesi del modello migliore, valutati sul test set

csv_pilot_DESKTOP-APC2VEU/                              pilota preliminare a 3 round

figure/                                                 le quattro figure
```

---

## 2. Ambiente

L'ambiente virtuale **non è incluso**: contiene percorsi assoluti e pesa circa 5 GB.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Serve **Python 3.12**. Su Python 3.13 mancano i wheel per numpy 1.26.4, pandas 2.2.1 e
pillow 10.3.0.

PyTorch va installato con il canale CUDA adatto alla propria GPU. Per schede Blackwell
(RTX 50xx) serve **CUDA 12.8 o superiore**, perché i kernel per quell'architettura non
sono presenti nelle build precedenti:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verifica:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Il progetto gira anche su CPU impostando `"device": ["cpu"]` nelle configurazioni, ma
i tempi si allungano di circa un ordine di grandezza.

---

## 3. Ricostruire il dataset da zero

I ritagli in `dataset/road_signs_binary` e `dataset/road_signs_test` sono già presenti.
Per rigenerarli dalla sorgente:

```bat
:: 1. ritagli dalle bounding box annotate, con i filtri geometrici
python crop_from_labels.py --root dataset/raw_capstone --out dataset/road_signs_binary ^
    --splits train,valid ^
    --damaged "Broken,Vandalize,Damaged Traffic Signs - v3 2026-01-31 6-14pm" ^
    --healthy "Clean" --min-size 32 --min-aspect 0.5 --max-aspect 2.0 --pad 0.05

python crop_from_labels.py --root dataset/raw_capstone --out dataset/road_signs_test ^
    --splits test ^
    --damaged "Broken,Vandalize,Damaged Traffic Signs - v3 2026-01-31 6-14pm" ^
    --healthy "Clean" --min-size 32 --min-aspect 0.5 --max-aspect 2.0 --pad 0.05

:: 2. deduplica: prima in prova, poi applicata
python dedup_crops.py --train dataset/road_signs_binary --test dataset/road_signs_test
python dedup_crops.py --train dataset/road_signs_binary --test dataset/road_signs_test --apply

:: 3. pre-ridimensionamento
python resize_crops.py --root dataset/road_signs_binary --size 224
python resize_crops.py --root dataset/road_signs_test   --size 224
```

`--list-classes` su `crop_from_labels.py` elenca le classi dichiarate nel `data.yaml`
con il conteggio delle bounding box: serve a verificare la mappatura prima di ritagliare.

---

## 4. Riprodurre gli esperimenti

### Dal pannello di controllo

Aprire `pannello_v2.ipynb` ed eseguire le celle **1 → 2 → 3 → 4** in ordine. La cella 3
sceglie il profilo termico, la 4 avvia il blocco selezionato, la 5 mostra
l'avanzamento, la 6 ferma l'esecuzione.

La cella 2 verifica che le due griglie siano **simmetriche**: possono differire solo
per `encryption_mode` e per la porta di rete. Se differiscono in altro, la cella 4 si
rifiuta di partire.

### Da riga di comando

```bat
set FL_CONFIG=block1_plaintext.json
set FL_WORKERS=3
set FL_COOLDOWN=30
python federated_grid_search.py
```

| Variabile | Effetto |
|---|---|
| `FL_CONFIG` | file di configurazione da eseguire |
| `FL_WORKERS` | esecuzioni in parallelo |
| `FL_COOLDOWN` | pausa in secondi fra una run e l'altra, applicata a run conclusa |

Le configurazioni già presenti nel CSV dei risultati vengono **saltate** in base a
un'impronta dei parametri: l'esecuzione è interrompibile e riprendibile senza ripetere
lavoro.

### Le due griglie

`block1_plaintext.json` e `block2_encrypted.json` sono identiche tranne che per
`encryption_mode` e `port`:

```
2 backbone × 2 profondità × 3 algoritmi × 3 learning rate × 3 config. client = 108
```

Eseguite entrambe: **216 esecuzioni, 108 coppie appaiate**.

---

## 5. Rigenerare figure e tabelle

```bat
python figure.py
```

Produce in `figure/`:

| File | Contenuto |
|---|---|
| `risultati_finali.png` | I quattro esiti principali |
| `testa2_vs_testa4.png` | Confronto sulla profondità della testa |
| `convergenza.png` | Curve per round: cifrato contro chiaro, e la divergenza |
| `accuratezza_costo.png` | Piano accuratezza / costo della cifratura |

Le tabelle si ottengono da `analisi_finale.ipynb`, che rilegge il CSV a ogni
esecuzione: aggiungendo nuove run, tabelle e figure si aggiornano da sole.

### Valutazione sul test set indipendente

```bat
python evaluate_test.py
```

Ricostruisce il modello dai pesi salvati in `csv_best_*/runs/*_weights.npz` e lo valuta
sui 957 ritagli mai visti.

In modalità cifrata il server detiene ciphertext e non un modello, quindi per quelle
esecuzioni non esistono pesi salvati: la valutazione sul test set riguarda il migliore
modello in chiaro.

---

## 6. Struttura dei CSV dei risultati

`federated_grid_search_results_*.csv` — una riga per esecuzione:

| Colonna | Significato |
|---|---|
| `model_name`, `num_custom_layers` | backbone e profondità della testa |
| `aggregation_algorithm`, `fedprox_mu` | algoritmo di aggregazione |
| `learning_rate`, `batch_size`, `image_size` | iperparametri |
| `num_clients`, `models_percentage` | configurazione federata |
| `global_epoch`, `local_epoch` | round globali ed epoche locali |
| `encryption_mode` | `no_encryption` oppure `direct_encrypted_update` |
| `best_f1`, `best_acc`, `best_prec`, `best_recall`, `best_loss` | metriche migliori |
| `best_round` | round in cui sono state raggiunte |
| `total_duration` | durata dell'esecuzione in secondi |
| `dataframe_path` | percorso del CSV con le metriche per round |

`runs/*.csv` — una riga per round, con `train_loss`, `test_loss`, `test_f1`,
`test_acc`, `test_prec`, `test_recall`.
