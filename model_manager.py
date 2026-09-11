try:
    # Fa usare a Python lo store certificati del sistema operativo invece del bundle
    # certifi. Necessario dietro antivirus/firewall che intercettano il traffico TLS:
    # senza, il download dei pesi pre-addestrati da Hugging Face fallisce con
    # CERTIFICATE_VERIFY_FAILED. Non disattiva la verifica, la delega all'OS.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import time
import numpy as np
from typing import Tuple, Dict, List


# Definizione della rete ConvNet
class ConvNet(torch.nn.Module):
    def __init__(self, image_size=32, hidden1=64, hidden2=32, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        conv_out_size = ((image_size - 7) // 3) + 1
        flattened_size = 4 * conv_out_size * conv_out_size
        self.fc1 = torch.nn.Linear(flattened_size, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.conv1(x);
        x = x * x
        x = x.view(x.size(0), -1)
        x = self.fc1(x);
        x = x * x
        x = self.fc2(x);
        x = x * x
        x = self.fc3(x)
        return x


class CachedImageFolder(Dataset):
    """
    Come ImageFolder, ma decodifica ogni immagine una sola volta e la tiene in RAM.

    Il framework rilegge lo stesso split molte volte: local_epoch epoche di training
    piu' una passata di validazione sul train e una sul valid, per ogni round globale.
    Con 20 round significa decodificare ogni JPEG un centinaio di volte per run, un
    costo interamente a carico della CPU che lascia la GPU in attesa.

    Qui la decodifica avviene una volta in fase di costruzione; a ogni accesso resta
    solo la trasformazione in tensore e la normalizzazione. Le immagini sono tenute
    come array uint8 (3 byte per pixel): l'intero dataset da 7.100 ritagli a 224x224
    occupa circa 1 GB per processo worker.
    """

    def __init__(self, root: str, transform):
        base = ImageFolder(root=root)
        self.transform = transform
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.images: List[Image.Image] = []
        self.targets: List[int] = []
        for path, label in base.samples:
            with Image.open(path) as handle:
                self.images.append(handle.convert('RGB').copy())
            self.targets.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.transform(self.images[index]), self.targets[index]


class ModelManager:
    """
    Gestisce la creazione, il training e la valutazione di diversi modelli di rete neurale.
    """

    def __init__(self, config: Dict, dataset_path: str):
        self.config = config
        self.model_name = config.get('model_name', 'ResNet18')
        self.dataset_path = dataset_path
        self.num_classes = config.get('num_classes', 2)
        self.device = self._get_device(config.get('device', 'cuda'))
        self.model = self._initialize_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.transform_pipeline = self._get_transforms()
        self.calibration_term = torch.zeros(self.num_classes, device=self.device)
        self._dataset_cache: Dict[str, Dataset] = {}

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == 'cuda' and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # GPU 1 è usata dal display server, la saltiamo
            usable_gpus = [i for i in range(num_gpus) if i != 1]
            if usable_gpus:
                worker_id = self.config.get('worker_id', 0) or 0
                gpu_index = usable_gpus[worker_id % len(usable_gpus)]
                return torch.device(f"cuda:{gpu_index}")
        return torch.device("cpu")

    def _get_transforms(self):
        image_size = self.config.get('image_size', 224)
        if self.model_name == 'ConvNet':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        else:  # Per ResNet, GoogLeNet, etc.
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _initialize_model(self) -> nn.Module:
        print(f"Initializing model: {self.model_name}")
        if 'ResNet' in self.model_name:
            return self._initialize_resnet()
        elif self.model_name == 'GoogLeNet':
            return self._initialize_googlenet()
        elif self.model_name == 'AlexNet':
            return self._initialize_alexnet()
        elif self.model_name == 'ConvNet':
            return self._initialize_convnet()
        elif self.model_name in ('DeiT-Tiny', 'ViT-Tiny'):
            return self._initialize_timm_vit()
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def _initialize_convnet(self) -> nn.Module:
        return ConvNet(
            image_size=self.config.get('image_size', 32),
            hidden1=self.config.get('convnet_hidden1', 64),
            hidden2=self.config.get('convnet_hidden2', 32),
            num_classes=self.num_classes
        )

    def _initialize_alexnet(self) -> nn.Module:
        """Initializes an AlexNet model with correct classifier handling."""
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print("num_custom_layers is 0. All AlexNet parameters will be trainable.")
        else:
            print("Freezing pre-trained AlexNet layers.")

        # Congela/Scongela i layer della base convoluzionale
        for param in model.features.parameters():
            param.requires_grad = train_full_model

        # --- MODIFICA CHIAVE: Logica corretta per AlexNet ---
        if num_custom_layers > 0:
            # Sostituisci l'intero blocco. L'input deve essere quello che esce dalla parte 'features'.
            # Per AlexNet, è 256 canali * 6x6 spatial size = 9216.
            in_features_for_block = 256 * 6 * 6
            custom_classifier = self._create_custom_classifier(in_features_for_block, self.num_classes,
                                                               num_custom_layers)
            model.classifier = nn.Sequential(*custom_classifier)
        else:
            # L'ultimo layer originale (indice 6) ha in_features=4096
            last_layer_in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(last_layer_in_features, self.num_classes)

        for param in model.classifier.parameters():
            param.requires_grad = True

        return model


    def _initialize_timm_vit(self) -> nn.Module:
        """Initializes a Vision Transformer backbone (DeiT-Tiny / ViT-Tiny) from timm.

        The backbone is frozen and only the custom head is trained, exactly as for the
        CNN backbones. With an embedding dimension of 192, the resulting head is
        markedly smaller than the ResNet18 one (512), which directly reduces the number
        of parameters to encrypt under the Paillier scheme.
        """
        import timm

        timm_names = {
            'DeiT-Tiny': 'deit_tiny_patch16_224',
            'ViT-Tiny': 'vit_tiny_patch16_224',
        }
        model = timm.create_model(timm_names[self.model_name], pretrained=True)

        num_features = model.head.in_features  # 192 for both Tiny variants
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print(f"num_custom_layers is 0. All {self.model_name} parameters will be trainable.")
        else:
            print(f"Freezing pre-trained {self.model_name} layers. Only the custom classifier will be trained.")

        for param in model.parameters():
            param.requires_grad = train_full_model

        if num_custom_layers > 0:
            layers = self._create_custom_classifier(num_features, self.num_classes, num_custom_layers)
            model.head = nn.Sequential(*layers)
        else:
            model.head = nn.Linear(num_features, self.num_classes)

        for param in model.head.parameters():
            param.requires_grad = True

        return model

    def _initialize_googlenet(self) -> nn.Module:
        """Initializes a GoogLeNet model, with logic for custom layers vs full fine-tuning."""
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features  # GoogLeNet ha 1024 features in input al FC
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print("num_custom_layers is 0. All GoogLeNet parameters will be trainable.")
        else:
            print(f"Freezing pre-trained GoogLeNet layers. Only the custom classifier will be trained.")

        for param in model.parameters():
            param.requires_grad = train_full_model

        if num_custom_layers > 0:
            layers = self._create_custom_classifier(num_features, self.num_classes, num_custom_layers)
            model.fc = nn.Sequential(*layers)
        else:
            model.fc = nn.Linear(num_features, self.num_classes)

        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def _initialize_resnet(self) -> nn.Module:
        if self.model_name == 'ResNet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.model_name == 'ResNet34':
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif self.model_name == 'ResNet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.model_name == 'ResNet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"ResNet model '{self.model_name}' not supported.")

        num_features = model.fc.in_features
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print("num_custom_layers is 0. All ResNet parameters will be trainable.")
        else:
            print(f"Freezing pre-trained ResNet layers. Only the custom classifier will be trained.")

        for param in model.parameters():
            param.requires_grad = train_full_model

        if num_custom_layers > 0:
            layers = self._create_custom_classifier(num_features, self.num_classes, num_custom_layers)
            model.fc = nn.Sequential(*layers)
        else:
            model.fc = nn.Linear(num_features, self.num_classes)

        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def _create_custom_classifier(self, in_features, out_features, num_layers):
        layers = []
        if num_layers < 2:
            layers.append(nn.Linear(in_features, out_features))
        else:
            hidden_size = 256
            layers.extend([nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Dropout(0.4)])
            in_size = hidden_size
            for _ in range(num_layers - 2):
                out_size = in_size // 2
                layers.extend([nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(0.4)])
                in_size = out_size
            layers.append(nn.Linear(in_size, out_features))
        return layers

    def _get_trainable_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_weights(self) -> List[np.ndarray]:
        return [param.data.cpu().numpy() for param in self._get_trainable_parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        trainable_params = self._get_trainable_parameters()
        for param, weight_array in zip(trainable_params, weights):
            param.data.copy_(torch.from_numpy(weight_array))

    def get_samples_per_class(self) -> np.ndarray:
        samples = torch.zeros(self.num_classes, device=self.device)
        try:
            train_loader = self._get_dataloader('train', batch_size=32)
            for _, labels in train_loader:
                for label in labels:
                    samples[label.item()] += 1
        except FileNotFoundError:
            print("Warning: Training data not found during sample count.")
        return samples.cpu().numpy()

    def set_calibration_term(self, calibration_val: np.ndarray):
        self.calibration_term = torch.from_numpy(calibration_val).float().to(self.device)

    def _get_dataloader(self, split: str, batch_size: int) -> DataLoader:
        data_path = os.path.join(self.dataset_path, split)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Dataset directory not found for split '{split}': {data_path}")
        # La cache e' per (istanza, split): costruita al primo accesso e riusata per
        # tutti i round successivi. Disattivabile con "cache_dataset": false nel JSON.
        if self.config.get('cache_dataset', True):
            if split not in self._dataset_cache:
                self._dataset_cache[split] = CachedImageFolder(data_path, self.transform_pipeline)
            dataset = self._dataset_cache[split]
        else:
            dataset = ImageFolder(root=data_path, transform=self.transform_pipeline)

        on_cuda = self.device.type == 'cuda'
        # num_workers > 0 su Windows genera processi a ogni creazione del DataLoader,
        # che qui avviene piu' volte per round: conviene solo se il costo di decodifica
        # e' alto. Default 0, configurabile per poterlo misurare.
        num_workers = int(self.config.get('dataloader_workers', 0))

        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': (split == 'train'),
            'pin_memory': on_cuda,          # trasferimento host->device piu' rapido
            'num_workers': num_workers,
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 4

        return DataLoader(dataset, **loader_kwargs)

    def train(self, epochs: int, lr: float, batch_size: int,
              algorithm: str = "FedAvg", global_weights: List[np.ndarray] = None,
              mu: float = 0.0) -> Tuple[float, Dict, float, int]:
        trainable_params = self._get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        train_loader = self._get_dataloader('train', batch_size)

        global_params_tensor = None
        if algorithm == 'FedProx' and global_weights is not None:
            global_params_tensor = [torch.from_numpy(w).to(self.device) for w in global_weights]

        final_train_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = self._run_training_epoch(train_loader, optimizer, algorithm, global_params_tensor, mu)
            final_train_loss = epoch_loss

        _, metric_score, _, dataset_size = self.validate(batch_size, split='train')
        return 0.0, metric_score, final_train_loss, dataset_size

    def _run_training_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                            algorithm: str, global_params: List[torch.Tensor], mu: float) -> float:
        self.model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)

            if algorithm == 'FedLC':
                loss = self.criterion(outputs - self.calibration_term.unsqueeze(0), labels)
            else:
                loss = self.criterion(outputs, labels)

            if algorithm == 'FedProx' and mu > 0 and global_params:
                prox_term = 0.0
                local_params = self._get_trainable_parameters()
                for local_w, global_w in zip(local_params, global_params):
                    prox_term += (local_w - global_w).norm(2)
                loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(loader.dataset)

    def validate(self, batch_size: int, split: str = 'valid') -> Tuple[float, Dict, float, int]:
        try:
            data_loader = self._get_dataloader(split, batch_size)
        except FileNotFoundError:
            return 0.0, {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}, 0.0, 0

        self.model.eval()
        all_preds, all_labels = [], []
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(data_loader.dataset)
        metrics = {
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        return val_loss, metrics, 0.0, len(data_loader.dataset)