# IMPORTOWANIE BIBLIOTEK
import torch
import torch.nn as nn
# CONFIG
import config


# POJEDYŃCZY BLOK DO SKŁADANIA +---------------------------------------------------

class LinearBlock(nn.Module):
    """
    One fully-connected layer with optional BatchNorm, ReLU activation,
    and Dropout. Used to build the hidden layers of the MLP.
    """

    def __init__(self, in_features: int, out_features: int, dropout_rate: float, use_batch_norm: bool, use_swish_activation: bool):
        super().__init__()

        layers = [nn.Linear(in_features, out_features)]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features)) # BATCHNORM

        if use_swish_activation:
            layers.append(nn.SiLU()) # AKTYWACJA SWISH
        else:
            layers.append(nn.ReLU()) # AKTYWACJA RELU

        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate)) # DROPOUT

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# KLASA SIECI +-----------------------------------------------------------------------

class PulsarMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary pulsar classification.

    Architecture (default config):
        Input(8) -> Linear(64) -> BN -> ReLU -> Dropout
                 -> Linear(32) -> BN -> ReLU -> Dropout
                 -> Linear(16) -> BN -> ReLU -> Dropout
                 -> Linear(1)   <- raw logit, NO sigmoid here

    The sigmoid is applied inside BCEWithLogitsLoss during training,
    and explicitly during inference (see predict() below).

    Args:
        input_size     : number of input features  (default: config.INPUT_SIZE)
        hidden_sizes   : list of hidden layer widths (default: config.HIDDEN_SIZES)
        output_size    : 1 for binary classification (default: config.OUTPUT_SIZE)
        dropout_rate   : dropout probability         (default: config.DROPOUT_RATE)
        use_batch_norm : whether to use BatchNorm1d  (default: config.USE_BATCH_NORM)
    """

    def __init__(
        self,
        input_size:     int   = config.INPUT_SIZE,
        hidden_sizes:   list  = config.HIDDEN_SIZES,
        output_size:    int   = config.OUTPUT_SIZE,
        dropout_rate:   float = config.DROPOUT_RATE,
        use_batch_norm: bool  = config.USE_BATCH_NORM,
        use_swish_activation: bool = config.USE_SWISH_ACTIVATION,
    ):
        super().__init__()

        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes

        # BUDUJE WARSTWY
        # za pomocą wartości z config
        # i LinearBlock()

        hidden_layers = []
        in_features = input_size
        for out_features in hidden_sizes:
            hidden_layers.append(
                LinearBlock(in_features, out_features, dropout_rate, use_batch_norm, use_swish_activation)
            )
            in_features = out_features

        self.hidden = nn.Sequential(*hidden_layers)

        # WARSTWA WYJŚCIOWA
        self.output = nn.Linear(in_features, output_size)

        # INICJALIZACJA WAG ( zarówno dla ReLU jak i dla Swish ten sam inicjalizator)
        self._init_weights()

    #INICJALIZACJA WAG
    def _init_weights(self) -> None:
        """Kaiming (He) init for ReLU networks — better than default Xavier here."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # FORWARD
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits. Shape: (batch_size, 1)."""
        x = self.hidden(x)
        return self.output(x)

    # PREDYKCJA DLA JEDNEGO PRZYKŁADU
    def predict(self, x: torch.Tensor, threshold: float = config.THRESHOLD) -> torch.Tensor:
        """
        Inference helper. Returns binary predictions (0 or 1).
        Applies sigmoid to logits then thresholds.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            return (probs >= threshold).float()

    # DAJE KLASYFIKACJĘ OBIEKTU DLA JEDNEGO PRZYKŁADU
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class-1 probabilities. Useful for ROC-AUC and threshold tuning.
        """
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    # WYPISUJE
    def summary(self) -> None:
        """Print a compact model summary."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'='*45}")
        print(f"  PulsarMLP")
        print(f"{'='*45}")
        print(f"  Input size    : {self.input_size}")
        print(f"  Hidden layers : {self.hidden_sizes}")
        print(f"  Output size   : 1  (binary logit)")
        print(f"  Total params  : {total:,}")
        print(f"  Trainable     : {trainable:,}")
        print(f"{'='*45}\n")
        print(self)

    # ZAPISUJE KSZTAŁT OBECNEGO MODELU
    def save_model_shape(self) -> None:
        config.USED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.USED_CONFIGS_DIR/"model_shape.txt","w") as file:
            total     = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

            file.write(f"\n{'='*45}\n")
            file.write(f"  PulsarMLP\n")
            file.write(f"{'='*45}\n")
            file.write(f"  Input size    : {self.input_size}\n")
            file.write(f"  Hidden layers : {self.hidden_sizes}\n")
            file.write(f"  Output size   : 1  (binary logit)\n")
            file.write(f"  Total params  : {total:,}\n")
            file.write(f"  Trainable     : {trainable:,}\n")
            file.write(f"{'='*45}\n\n")
            file.write(str(self))


# SZYBKI TEST +------------------------------------------------------------

if __name__ == "__main__":
    model = PulsarMLP()
    model.summary()

    # Forward pass with a random batch
    dummy  = torch.randn(config.BATCH_SIZE, config.INPUT_SIZE)
    logits = model(dummy)
    preds  = model.predict(dummy)
    probs  = model.predict_proba(dummy)

    print(f"\nDummy batch  : {tuple(dummy.shape)}")
    print(f"Logits shape : {tuple(logits.shape)}")
    print(f"Preds shape  : {tuple(preds.shape)}   unique: {preds.unique().tolist()}")
    print(f"Probs range  : [{probs.min():.3f}, {probs.max():.3f}]")