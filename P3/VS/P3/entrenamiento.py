from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(42)

CORPUS_PATH = Path("dataset_funciones.c")

if not CORPUS_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontro '{CORPUS_PATH}'. "
        "Coloca dataset_funciones.c en la misma carpeta que entrenamiento.py."
    )

CORPUS = CORPUS_PATH.read_text(encoding="utf-8")
print(f"Dataset cargado correctamente: {len(CORPUS):,} caracteres")

# -----------------------------------------------------------------------------
# 2. Crear vocabulario a nivel de caracteres
# -----------------------------------------------------------------------------
chars = sorted(set(CORPUS))
stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
itos: dict[int, str] = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(chars)

print(f"Tamano del vocabulario: {VOCAB_SIZE} caracteres unicos")


def encode(texto: str) -> list[int]:
    """Convierte una cadena de texto en una lista de indices."""
    return [stoi[caracter] for caracter in texto if caracter in stoi]


def decode(indices: list[int]) -> str:
    """Convierte una lista de indices en una cadena de texto."""
    return "".join(itos[indice] for indice in indices)


SEQ = np.array(encode(CORPUS), dtype=np.int64)
print(f"Secuencia codificada: {len(SEQ):,} tokens")

# -----------------------------------------------------------------------------
# 3. Hiperparametros del modelo
# -----------------------------------------------------------------------------
BLOCK_SIZE = 64      # Longitud de cada ventana de contexto
EMBED_DIM = 64       # Tamano del vector de embedding por caracter
HIDDEN = 128         # Unidades ocultas de la capa SimpleRNN
EPOCHS = 80          # Numero de epocas de entrenamiento
BATCH_SIZE = 32      # Tamano de lote

if len(SEQ) < BLOCK_SIZE + 1:
    raise ValueError(
        f"El corpus es demasiado corto ({len(SEQ)} tokens) "
        f"para BLOCK_SIZE={BLOCK_SIZE}. Amplia el corpus o reduce BLOCK_SIZE."
    )

# -----------------------------------------------------------------------------
# 4. Construir ventanas de entrenamiento
# -----------------------------------------------------------------------------
print("Construyendo ventanas de entrenamiento...")
X_rows: list[np.ndarray] = []
Y_rows: list[np.ndarray] = []

for i in range(0, len(SEQ) - BLOCK_SIZE):
    X_rows.append(SEQ[i : i + BLOCK_SIZE])
    Y_rows.append(SEQ[i + 1 : i + 1 + BLOCK_SIZE])

X = np.stack(X_rows)   # Forma: (N, BLOCK_SIZE)
Y = np.stack(Y_rows)   # Forma: (N, BLOCK_SIZE)
print(f"Entradas X: {X.shape}  Salidas Y: {Y.shape}")

# -----------------------------------------------------------------------------
# 5. Arquitectura RNN Vanilla
# -----------------------------------------------------------------------------
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(BLOCK_SIZE,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        tf.keras.layers.SimpleRNN(
            HIDDEN,
            activation="tanh",
            return_sequences=True,
        ),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(VOCAB_SIZE)
        ),
    ],
    name="rnn_vanilla_char_codigo_c",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.summary()

# -----------------------------------------------------------------------------
# 6. Entrenamiento del modelo
# -----------------------------------------------------------------------------
print(f"\nIniciando entrenamiento: {EPOCHS} epocas, batch_size={BATCH_SIZE}")

history = model.fit(
    X,
    Y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

print(f"\nEpocas completadas : {len(history.history['loss'])}")
print(f"Perdida inicial    : {history.history['loss'][0]:.4f}")
print(f"Perdida final      : {history.history['loss'][-1]:.4f}")

# -----------------------------------------------------------------------------
# 7. Funciones de autocompletado
# -----------------------------------------------------------------------------

def complete(prompt: str, max_new: int = 120, temperature: float = 0.75) -> str:
    """
    Genera texto nuevo a partir de un prefijo.
    El modelo usa los ultimos BLOCK_SIZE caracteres como contexto.
    """
    ids = encode(prompt)
    if not ids:
        ids = [0]
    rng = np.random.default_rng(42)

    for _ in range(max_new):
        x = np.array(ids[-BLOCK_SIZE:], dtype=np.int64)
        if x.shape[0] < BLOCK_SIZE:
            pad = np.full(BLOCK_SIZE - x.shape[0], ids[0], dtype=np.int64)
            x = np.concatenate([pad, x])

        logits = model(x.reshape(1, BLOCK_SIZE), training=False).numpy()[0, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))

    return decode(ids)


def suggest(prefix: str, n: int = 5, max_new: int = 80) -> list[str]:
    """
    Genera varias sugerencias unicas para un mismo prefijo.
    La temperatura cambia ligeramente para aumentar la diversidad.
    """
    seen: set[str] = set()
    out: list[str] = []

    for i in range(n * 4):
        texto = complete(prefix, max_new=max_new, temperature=0.65 + 0.05 * i)
        linea = (prefix + texto[len(prefix):].split("\n")[0])[:100]
        if linea not in seen and len(linea) > len(prefix):
            seen.add(linea)
            out.append(linea)
        if len(out) >= n:
            break

    return out


# -----------------------------------------------------------------------------
# 8. Prueba rapida del autocompletado
# -----------------------------------------------------------------------------
print("\n--- Prueba rapida de autocompletado ---")
test_prompts = [
    "float horasAMinutos",
    "char minAMas",
    "bool multiploDeCinco",
    "int sumarEnteros",
]

for prompt in test_prompts:
    resultado = complete(prompt, max_new=60, temperature=0.75)
    print(f"\nPrompt : {repr(prompt)}")
    print(f"Salida : {repr(resultado)}")

print("\n--- Sugerencias multiples ---")
for sugerencia in suggest("int sumar", n=3):
    print(" -", sugerencia)

# -----------------------------------------------------------------------------
# 9. Guardar modelo y metadatos
# -----------------------------------------------------------------------------
DEPLOY_DIR = Path("rnn-keras-autocomplete")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

model.save(DEPLOY_DIR / "model.keras")

meta = {
    "block_size": BLOCK_SIZE,
    "chars": chars,
    "vocab_size": VOCAB_SIZE,
    "embed_dim": EMBED_DIM,
    "hidden": HIDDEN,
    "epochs": EPOCHS,
    "loss_final": float(history.history["loss"][-1]),
}
(DEPLOY_DIR / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\nModelo guardado en : {(DEPLOY_DIR / 'model.keras').resolve()}")
print(f"Metadatos en       : {(DEPLOY_DIR / 'meta.json').resolve()}")
print("Entrenamiento finalizado correctamente.")
