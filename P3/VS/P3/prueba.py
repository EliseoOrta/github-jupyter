from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# Carpeta donde entrenamiento.py guarda el modelo y los metadatos
ROOT = Path("rnn-keras-autocomplete")
MODEL_PATH = ROOT / "model.keras"
META_PATH = ROOT / "meta.json"

if not META_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontro '{META_PATH}'. Ejecuta primero: python entrenamiento.py"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontro '{MODEL_PATH}'. Ejecuta primero: python entrenamiento.py"
    )


# Cargar vocabulario y configuracion usada durante el entrenamiento
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
BLOCK_SIZE = int(meta["block_size"])
chars = meta["chars"]
stoi = {caracter: indice for indice, caracter in enumerate(chars)}
itos = {indice: caracter for caracter, indice in stoi.items()}


# Cargar modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)


def encode(texto: str) -> list[int]:
    """Convierte texto a indices segun el vocabulario entrenado."""
    return [stoi[caracter] for caracter in texto if caracter in stoi]


def decode(indices: list[int]) -> str:
    """Convierte indices del vocabulario a texto."""
    return "".join(itos[indice] for indice in indices)


def completado(prompt: str, max_new: int = 120, temperature: float = 0.75) -> str:
    """
    Genera caracteres nuevos a partir de un prefijo.

    prompt: texto inicial escrito por el usuario.
    max_new: cantidad maxima de caracteres nuevos.
    temperature: controla creatividad. Menor valor = salida mas estable.
    """
    ids = encode(prompt)

    if not ids:
        ids = [0]

    generador = np.random.default_rng(42)

    for _ in range(max_new):
        contexto = np.array(ids[-BLOCK_SIZE:], dtype=np.int64)

        if contexto.shape[0] < BLOCK_SIZE:
            relleno = np.full(BLOCK_SIZE - contexto.shape[0], ids[0], dtype=np.int64)
            contexto = np.concatenate([relleno, contexto])

        logits = model(contexto.reshape(1, BLOCK_SIZE), training=False).numpy()[0, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()

        probabilidades = np.exp(logits)
        probabilidades = probabilidades / probabilidades.sum()

        siguienteIndice = int(generador.choice(len(probabilidades), p=probabilidades))
        ids.append(siguienteIndice)

    return decode(ids)


def CompletaTodo(prompt: str, max_new: int = 160, temperature: float = 0.75) -> str:
    """
    Genera codigo y corta la salida cuando encuentra el cierre de una funcion.
    El corte se realiza cuando se cierra la llave principal.
    """
    resultado = completado(prompt, max_new=max_new, temperature=temperature)
    texto_generado = resultado[len(prompt):]

    profundidad_llaves = 0
    ya_abrio_funcion = False
    punto_corte = len(texto_generado)

    for posicion, caracter in enumerate(texto_generado):
        if caracter == "{":
            profundidad_llaves += 1
            ya_abrio_funcion = True
        elif caracter == "}":
            profundidad_llaves -= 1
            if ya_abrio_funcion and profundidad_llaves == 0:
                punto_corte = posicion + 1
                break

    return prompt + texto_generado[:punto_corte]


# Prompts de prueba para el nuevo dataset de funciones basicas en C
test_prompts = [
    "float horasAMinutos",
    "char minAMas",
    "bool multiploDeCinco",
    "int sumarEnteros",
    "float calcularArea",
    "int contar",
]


print("Prueba local del asistente RNN para codigo C")
print("Modelo cargado desde:", MODEL_PATH)
print("Vocabulario:", len(chars), "caracteres")

for prompt in test_prompts:
    print("\nPrompt:", repr(prompt))
    print("Salida:")
    print(CompletaTodo(prompt, max_new=160, temperature=0.70))
