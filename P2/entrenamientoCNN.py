import os
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. CONFIGURACIÓN
ruta_dataset = "dataset_limpio"
tamanio_lote = 32
tamanio_img = (32, 32) # ¡Nuevo tamaño de las imágenes cocinadas!

print("--- CARGANDO DATOS DE FORMA INTELIGENTE ---")
# 2. CARGA DE DATOS SEGUROS PARA LA RAM
# Esto divide automáticamente 80% entrenamiento y 20% prueba
train_ds = tf.keras.utils.image_dataset_from_directory(
  ruta_dataset,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=tamanio_img,
  batch_size=tamanio_lote)

val_ds = tf.keras.utils.image_dataset_from_directory(
  ruta_dataset,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=tamanio_img,
  batch_size=tamanio_lote)

clases = train_ds.class_names
print(f"Clases detectadas automáticamente: {clases}")

# Optimización para que la CPU lea datos mientras la red neuronal entrena
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n--- CONSTRUYENDO EL MODELO V2.0 ---")
# 3. DATA AUGMENTATION (El secreto para llegar al 80%)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"), # Voltea al animal como un espejo
  layers.RandomRotation(0.1),      # Lo rota ligeramente
  layers.RandomZoom(0.1),          # Le hace un poco de zoom
])

# 4. ARQUITECTURA DE LA RED
modelo = models.Sequential([
  # Aplicamos el aumento de datos solo durante el entrenamiento
  data_augmentation,
  
  # Normalización automática (convierte píxeles de 0-255 a 0.0-1.0)
  layers.Rescaling(1./255, input_shape=(32, 32, 3)),

  # Capas Convolucionales (Agregamos una capa extra por tener imágenes de 32x32)
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(128, 3, padding='same', activation='relu'), # Capa nueva!
  layers.MaxPooling2D(),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  
  # Capa de salida: 5 clases
  layers.Dense(len(clases), activation='softmax')
])

# Usamos sparse_categorical_crossentropy porque este nuevo método 
# de carga numera las clases (0, 1, 2, 3, 4) en lugar de usar vectores
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. ENTRENAMIENTO
print("\n--- INICIANDO ENTRENAMIENTO (20 ÉPOCAS) ---")
historial = modelo.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20 # Subimos a 20 épocas porque hay más datos
)

# 6. GUARDAR EL MODELO
modelo.save("Modelos/animales5.keras")
print("\n¡Entrenamiento finalizado y modelo guardado como 'modelo_animales_gris.keras'!")