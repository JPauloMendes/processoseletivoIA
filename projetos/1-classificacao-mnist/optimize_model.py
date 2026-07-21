import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

print("Iniciando carregamento do modelo original (model.h5)...")
model = tf.keras.models.load_model("model.h5")

print("Preparando o conversor para TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("Aplicando otimização (Dynamic Range Quantization)...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Convertendo o modelo (isso pode levar alguns segundos)...")
tflite_model = converter.convert()

print("Salvando o modelo otimizado...")
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("\n=== Otimização Concluída ===")
print("Modelo reduzido e salvo com sucesso como 'model.tflite'!")