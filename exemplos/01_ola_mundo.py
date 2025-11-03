"""
Exemplo 1: Olá Mundo - Neural Bifurcation

O exemplo mais simples possível.
Treina MNIST com objetivo 'balanced' e compara com tradicional.

Tempo: ~5 minutos
"""

import tensorflow as tf
from tensorflow import keras
from neuralbifurcation import MultiObjectiveDetector

print("="*60)
print("🎯 Exemplo 1: Olá Mundo - Neural Bifurcation")
print("="*60)
print()

# ============================================================================
# 1. CARREGAR DADOS (MNIST)
# ============================================================================

print("📦 Carregando MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Pegar subset pra ser rápido (opcional)
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

print(f"   Train: {len(X_train)} imagens")
print(f"   Test: {len(X_test)} imagens")
print()

# ============================================================================
# 2. CRIAR MODELO
# ============================================================================

print("🏗️ Criando modelo...")
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("   Modelo criado!")
print()

# ============================================================================
# 3. TREINAR COM NEURAL BIFURCATION
# ============================================================================

print("🎯 Treinando com Neural Bifurcation (objetivo: balanced)")
print("-"*60)

# A MÁGICA ACONTECE AQUI! 🎩✨
detector = MultiObjectiveDetector(
    objective='balanced',
    patience=5,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=128,
    callbacks=[detector],
    verbose=0  # Framework já mostra progresso
)

print()
print("="*60)
print("✅ TREINAMENTO COMPLETO!")
print("="*60)
print()

# ============================================================================
# 4. RESULTADOS
# ============================================================================

# Avaliar no test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("📊 RESULTADOS:")
print(f"   Melhor Epoch: {detector.best_epoch + 1}")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Custo Total: ${detector.total_cost:.2f}")
print()

# Comparar com treino tradicional
print("💡 COMPARAÇÃO:")
print(f"   Neural Bifurcation: {detector.best_epoch + 1} epochs, ${detector.total_cost:.2f}")
print(f"   Tradicional (20 epochs): 20 epochs, ${20 * detector.cost_per_epoch:.2f}")
print(f"   Economia: {((20 - detector.best_epoch - 1) / 20 * 100):.0f}%")
print()

print("="*60)
print("🎉 Parabéns! Você rodou seu primeiro modelo com Neural Bifurcation!")
print()
print("📚 Próximos passos:")
print("   1. Rode: python exemplos/02_comparar_objetivos.py")
print("   2. Teste no seu próprio modelo!")
print("   3. Me mande feedback :)")
print("="*60)
