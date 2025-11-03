"""
Exemplo 2: Comparar Objetivos

Treina o MESMO modelo com os 4 objetivos diferentes
e compara os resultados lado a lado.

Tempo: ~10 minutos
"""

import tensorflow as tf
from tensorflow import keras
from neuralbifurcation import MultiObjectiveDetector
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("🎯 Exemplo 2: Comparando os 4 Objetivos")
print("="*60)
print()

# ============================================================================
# 1. PREPARAR DADOS
# ============================================================================

print("📦 Carregando MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Subset
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
print()

# ============================================================================
# 2. FUNÇÃO PARA CRIAR MODELO (vamos criar 4x)
# ============================================================================

def criar_modelo():
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
    return model

# ============================================================================
# 3. TREINAR COM CADA OBJETIVO
# ============================================================================

objetivos = ['robustness', 'accuracy', 'cost', 'balanced']
resultados = {}

for objetivo in objetivos:
    print("="*60)
    print(f"🎯 Treinando com objetivo: {objetivo.upper()}")
    print("="*60)
    
    # Criar modelo novo
    model = criar_modelo()
    
    # Detector
    detector = MultiObjectiveDetector(
        objective=objetivo,
        patience=5,
        cost_per_epoch=2.0,
        verbose=1
    )
    
    # Treinar
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=128,
        callbacks=[detector],
        verbose=0
    )
    
    # Avaliar
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Salvar resultados
    resultados[objetivo] = {
        'melhor_epoch': detector.best_epoch + 1,
        'test_accuracy': test_acc,
        'custo': detector.total_cost,
        'detector': detector
    }
    
    print()

# ============================================================================
# 4. COMPARAR RESULTADOS
# ============================================================================

print("="*60)
print("📊 COMPARAÇÃO FINAL")
print("="*60)
print()

print(f"{'Objetivo':<15} | {'Epoch':<6} | {'Accuracy':<10} | {'Custo':<8}")
print("-"*60)

for obj, res in resultados.items():
    print(f"{obj:<15} | {res['melhor_epoch']:<6} | "
          f"{res['test_accuracy']*100:>8.2f}% | ${res['custo']:<6.2f}")

print()

# ============================================================================
# 5. VISUALIZAR (OPCIONAL)
# ============================================================================

print("📈 Gerando gráfico comparativo...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Accuracy
axes[0].bar(objetivos, [resultados[obj]['test_accuracy']*100 for obj in objetivos])
axes[0].set_title('Accuracy no Test Set')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([90, 100])

# Plot 2: Epochs
axes[1].bar(objetivos, [resultados[obj]['melhor_epoch'] for obj in objetivos])
axes[1].set_title('Epochs Treinados')
axes[1].set_ylabel('Epochs')

# Plot 3: Custo
axes[2].bar(objetivos, [resultados[obj]['custo'] for obj in objetivos])
axes[2].set_title('Custo Total')
axes[2].set_ylabel('Custo ($)')

plt.tight_layout()
plt.savefig('comparacao_objetivos.png', dpi=150, bbox_inches='tight')
print("   Gráfico salvo: comparacao_objetivos.png")
print()

# ============================================================================
# 6. ANÁLISE
# ============================================================================

print("💡 ANÁLISE:")
print()

# Melhor accuracy
melhor_acc = max(resultados.items(), key=lambda x: x[1]['test_accuracy'])
print(f"🏆 Melhor Accuracy: {melhor_acc[0]} ({melhor_acc[1]['test_accuracy']*100:.2f}%)")

# Mais econômico
mais_barato = min(resultados.items(), key=lambda x: x[1]['custo'])
print(f"💰 Mais Econômico: {mais_barato[0]} (${mais_barato[1]['custo']:.2f})")

# Melhor custo-benefício
for obj, res in resultados.items():
    res['roi'] = res['test_accuracy'] / res['custo']

melhor_roi = max(resultados.items(), key=lambda x: x[1]['roi'])
print(f("⚖️ Melhor Custo-Benefício: {melhor_roi[0]} "
      f"({melhor_roi[1]['roi']:.3f} acc/dólar)")

print()
print("="*60)
print("✅ Comparação completa!")
print()
print("🤔 Qual objetivo você escolheria?")
print("   Depende do seu caso de uso!")
print("="*60)
