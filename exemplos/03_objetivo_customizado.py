"""
Exemplo 3: Objetivo Customizado

Mostra como criar seu próprio objetivo personalizado.

Caso de uso: Você quer otimizar para PRECISION (não accuracy),
porque falsos positivos são mais caros que falsos negativos.

Tempo: ~15 minutos
"""

import tensorflow as tf
from tensorflow import keras
from neuralbifurcation import MultiObjectiveDetector, ObjectiveConfig

print("="*60)
print("🎯 Exemplo 3: Objetivo Customizado")
print("="*60)
print()

# ============================================================================
# 1. CARREGAR DADOS
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
# 2. CRIAR MODELO (com precision como métrica)
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
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),  # ← Importante!
        keras.metrics.Recall(name='recall')
    ]
)
print("   Modelo criado!")
print()

# ============================================================================
# 3. CRIAR OBJETIVO CUSTOMIZADO
# ============================================================================

print("🎨 Criando objetivo customizado...")
print()

# Definir configuração do objetivo
objetivo_precision = ObjectiveConfig(
    name='precision_focused',
    description='Maximizar precision (minimizar falsos positivos)',
    
    # Métrica primária: precision (precisa estar em model.metrics!)
    primary_metric='precision',
    
    # Função de decisão: 70% precision + 20% recall + 10% accuracy
    decision_function=lambda metrics: (
        0.7 * metrics.get('precision', 0) +
        0.2 * metrics.get('recall', 0) +
        0.1 * metrics.get('val_accuracy', 0)
    ),
    
    # Tolerância (quanto de queda aceitar antes de parar)
    tolerance=0.02,  # 2%
    
    # Pesos para score composto
    weights={
        'val_accuracy': 0.2,
        'At': 0.2,
        'Rt': 0.1,
        'roi': 0.1,
        'precision': 0.4  # ← Peso alto em precision!
    }
)

print("✅ Objetivo criado:")
print(f"   Nome: {objetivo_precision.name}")
print(f"   Descrição: {objetivo_precision.description}")
print(f"   Métrica primária: {objetivo_precision.primary_metric}")
print(f"   Fórmula: 70% precision + 20% recall + 10% accuracy")
print()

# ============================================================================
# 4. TREINAR COM OBJETIVO CUSTOMIZADO
# ============================================================================

print("🚀 Treinando com objetivo customizado...")
print("-"*60)

detector = MultiObjectiveDetector(
    custom_objective=objetivo_precision,  # ← Passa seu objetivo
    patience=5,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=128,
    callbacks=[detector],
    verbose=0
)

print()
print("="*60)
print("✅ TREINAMENTO COMPLETO!")
print("="*60)
print()

# ============================================================================
# 5. AVALIAR RESULTADOS
# ============================================================================

# Avaliar no test set
results = model.evaluate(X_test, y_test, verbose=0)
test_loss, test_acc, test_precision, test_recall = results

print("📊 RESULTADOS FINAIS:")
print(f"   Melhor Epoch: {detector.best_epoch + 1}")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Test Precision: {test_precision*100:.2f}%")
print(f"   Test Recall: {test_recall*100:.2f}%")
print()

# Comparar com objetivo 'accuracy' tradicional
print("💡 COMPARAÇÃO:")
print(f"   Seu objetivo customizado priorizou PRECISION")
print(f"   Precision: {test_precision*100:.2f}%")
print(f"   Isso significa: menos falsos positivos!")
print()

# Mostrar relatório completo
report = detector.get_multi_objective_report()
print("📋 ANÁLISE DO FRAMEWORK:")
print(f"   Estado final: {report['final_state']}")
print(f"   Motivo da parada: {report['stop_reason']}")
print(f"   Score final: {report['best_score']:.3f}")
print()

print("="*60)
print("🎉 Sucesso! Você criou um objetivo customizado!")
print()
print("💡 DICA: Use objetivos customizados quando:")
print("   - Seu caso de uso é específico")
print("   - Falsos positivos/negativos têm custos diferentes")
print("   - Quer otimizar para F1, AUC, ou outra métrica")
print("="*60)
