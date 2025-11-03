# 🎯 Neural Bifurcation Framework (BETA)

> **Early Stopping Multi-Objetivo para Deep Learning**  
> Pare o treinamento no momento CERTO para o SEU objetivo.

---

## 🚀 Início Rápido (< 5 minutos)

### Opção 1: Google Colab (Zero Instalação) ⭐ RECOMENDADO
👉 **[Abrir no Colab](link-aqui)** ← Clique e rode!

### Opção 2: Instalação Local
```bash
# Clone este repositório
git clone https://github.com/[seu-usuario]/neural-bifurcation-beta.git
cd neural-bifurcation-beta

# Instale as dependências
pip install -r requirements.txt

# Rode o exemplo básico
python exemplos/01_ola_mundo.py
```

**Pronto!** O framework vai treinar no MNIST e mostrar a diferença.

---

## 🤔 O Que É Isso?

Early stopping tradicional **só maximiza accuracy de validação**.

Mas no mundo real, você frequentemente quer:
- 💰 **Minimizar custo** (orçamento de GPU apertado)
- 🛡️ **Maximizar robustez** (deploy em produção)
- ⚖️ **Balancear trade-offs** (uso geral)
- 🔬 **Generalização** (transfer learning)

**Este framework deixa você ESCOLHER o objetivo.**

---

## 🎯 Como Funciona
```python
from neuralbifurcation import MultiObjectiveDetector

# Escolha seu objetivo
detector = MultiObjectiveDetector(
    objective='balanced'  # ou 'accuracy', 'cost', 'robustness'
)

# Treine normalmente
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[detector]  # ← Adicione esta linha
)

# O framework automaticamente:
# ✅ Detecta quando parar
# ✅ Salva o melhor modelo
# ✅ Previne overfitting
# ✅ Otimiza para SEU objetivo
```

É isso! Apenas **1 linha de código**.

---

## 📊 Resultados Reais (Detecção de Pneumonia em Raio-X)

Validei este framework em um dataset médico real (624 raios-X).

| Objetivo | Accuracy | Mortes (Falsos Neg) | Custo | Robustez |
|----------|----------|---------------------|-------|----------|
| **Tradicional** | 90.5% | 21 ☠️ | $39 | Frágil |
| **Cost** | 90.4% | 21 ☠️ | $24 | Médio |
| **Balanced** | 84.6% | **5 ☠️** | $51 | **2× Mais Forte** |

**Objetivo "balanced" salvou 16 vidas (76% de redução) e foi 2× mais robusto a equipamento com ruído.**

Veja resultados completos: [resultados/xray_resultados.png](resultados/xray_resultados.png)

---

## 🎨 Objetivos Disponíveis

### 1. `accuracy` (Tradicional)
**Quando usar:** Competições Kaggle, benchmarks, prova de conceito  
**Otimiza:** Performance máxima de validação  
**Trade-off:** Pode dar overfit, custo maior

**Exemplo:**
```python
detector = MultiObjectiveDetector(objective='accuracy')
```

---

### 2. `robustness` (Produção)
**Quando usar:** Deploy em produção, IA médica, sistemas críticos  
**Otimiza:** Generalização, estabilidade, menos falsos negativos  
**Trade-off:** -5% accuracy por +2× robustez

**Exemplo:**
```python
detector = MultiObjectiveDetector(objective='robustness')
```

**Caso de uso real:**
- Detecção médica (minimizar mortes)
- Sistemas financeiros (evitar fraudes não detectadas)
- Carros autônomos (segurança crítica)

---

### 3. `cost` (Econômico)
**Quando usar:** Orçamento limitado, iteração rápida, MVPs  
**Otimiza:** Melhor performance por real gasto  
**Trade-off:** Para mais cedo, modelo "bom o suficiente"

**Exemplo:**
```python
detector = MultiObjectiveDetector(
    objective='cost',
    cost_per_epoch=3.0  # Custo da sua GPU (R$/epoch)
)
```

**Resultado típico:**
- 90% da max accuracy
- 40-60% do custo
- ROI 2-3× melhor

---

### 4. `balanced` (Equilibrado)
**Quando usar:** Deploy real, não sabe prioridades, uso geral  
**Otimiza:** Trade-off entre todas métricas  
**Trade-off:** "Canivete suíço" - bom em tudo, ótimo em nada

**Exemplo:**
```python
detector = MultiObjectiveDetector(objective='balanced')
```

**Melhor para:**
- Primeira vez usando o framework
- Aplicações reais complexas
- Quando múltiplos critérios importam

---

## 🔥 Por Que Isso Importa?

### Problema Atual
```python
# Early stopping tradicional
early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

# Problema: SEMPRE otimiza accuracy
# Mas e se você quiser:
# - Economizar GPU?
# - Ter modelo mais robusto?
# - Balancear múltiplos objetivos?
```

### Com Neural Bifurcation
```python
# Você ESCOLHE o objetivo
detector = MultiObjectiveDetector(objective='cost')  # ou outro

# Framework se adapta automaticamente
# Para no momento CERTO para SEU objetivo
```

---

## 📚 Exemplos Inclusos

### Exemplo 1: Olá Mundo (5 min)
```bash
python exemplos/01_ola_mundo.py
```
Treina MNIST com objetivo "balanced". Mostra o básico.

### Exemplo 2: Comparar Objetivos (10 min)
```bash
python exemplos/02_comparar_objetivos.py
```
Treina o MESMO modelo com os 4 objetivos e compara lado a lado.

### Exemplo 3: Objetivo Customizado (15 min)
```bash
python exemplos/03_objetivo_customizado.py
```
Mostra como criar seu próprio objetivo.

---

## 🎓 Como Foi Validado?

### Dataset 1: CIFAR-10 (Imagens)
- 50k treino, 10k teste
- 4 objetivos testados
- Economia: 40-70% de custo

### Dataset 2: Chest X-Ray (Médico)
- 5.2k treino, 624 teste
- Pneumonia detection (crítico)
- Resultado: 76% menos mortes com "balanced"

### Teoria: Teorema da Lei das Leis
Framework baseado em teoria matemática sobre transição de regimes 
em sistemas dinâmicos não-autônomos.

📄 [Leia o paper](paper/teorema_ajustado.pdf)

---

## ⚙️ Requisitos
```bash
Python >= 3.8
TensorFlow >= 2.10
NumPy >= 1.19
scikit-learn >= 0.24
```

Instale tudo:
```bash
pip install -r requirements.txt
```

---

## 🐛 Problemas? Dúvidas?

1. **Leia primeiro:** [DUVIDAS_FREQUENTES.md](DUVIDAS_FREQUENTES.md)
2. **Teste no Colab:** Link na pasta raiz (ambiente controlado)
3. **Me avise:** [marcelo.galdino@outlook.com.br]

---

## 🤝 Como Ajudar (Beta Tester)

**O que preciso de você:**

1. **Teste no seu modelo** (15-30 min)
2. **Me conte:**
   - Funcionou?
   - Melhorou algo?
   - Bugs?
   - Sugestões?

3. **Compartilhe resultados** (opcional)
   - Print do output
   - Métricas antes/depois
   - Caso de uso

**Em troca você ganha:**
- ✅ Acesso vitalício grátis (quando lançar versão paga)
- ✅ Crédito como early adopter no GitHub
- ✅ Prioridade em features futuras
- ✅ Uma ferramenta útil de graça!

---

## 📬 Contato

**Marcelo Galdino**  
📧 Email: [marcelo.galdino@outlook.com.br]  
💬 WhatsApp: [+55 11 942338841]  
🐙 GitHub: [(https://github.comarcelogalds-hub)]

---

## 📝 Licença

MIT License - use à vontade, comercialmente ou não.

---

**Feito por Marcelo Galdino**  
**Baseado no Teorema da Lei das Leis** 🌌
