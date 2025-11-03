# ❓ Dúvidas Frequentes

---

## 🤔 Conceitos Básicos

### O que é "early stopping"?

**Resposta simples:**
É parar o treinamento antes do número máximo de epochs, quando o modelo não está mais melhorando.

**Exemplo:**
```python
# Você pede 50 epochs
model.fit(..., epochs=50)

# Mas o modelo para de melhorar no epoch 12
# Early stopping: para no epoch 12, não desperdiça 38 epochs
```

### O que é "multi-objetivo"?

**Resposta simples:**
Ao invés de só otimizar accuracy (tradicional), você escolhe O QUE otimizar:
- Custo (gastar menos)
- Robustez (funcionar em mais situações)
- Balanceado (meio termo)

**Analogia:**
- Tradicional = correr uma maratona o mais RÁPIDO possível
- Multi-objetivo = você escolhe: mais rápido, ou menos cansaço, ou meio termo

---

## 🎯 Sobre os Objetivos

### Qual objetivo eu devo escolher?

**Depende do seu caso:**

| Seu Caso | Objetivo Recomendado |
|----------|---------------------|
| Competição Kaggle | `accuracy` |
| MVP / Startup | `cost` |
| Produção / Deploy real | `balanced` |
| IA Médica / Crítica | `robustness` |
| Não sei / Primeira vez | `balanced` |

### Posso criar meu próprio objetivo?

**Sim!** Veja o exemplo:
```bash
python exemplos/03_objetivo_customizado.py
```

Ou leia: [docs/objetivos_customizados.md](docs/objetivos_customizados.md)

### Posso testar todos e escolher depois?

**Sim! É até recomendado:**
```bash
python exemplos/02_comparar_objetivos.py
```

Isso treina com os 4 objetivos e mostra lado a lado.

---

## 💻 Técnicas

### Funciona com PyTorch?

**Ainda não.** Apenas TensorFlow/Keras por enquanto.

PyTorch está no roadmap para versão 2.0.

### Funciona com qualquer modelo?

**Sim**, desde que seja TensorFlow/Keras.

Testado em:
- ✅ Sequential models
- ✅ Functional API
- ✅ Model subclassing
- ✅ CNNs, RNNs, Transformers

### Preciso mudar meu código?

**Não!** Só adicionar 2 linhas:
```python
from neuralbifurcation import MultiObjectiveDetector  # linha 1

detector = MultiObjectiveDetector(objective='balanced')  # linha 2

model.fit(..., callbacks=[detector])  # adicionar callback
```

Resto do código fica igual.

### Funciona com transfer learning?

**Sim!** Inclusive é MUITO útil.

Exemplo:
```python
# Transfer learning com VGG16
base = VGG16(weights='imagenet', include_top=False)
model = Sequential([base, Dense(10, activation='softmax')])

# Usar objetivo 'robustness' ou 'discovery'
detector = MultiObjectiveDetector(objective='robustness')

model.fit(..., callbacks=[detector])
```

---

## 📊 Sobre Métricas

### O que é "At" (Autonomia)?

**Resposta simples:**
```
At = val_accuracy / train_accuracy
```

**O que significa:**
- At ≈ 1.0: Modelo balanceado (ideal)
- At < 0.8: Overfitting (decorou treino)
- At > 1.2: Underfitting (pode melhorar)

**Por que importa:**
Modelos com At próximo de 1.0 geralmente generalizam melhor.

### O que é "Rt" (Robustez)?

**Resposta simples:**
Medida de quão estável o modelo é.

- Rt alto: Modelo estável, confiável
- Rt baixo: Modelo oscilando, instável

**Você NÃO precisa entender isso pra usar o framework!**  
É calculado automaticamente.

### O que é "ROI"?

**Resposta simples:**
```
ROI = (melhoria de accuracy) / (custo gasto)
```

**Exemplo:**
- Treinou 5 epochs, gastou $15
- Accuracy subiu de 80% para 85% (+5 pontos)
- ROI = 5 / 15 = $0.33 por ponto

ROI alto = está valendo a pena continuar  
ROI baixo = melhor parar

### O que são os "Estados"?

**Resposta simples:**
O framework classifica cada epoch em um estado:

| Estado | O Que Significa |
|--------|----------------|
| inicializacao | Primeiros 4 epochs (calibrando) |
| aprendizado_saudavel | Tudo OK, progredindo bem |
| aprendizado_rapido | Progredindo MUITO (ROI alto) |
| plateau | Parou de melhorar (estagnado) |
| overfitting_inicial | Começando a decorar (alerta) |
| overfitting_severo | Decorando muito (framework para) |
| instabilidade | Métricas oscilando (problema) |

**Você NÃO precisa fazer nada!**  
Framework usa isso pra decidir quando parar.

---

## 🐛 Problemas Comuns

### "Framework parou muito cedo!"

**Possíveis causas:**

1. **Val set muito pequeno** (< 100 exemplos)
   - Solução: Aumentar val set para pelo menos 500 exemplos
   
2. **Métricas oscilando** (instabilidade)
   - Solução: Diminuir learning rate
   - Ou aumentar batch size
   
3. **Patience muito baixo**
   - Solução: Aumentar patience:
```python
   detector = MultiObjectiveDetector(
       objective='balanced',
       patience=15  # ao invés de 8 (padrão)
   )
```

### "Framework nunca para!"

**Possíveis causas:**

1. **Modelo muito simples** (sempre melhora)
   - Isso é BOM! Deixe treinar
   
2. **Learning rate muito baixo** (melhora muito devagar)
   - Solução: Aumentar LR

3. **Objetivo 'accuracy' + modelo bom**
   - Esperado! Accuracy sempre tenta mais epochs

### "Resultados piores que tradicional"

**Possíveis causas:**

1. **Objetivo errado pro seu caso**
   - Solução: Teste outros objetivos:
```bash
   python exemplos/02_comparar_objetivos.py
```

2. **Val set muito diferente do test set**
   - Framework otimiza pro val
   - Se val não representa test, problema
   - Solução: Melhorar split dos dados

3. **Dataset muito pequeno** (< 1000 exemplos)
   - Framework precisa de dados pra aprender
   - Solução: Data augmentation ou mais dados

### Meu modelo usa `model.fit_generator()` (deprecated)

**Solução:**
Use `model.fit()` com generators. Funciona igual:
```python
# Antes (deprecated):
model.fit_generator(train_gen, ...)

# Agora:
model.fit(train_gen, ..., callbacks=[detector])
```

---

## 💰 Sobre Custos

### Como funciona o tracking de custo?

**Você informa o custo por epoch:**
```python
detector = MultiObjectiveDetector(
    objective='cost',
    cost_per_epoch=3.0  # R$3 por epoch na sua GPU
)
```

Framework multiplica:
```
Custo Total = epochs_treinados × cost_per_epoch
```

### Como sei quanto minha GPU custa por epoch?

**Opção 1: Google Colab**
- Grátis: $0
- Colab Pro: ~$0.50/epoch (estimativa)

**Opção 2: Cloud (AWS, GCP, Azure)**
- GPU T4: ~R$1.50/epoch
- GPU A100: ~R$5-10/epoch
- Calcule: (custo_por_hora × tempo_por_epoch)

**Opção 3: GPU Local**
- Considere custo de energia + depreciação
- Ou simplesmente use $0 (recurso já seu)

### Preciso informar custo?

**Não!** É opcional.

Se não informar, framework ainda funciona, só não mostra economia.

---

## 🔬 Sobre a Teoria

### Preciso entender a matemática?

**NÃO!** 

Framework é plug-and-play. Use sem entender a teoria.

Mas se tiver curiosidade: [paper/teorema_ajustado.pdf](paper/teorema_ajustado.pdf)

### O que é "Teorema da Lei das Leis"?

**Resumo ultra-simples:**

É uma teoria matemática sobre como sistemas mudam de comportamento.

Aplicada ao ML: explica quando seu modelo muda de "aprendendo" pra "overfitting".

Framework detecta essas mudanças automaticamente.

### Isso foi validado cientificamente?

**Sim:**
- ✅ Testado em 2 datasets reais
- ✅ Teoria matemática sólida
- ✅ Resultados replicáveis
- 📄 Paper em preparação

---

## 🤝 Beta Testing

### O que você espera que eu teste?

**Mínimo (15 min):**
1. Rodar exemplo básico
2. Me dizer se funcionou
3. Bugs? Erros? Problemas?

**Ideal (30 min):**
1. Testar no SEU modelo
2. Comparar com treino tradicional
3. Melhorou? Piorou? Igual?
4. Feedback sobre UX

**Avançado (1h+):**
1. Testar múltiplos objetivos
2. Criar objetivo customizado
3. Caso de uso detalhado
4. Sugestões de features

### Como reporto bugs?

**Me manda mensagem com:**
1. Print do erro
2. Código que você rodou (ou exemplo que falhou)
3. Seu ambiente (OS, Python version, TF version)

📧 Email: [marcelo.galdino@outlook.com.br]  

### Quanto tempo você precisa do meu feedback?

**Sem pressa!**

Teste quando tiver tempo. Qualquer feedback é valioso, mesmo que demore 1-2 semanas.

### Posso compartilhar isso?

**Pode!** 

Só peço que mencione que é versão BETA e pode ter bugs.

Se quiser, mande o link do repo pra outras pessoas testarem também.

---

## 📬 Outras Dúvidas?

**Sua dúvida não está aqui?**

📧 Me manda email: [marcelo.galdino@outlook.com.br]  

Respondo em até 24h!

---

**Atualizado em:** [data]
