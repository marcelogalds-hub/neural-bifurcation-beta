# 🚀 Início Rápido - Neural Bifurcation

**Objetivo: Você testando em 5 minutos**

---

## Opção 1: Google Colab (MAIS FÁCIL)

1. Abra este link: [Link do Colab aqui]
2. Clique em "Runtime" → "Run all"
3. Aguarde 3-4 minutos
4. Veja os resultados!

**Pronto!** Zero instalação, zero configuração.

---

## Opção 2: No Seu Computador

### Passo 1: Clonar o Repositório
```bash
git clone https://github.com/[seu-usuario]/neural-bifurcation-beta.git
cd neural-bifurcation-beta
```

### Passo 2: Instalar Dependências
```bash
pip install tensorflow numpy scikit-learn matplotlib
```

### Passo 3: Rodar Exemplo
```bash
python exemplos/01_ola_mundo.py
```

### Passo 4: Ver Resultado
O script vai:
1. Baixar MNIST automaticamente
2. Treinar com objetivo "balanced"
3. Comparar com treino tradicional
4. Mostrar a diferença

**Output esperado:**
```
🎯 Treinando com objetivo: balanced
Epoch 1/20: val_acc=0.92 | At=0.95 | Estado: aprendizado_saudavel
Epoch 2/20: val_acc=0.94 | At=0.97 | Estado: aprendizado_saudavel
...
Epoch 8/20: val_acc=0.98 | At=1.02 | Estado: plateau

🛑 PAROU no Epoch 8 (objetivo: balanced)

📊 RESULTADO:
   Accuracy: 98.2%
   Custo: $24 (vs $60 tradicional)
   Economia: 60%!
```

---

## Opção 3: Integrar no Seu Modelo

Se você já tem um modelo TensorFlow/Keras treinando:

### Antes:
```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50
)
```

### Depois:
```python
from neuralbifurcation import MultiObjectiveDetector

detector = MultiObjectiveDetector(objective='balanced')

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[detector]  # ← Adicione esta linha
)
```

**É só isso!**

---

## O Que Testar?

### Teste Básico (5 min)
```bash
python exemplos/01_ola_mundo.py
```
Só pra ver funcionando.

### Teste Comparativo (10 min)
```bash
python exemplos/02_comparar_objetivos.py
```
Veja a diferença entre os 4 objetivos.

### Teste no Seu Modelo (15-30 min)
1. Copie o código do seu modelo
2. Adicione 2 linhas (import + callback)
3. Rode e compare com o treino normal

---

## Deu Problema?

### Erro: "ModuleNotFoundError: No module named 'neuralbifurcation'"

**Solução:**
```bash
# Adicione a pasta ao Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou rode a partir da pasta raiz:
cd neural-bifurcation-beta
python exemplos/01_ola_mundo.py
```

### Erro: "TensorFlow não instalado"

**Solução:**
```bash
pip install tensorflow
```

### Outros Problemas

📧 Me manda mensagem: [seu-email]  
Ou tenta no Colab (ambiente garantido)

---

## Próximos Passos

1. ✅ Rode o exemplo básico
2. ✅ Teste no seu modelo
3. ✅ Me mande feedback!
4. 📧 Dúvidas? [DUVIDAS_FREQUENTES.md](DUVIDAS_FREQUENTES.md)

**Boa sorte!** 🚀
