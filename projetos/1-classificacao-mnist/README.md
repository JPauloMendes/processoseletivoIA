# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** João Paulo Mendes de Souza

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura que eu implementei no train_model.py é uma Rede Neural Convolucional (CNN) sequencial que agora conta com 3 blocos convolucionais. O primeiro bloco usa uma camada Conv2D com 32 filtros, o segundo usa 64 filtros, e o terceiro utiliza 128 filtros, extraindo padrões cada vez mais complexos das imagens. Cada convolução é seguida de um BatchNormalization para estabilizar o treinamento e um MaxPooling2D para reduzir o tamanho das matrizes e focar na informação mais relevante. Antes da camada densa de decisão final, adicionei um Dropout (0.5) para desligar alguns neurônios aleatoriamente, obrigando a rede a aprender de verdade em vez de apenas "decorar" as imagens. Para otimizar o tempo, usei a estratégia de EarlyStopping, que monitora o erro na validação e interrompe o processo sozinho se o modelo parar de melhorar.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow / Keras (2.12): Foi a biblioteca principal que usei para montar as camadas da rede neural, treinar e depois converter o modelo.
NumPy (2.4.6): Utilizada na parte de inferência para ajustar o formato dos dados e extrair a resposta final usando o argmax.
OS (Python 3.11.15): Utilizada para mexer nas variáveis de ambiente e forçar o sistema a executar o modelo apenas na CPU.

### 3️⃣ Técnica de Otimização do Modelo

No arquivo optimize_model.py, utilizei a técnica de Quantização (Dynamic Range Quantization) nativa do TensorFlow Lite. Ela converte os "pesos" da rede neural de números pesados de ponto flutuante (float32) para números inteiros menores (int8). Isso reduz muito o tamanho do modelo para que ele possa rodar em dispositivos de borda com pouca memória, mantendo a precisão quase intacta.

### 4️⃣ Resultados Obtidos

Acurácia no conjunto de teste: 99.16% (0.9916).
Tamanho do model.h5 (Original): 1.2MB.
Tamanho do model.tflite (Otimizado): 108KB.

### 5️⃣ Comentários Adicionais (Opcional)

A adição do terceiro bloco convolucional foi uma mudança para cumprir os requisitos do projeto e aumentar a capacidade de aprendizado da rede. O resultado foi muito bom, alcançando uma acurácia final de 0.9916. Foi interessante o impacto físico dessa mudança uma vez que como o modelo ganhou mais parâmetros matemáticos, o tamanho do arquivo original subiu de 643KB para 1.2MB, o qual duplicou praticamente. Mesmo com esse aumento, a técnica de quantização continuou sendo super eficiente, comprimindo o modelo final para apenas 108KB, o que é ótimo para dispositivos menores sem muita capacidade de armazenamento.

### 6️⃣ Exemplo de Inferência

Abaixo está a saída do terminal quando rodei o meu arquivo run_inference.py com o modelo otimizado:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Fiquei muito satisfeito com esse teste. O modelo em .tflite acertou 100% dessas 5 primeiras amostras. Isso mostra que a técnica de quantização realmente funciona uma vez que reduzimos o peso do arquivo em cerca de 10 vezes, mesmo assim, foi mantido a inteligência da rede neural.
