[English](#english) • [Русский](#russian)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

<a id="english"></a>

## 🔬 Applied machine learning research suite

### 📌 Overview

This repository documents a multi-track experimental program in **applied machine learning**: from interpretable pattern mining and manifold learning to classical and ensemble classification, class-imbalance handling, metaheuristic optimization, unsupervised structure discovery, and recurrent deep models for prognostics. The work is organized as **seven integrated research modules**, each with reproducible Jupyter notebooks, explicit methodology, and quantitative reporting.

The motivation is twofold: (1) to compare algorithms under **controlled, comparable protocols** (same splits, metrics, and random seeds where applicable), and (2) to connect method choices to **downstream impact**—whether that is rule quality, visualization fidelity, credit-decision metrics, optimization success rates, clustering validity, or remaining-useful-life modeling for engineered systems.

---

### 🧭 Research modules

| Module | Topic | Methods & tooling | Primary datasets |
|--------|--------|---------------------|------------------|
| **01–02** | Associative rule mining | Apriori, Efficient Apriori, FPGrowth (`apriori-python`, `efficient-apriori`, `fpgrowth-py`); custom rule metrics | Discretized **Australian Credit Approval**; synthetic retail-style transactions |
| **03–04** | Nonlinear dimensionality reduction | t-SNE, **UMAP**, TriMAP, **PaCMAP**; scaling (Standard / Robust / MinMax) | **Australian Credit Approval**; **Mammoth** 3D point cloud (`mammoth.csv`) |
| **05–06** | Supervised classification | **SVM** (multi-kernel + GridSearchCV), **kNN**, **Random Forest**; confusion matrices & manifold plots | **Australian Credit Approval** |
| **07–08** | Imbalanced learning | **SMOTE**, **Borderline-SMOTE**, **Borderline-SMOTE2** (`imbalanced-learn`); SVM / kNN / RF | **Australian Credit Approval** |
| **09–10** | Global optimization | **Styblinski–Tang** benchmark; L-BFGS-B, Nelder–Mead, GA, PSO, differential evolution, bees, continuous ACO | Synthetic 2D objective (closed-form global minimum) |
| **11–12** | Clustering & validation | **K-Means**, hierarchical (Ward), **DBSCAN**, fuzzy c-means; **Silhouette**, **ARI** | **Australian Credit Approval**; **Mammoth** |
| **13–16** | Sequence modeling & prognostics | **LSTM**, vanilla **RNN**, **GRU** (**TensorFlow / Keras**); windowed sensors; early stopping | **NASA CMAPSS** turbofan (unit **id = 20**) |

---

### 🏆 Key results (reported from experiments)

#### Module 01–02 — Association rules
- On the benchmark transaction basket (`min_support=0.3`), **3 rules** were obtained at **60% confidence** and **2 rules** at **80%** (manual and library pipelines agree).
- All three implementations return **consistent rule sets** on the small benchmark; wall-clock times are **sub-millisecond**, so **FPGrowth** and **Efficient Apriori** mainly differentiate at larger scales (FPGrowth avoids candidate enumeration).

#### Module 03–04 — Dimensionality reduction & Mammoth
- On Australian Credit, **UMAP** delivered the best trade-off of **speed**, **reproducibility** (fixed `random_state`), and **global layout** versus t-SNE; **StandardScaler** / **RobustScaler** gave clearer class separation than MinMax (sensitive to heavy-tailed features such as raw balance fields).
- For **`mammoth.csv`**, nonlinear embeddings (including **PaCMAP** / **UMAP** pipelines in the notebooks) produce **stable 2D structure** for the ~690-point cloud suitable for qualitative assessment and downstream clustering visuals.

#### Module 05–06 — Classification (test set, tuned models)

| Model | Accuracy | Precision | Recall | F1 |
|--------|----------|------------|--------|-----|
| **SVM** (GridSearchCV-selected) | 0.81 | 0.85† | 0.81† | 0.81† |
| **kNN** (best explored config, e.g. k≈9–15) | **0.848** | **0.845** | 0.803 | **0.824** |
| **Random Forest** (strong default grid point) | **0.833** | 0.771 | **0.885** | **0.824** |

†Weighted averages from the printed `classification_report` for the tuned SVM.

#### Module 07–08 — Class balancing
- **Borderline-SMOTE (`borderline-1`)** achieved the **highest test F1 for Random Forest (0.8296)** in the consolidated leaderboard and competitive metrics for kNN, matching the **prior hypothesis** that **borderline-focused oversampling** is most informative near the decision boundary.
- **SVM** remained sensitive to synthetic point placement; **Borderline-SMOTE2** improved SVM test F1 to **0.8000** in the summary table versus some SMOTE-only configurations.

#### Module 09–10 — Styblinski–Tang optimization (100 runs each)

Global minimum reference: **f\* ≈ −78.33234** at **x\* ≈ (−2.903534, −2.903534)**.

| Algorithm | Mean f | Variance | % global min | Mean time (s) |
|-----------|--------|----------|--------------|---------------|
| Bees Algorithm | −78.33233 | 0.000000 | **100.0** | 0.523 |
| ACOR (continuous ACO) | −78.33233 | 0.000000 | **100.0** | 0.474 |
| Differential Evolution | −78.33233 | 0.000000 | **100.0** | **0.111** |
| Genetic Algorithm | −78.33072 | 0.000010 | 96.0 | 2.659 |
| PSO | −77.62550 | 9.588610 | 95.0 | 0.028 |
| L-BFGS-B | −68.57800 | 91.626741 | 43.0 | 0.004 |
| Nelder–Mead | −65.18518 | 105.999564 | 30.0 | 0.004 |

**Takeaway:** Evolutionary / population methods reliably approach the **global basin**; **differential evolution** offers an excellent **accuracy–cost** ratio. Local classical optimizers often terminate in **non-global** minima from random starts.

#### Module 11–12 — Clustering comparison (representative best configs)

| Method | Silhouette (≈) | ARI (≈) | Notes |
|--------|----------------|---------|--------|
| Hierarchical (Ward) | **0.757** | 0.001 | High internal cohesion; weak agreement with credit labels |
| **K-Means** (selected k) | 0.165 | **0.404** | Best **label alignment (ARI)** among centroid models |
| **DBSCAN** (tuned) | **0.688** | **−0.004** | High silhouette but **poor external match** to binary labels |
| Fuzzy c-means (module narrative) | 0.147 | **0.444** | Strongest **ARI** in the fuzzy clustering track |

**Profiling:** clusters on the credit dataset reflect **geometry in mixed feature space** more than raw approval labels; Mammoth segments highlight **spatial density** versus noise points.

#### Module 13–16 — RNN family vs NASA CMAPSS (engine **id = 20**)
- **Trainable parameters:** **LSTM 77 851** vs **SimpleRNN 19 501** vs **GRU 58 851** (two recurrent layers + dense head, as built in the notebook).
- **CMAPSS** stream is isolated **per engine**; unit **20** is used end-to-end for **train/test cycle plots**, sensor trimming (constant sensors removed → **15 informative channels**), and **binary failure horizon** classification with recurrent baselines.

---

### 🗂 Datasets

| Dataset | Role | Notes |
|---------|------|--------|
| **Australian Credit Approval** (Statlog) | Classification, DR, balancing, clustering | Encoded mixed features; standard train/test splits in notebooks |
| **Mammoth point cloud** (`mammoth.csv`) | Manifold learning & density clustering | 3D coordinates, ~690 rows in the experiment track |
| **NASA CMAPSS** | Turbofan degradation / RUL-oriented modeling | Multivariate sensor time series; **engine id = 20** subset |

> Raw `.dat` / proprietary dumps are **not** committed (see `.gitignore`). Place public copies locally where notebook paths expect them.

---

### 🛠 Tech stack

**Core:** Python 3.10, Jupyter, NumPy, pandas, SciPy, Matplotlib, Seaborn  

**Classical ML:** scikit-learn, imbalanced-learn, graphviz (tree export)  

**Association rules:** apriori-python, efficient-apriori, fpgrowth-py, mlxtend, pyarmviz, NetworkX  

**Manifold learning:** umap-learn, TriMAP, PaCMAP  

**Optimization:** scipy.optimize, geneticalgorithm, pyswarm, bees-algorithm, custom ACOR-style ACO  

**Fuzzy clustering:** scikit-fuzzy  

**Deep learning:** TensorFlow, Keras  

---

### 📁 Repository structure

```
ist_practices/
├── README.md                 # This file
├── requirements.txt          # Unified dependency list
├── practice_01_02/           # Association rules
├── practice_03_04/           # Dimensionality reduction + Mammoth
├── practice_05_06/           # SVM, kNN, Random Forest
├── practice_07_08/           # SMOTE family & classifiers
├── practice_09_10/           # Evolutionary & classical optimization
├── practice_11_12/         # Clustering & fuzzy methods
└── practice_13_16/           # LSTM / RNN / GRU on CMAPSS
```

Each `practice_*` folder contains its own `practice_*.ipynb` and a **local `requirements.txt`** for minimal installs.

---

### ▶️ How to run

```bash
cd ist_practices
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
jupyter notebook
```

Open the notebook for the module you need (e.g. `practice_09_10/practice_09_10.ipynb`) and execute cells top to bottom. Download datasets separately if paths in the notebook point to local files.

---

<a id="russian"></a>

## 🇷🇺 Прикладные исследования по машинному обучению

### 📌 Обзор

Репозиторий объединяет **семь модулей прикладных исследований** в области машинного обучения: поиск ассоциативных правил, нелинейное снижение размерности, классификация (SVM, kNN, случайный лес), балансировка классов, эволюционная и классическая оптимизация, кластеризация с валидацией и рекуррентные нейросети для прогнозирования отказов по данным датчиков. Для каждого модуля приведены методики, метрики и **воспроизводимые** Jupyter-ноутбуки.

Цель — сопоставлять алгоритмы в **единых условиях** (одинаковые разбиения, метрики, фиксация случайности) и связывать выбор метода с **измеримым эффектом** на качество модели или интерпретацию данных.

---

### 🧭 Таблица модулей

| Модуль | Тема | Методы | Данные |
|--------|------|--------|--------|
| **01–02** | Ассоциативные правила | Apriori, Efficient Apriori, FPGrowth | Australian Credit (дискретизация), тестовые транзакции |
| **03–04** | Снижение размерности | t-SNE, **UMAP**, TriMAP, **PaCMAP** | Australian Credit; облако **Mammoth** |
| **05–06** | Классификация | **SVM**, **kNN**, **Random Forest**, Grid Search | Australian Credit |
| **07–08** | Дисбаланс классов | **SMOTE**, **Borderline-SMOTE**, **Borderline-SMOTE2** | Australian Credit |
| **09–10** | Оптимизация | Функция **Стыбинского–Танга**; L-BFGS-B, Nelder–Mead, ГА, PSO, ДЭ, пчёлы, ACO | Синтетическая 2D цель |
| **11–12** | Кластеризация | K-Means, иерархия, **DBSCAN**, нечёткие c-средние; **Silhouette**, **ARI** | Australian Credit; Mammoth |
| **13–16** | Прогнозирование по рядам | **LSTM**, **RNN**, **GRU** (TensorFlow/Keras) | **NASA CMAPSS**, двигатель **id = 20** |

---

### 🏆 Ключевые результаты

**Модули 01–02:** при `min_support=0.3` получено **3 правила** при доверии **60%** и **2 правила** при **80%**; реализации **согласованы**; на малых данных все методы быстрые, **FPGrowth** масштабируется за счёт FP-дерева.

**Модули 03–04:** **UMAP** показал лучший баланс **скорости** и **глобальной структуры** относительно t-SNE; для **mammoth.csv** нелинейные проекции (**UMAP** / **PaCMAP**) дают устойчивую 2D-визуализацию облака точек.

**Модули 05–06** (тест, настроенные модели):

| Модель | Accuracy | Precision | Recall | F1 |
|--------|----------|------------|--------|-----|
| **SVM** (после Grid Search) | 0.81 | 0.85† | 0.81† | 0.81† |
| **kNN** | **0.848** | **0.845** | 0.803 | **0.824** |
| **Random Forest** | **0.833** | 0.771 | **0.885** | **0.824** |

†Взвешенные средние из `classification_report` для лучшей SVM.

**Модули 07–08:** **Borderline-SMOTE (borderline-1)** дал **максимальный F1 на тесте для Random Forest (0.8296)** в сводной таблице; это согласуется с гипотезой о пользе **фокусировки на граничных** объектах миноритарного класса.

**Модули 09–10:** глобальный минимум **f\* ≈ −78.33234**. **Дифференциальная эволюция**, **пчелиный** и **ACOR** стабильно попадают в глобальный минимум (**100%** прогонов в таблице); **ДЭ** — самый быстрый из этой группы (**~0.11 с** среднее время). **L-BFGS-B** и **Nelder–Mead** часто останавливаются в **локальных** минимумах.

**Модули 11–12:** **иерархическая** кластеризация — высокий **Silhouette (~0.76)**, низкий **ARI**; **K-Means** — **ARI ≈ 0.404**; **DBSCAN** — **Silhouette ≈ 0.688**, **ARI ≈ −0.004**; **нечёткие c-средние** — **ARI до ~0.444**. Профили кластеров отражают **геометрию признаков**, а не только метку одобрения.

**Модули 13–16:** число обучаемых параметров: **LSTM 77 851**, **RNN 19 501**, **GRU 58 851**; для **CMAPSS** используется траектория **двигателя с id = 20** (обучение/тест, отбор сенсоров, бинарная постановка).

---

### 🗂 Наборы данных

| Набор | Назначение |
|--------|------------|
| **Australian Credit Approval** | Классификация, визуализация, балансировка, кластеризация |
| **Mammoth** (`mammoth.csv`) | Многообразия и плотностная кластеризация |
| **NASA CMAPSS** | Многомерные ряды датчиков турбовентилятора |

Файлы **`*.dat`** в репозиторий не входят (см. `.gitignore`).

---

### 🛠 Технологии

Python 3.10, Jupyter, NumPy, pandas, SciPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, umap-learn, TriMAP, PaCMAP, TensorFlow/Keras, библиотеки ассоциативных правил и эволюционной оптимизации (см. `requirements.txt`).

---

### 📁 Структура репозитория

См. английский раздел **Repository structure** — каталоги `practice_01_02` … `practice_13_16` с ноутбуками и локальными `requirements.txt`.

---

### ▶️ Как запустить

```bash
cd ist_practices
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Откройте нужный `practice_*_*.ipynb` и выполните ячейки по порядку. Датасеты загрузите локально при необходимости.

---

**Remote:** [https://github.com/Saitama4722/ist_practices](https://github.com/Saitama4722/ist_practices)
