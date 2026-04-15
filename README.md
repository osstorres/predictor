# Football Match Predictor

Sistema de predicción de partidos de fútbol que combina tres fuentes de probabilidad: odds de casas de apuestas (Bet365), mercados de predicción descentralizados (Polymarket) y un modelo de ML propio (XGBoost ensemble). Claude AI actúa como capa de interpretación para analizar divergencias y generar reportes en lenguaje natural.

## Arquitectura

```
football-data.co.uk  →  Data Layer (CSV histórico)
                              ↓
              Feature Engineering (ELO, xG, fatiga, H2H)
                              ↓
         ┌────────────────────┼────────────────────┐
    Bookmaker odds      Polymarket API         ML Ensemble
    (Bet365 norm.)   (crowd intelligence)   (XGBoost + RF + LR)
         └────────────────────┼────────────────────┘
                              ↓
                    Claude AI — análisis de divergencias
                    y reporte en lenguaje natural
                              ↓
                  Visualizaciones + Predicción final
```

## Requisitos

- Python 3.10+
- API key de Anthropic ([console.anthropic.com](https://console.anthropic.com))
- Polymarket **no requiere** API key (API pública)

## Instalación

```bash
git clone https://github.com/osstorres/predictor.git
cd predictor

pip install -r requirements.txt

cp .env.example .env
# Editar .env y agregar tu clave:
# ANTHROPIC_API_KEY=sk-ant-...
```

## Uso

### Pipeline completo

Descarga datos, entrena modelos, genera visualizaciones y ejecuta backtest.

```bash
python main.py
```

Usa 5 temporadas (2020-2025) de Premier League, La Liga y Bundesliga. Tarda ~5-10 minutos.

### Modo rápido

Solo 2 temporadas, sin walk-forward backtest. Útil para probar que todo funciona.

```bash
python main.py --quick
```

### Pipeline + reportes Claude

Ejecuta el pipeline completo y al final genera reportes en lenguaje natural para los últimos 5 partidos de la Premier League via Claude AI.

```bash
python main.py --quick --predict
```

### Predecir un partido específico

Analiza un partido puntual combinando Claude + Polymarket + odds manuales. No requiere entrenar el modelo completo.

```bash
python predict_match.py --home Arsenal --away Chelsea --league "Premier League"
python predict_match.py --home "Real Madrid" --away "Barcelona" --league "La Liga"
python predict_match.py --home "Bayern Munich" --away "Dortmund" --league "Bundesliga"
```

## Ligas disponibles

Los datos históricos vienen de [football-data.co.uk](https://www.football-data.co.uk) (gratuito, sin API key).

| Código | Liga | País |
|--------|------|------|
| `E0` | Premier League | Inglaterra |
| `SP1` | La Liga | España |
| `D1` | Bundesliga | Alemania |
| `I1` | Serie A | Italia |
| `F1` | Ligue 1 | Francia |

Para cambiar las ligas editá `main.py`:

```python
LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]  # todas
LEAGUES = ["I1"]                             # solo Serie A
```

## Datos de entrada

El sistema descarga automáticamente CSVs con estas columnas por partido:

| Campo | Descripción |
|-------|-------------|
| `FTHG / FTAG` | Goles final (local / visitante) |
| `HS / AS` | Tiros totales |
| `HST / AST` | Tiros al arco |
| `HC / AC` | Córners |
| `HF / AF` | Faltas |
| `HY / AY / HR / AR` | Amarillas y rojas |
| `B365H / B365D / B365A` | Cuotas Bet365 (local / empate / visitante) |

## Features que genera el modelo

A partir de los datos crudos el sistema calcula ~60 features por partido:

**Rolling stats (últimos 5 partidos)**
- Promedio de goles a favor/en contra, tiros, tiros al arco, córners, faltas
- Diferencial entre equipo local y visitante para cada stat

**ELO**
- Rating ELO de cada equipo antes del partido
- Diferencial y probabilidad esperada según ELO

**xG Proxy**
- Expected goals aproximado desde tiros y tiros al arco
- Overperformance vs xG (señal de regresión a la media)

**Fatiga**
- Días de descanso desde el último partido de cada equipo
- Flag de partido entre semana (martes/miércoles)
- Ventaja de descanso (diferencial local - visitante)

**Head-to-Head**
- % de victorias del local en los últimos 5 encuentros directos
- % de empates y promedio de goles en esos enfrentamientos

**Odds normalizadas**
- Probabilidades implícitas Bet365 sin margen (normalización)
- Spread favorito/underdog

## Outputs

### Archivos generados (PNG)

| Archivo | Contenido |
|---------|-----------|
| `model_comparison.png` | Accuracy y Log Loss comparado entre LR, RF y XGBoost |
| `confusion_matrix.png` | Matriz de confusión del ensemble en el set de test |
| `calibration_curve.png` | Calibración de probabilidades (qué tan bien mapean a resultados reales) |
| `feature_importance.png` | Top 15 features más importantes según XGBoost |
| `triple_radar.png` | Radar chart comparando las 3 fuentes de probabilidad para un partido |

### Métricas en consola

**Por modelo (TimeSeriesSplit CV, 5 folds):**
```
Logistic Regression    Acc: 0.5123 ± 0.0210   LogLoss: 1.0234
Random Forest          Acc: 0.5287 ± 0.0195   LogLoss: 0.9876
XGBoost                Acc: 0.5401 ± 0.0178   LogLoss: 0.9612
```

**Ensemble (test set 20%):**
```
              precision  recall  f1-score
Away Win       0.52      0.48     0.50
Draw           0.28      0.22     0.25
Home Win       0.57      0.65     0.61
```

**Walk-forward backtest (paso de 38 partidos = 1 jornada):**
```
Total predictions: 1240
Accuracy: 0.5318
Log Loss: 0.9734
```

### Reporte Claude (texto)

Cuando se usa `--predict` o `predict_match.py`, Claude genera:

1. **Análisis contextual** — scores de ataque, defensa, momentum (JSON 0-1)
2. **Reporte narrativo** — factores clave, fortalezas/debilidades, predicción con nivel de confianza
3. **Análisis de divergencias** — interpreta diferencias entre Bet365, Polymarket y el modelo ML
4. **Resumen de jornada** — tabla con predicción, confianza (⭐/⭐⭐/⭐⭐⭐) y mejores picks del día

Ejemplo de output de `predict_match.py`:

```
Arsenal vs Chelsea | Premier League

── Claude contextual analysis ─────────
{
  "home_attack_strength": 0.78,
  "home_defense_strength": 0.71,
  "away_attack_strength": 0.65,
  ...
  "reasoning": "Arsenal's superior shot conversion and 5-match form..."
}

── Divergence analysis ─────────────────
Polymarket asigna un 12% más de probabilidad al empate vs Bet365,
posiblemente reflejando información sobre ausencias de última hora...

── Full prediction report ───────────────
Prediction: Home Win | Confidence: High
Arsenal llega con ventaja clara en xG y forma reciente...
```

## Estructura del proyecto

```
predictor/
├── main.py              # Pipeline completo
├── predict_match.py     # Predicción de un partido individual
├── data_loader.py       # Descarga y limpieza de datos históricos
├── features.py          # ELO, rolling stats, xG, fatiga, H2H, odds
├── model.py             # LR + RF + XGBoost + Ensemble + backtest
├── polymarket.py        # Cliente Polymarket Gamma API + triple-layer fusion
├── claude_analyst.py    # Integración Claude API
├── visualize.py         # Gráficas matplotlib/seaborn
├── requirements.txt
└── .env.example
```

## Limitaciones

- El modelo no tiene acceso a alineaciones, lesiones ni información de último momento — esos factores los cubre parcialmente la capa de Polymarket y el análisis contextual de Claude.
- La precisión teórica en 3-way prediction ronda 52-55% (vs ~45% baseline de predecir siempre local). Superar ese techo requiere datos de pago (xG real de StatsBomb/Opta, tracking data).
- Las cuotas de Bet365 en los CSVs históricos son pre-partido, no en vivo.
