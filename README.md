# FridgeChef — a pantry-aware hybrid recipe recommender

Mini project for Recommender Systems (AIM3143), Manipal University Jaipur.

FridgeChef recommends recipes you will like, that you can cook with what is in your fridge,
and that are safe for your diet and allergies. It is a **pipelined (cascade) hybrid**:

1. **Safety filter** (knowledge-based) — diet and allergy rules remove recipes; never relaxed.
2. **Pantry check** (constraint-based) — keep recipes missing at most *m* ingredients;
   if fewer than 10 remain, suggest the smallest soft constraint to relax.
3. **Taste ranking** — collaborative filtering (ALS, item-kNN, …) for known users,
   content-based profiles for new users (switching hybrid).
4. **Context re-ranking** — use-it-up mode boosts recipes that use expiring food.

## Data

Food.com Recipes and User Interactions (Majumder et al., 2019), from Kaggle:
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

Put `RAW_recipes.csv` and `RAW_interactions.csv` (or their `.gz` versions) in `data/`.
The scripts read `data/RAW_recipes.csv.gz` and `data/RAW_interactions.csv.gz`; gzip the CSVs
(`gzip -k RAW_recipes.csv RAW_interactions.csv`) or change the two paths in `src/prep.py`.

## Setup

```bash
pip install numpy pandas scipy scikit-learn implicit scikit-surprise torch
```

## Run (from `src/`)

| Step | Command | What it does | Output |
| --- | --- | --- | --- |
| 1 | `python prep.py` | cleaning, ingredient normalisation, diet/allergen labels, 10-core, temporal split | `processed/` |
| 2 | `python tune.py User-kNN Item-kNN PureSVD Content NeuMF` and `python tune.py ALS BPR` | hyper-parameters on a validation split from train only | `results/tuning_*.json` |
| 3 | `python exp_rating.py` | RMSE/MAE, normalisation, missing-value strategies | `results/exp_rating.json` |
| 4 | `python exp_main.py` | top-10 ranking: 8 base models + 6 hybrid designs | `results/exp_main.json` |
| 5 | `python exp_fridge.py` | pantry-aware ranking, diet personas, relaxation, use-it-up, swaps, critiquing | `results/exp_fridge.json` |
| 6 | `python exp_coldstart.py` | cold start and the switching threshold | `results/exp_coldstart.json` |
| 7 | `python exp_shilling.py` | push attacks and a detection-based defence | `results/exp_shilling.json` |
| 8 | `python export_app.py` | data for the web prototype | `app/data.json` |

## Web prototype

`app/` is a plain HTML, CSS and JavaScript app (no framework):

| File | What it does |
| --- | --- |
| `index.html` | the page: fridge list, diet, allergies, time, shopping limit, loved recipes |
| `style.css` | the Faded Rose look, with `fridge.svg` faded in the background |
| `app.js` | the recommender: safety filter, fridge check, taste ranking, "Quicker" button |
| `data.json` | 5,000 recipes with their neighbour lists, made by `src/export_app.py` |

Open it through a local server (browsers block `fetch` from `file://`):

```bash
cd app && python -m http.server 8000   # then open http://localhost:8000
```

## Code map

| File | Contents |
| --- | --- |
| `src/prep.py` | data cleaning, ingredient normaliser, allergen/diet keyword rules, k-core, temporal split |
| `src/common.py` | `Data` class (index maps, sparse matrices), ranking metrics (P/R/F1/HR/NDCG/MAP@10, coverage, novelty) |
| `src/models.py` | Popularity, User-kNN, Item-kNN, PureSVD, ALS, BPR-MF, NeuMF (PyTorch), Content-based TF-IDF |
| `src/hybrids.py` | weighted, switching, mixed, cascade, feature combination, feature augmentation, meta-level |
| `src/fridge.py` | `Kitchen` (pantry coverage, constraints, relaxation, explanations), ingredient swaps (PPMI + SVD), case-based critiquing |
| `src/exp_*.py` | the experiments listed above |
| `src/export_app.py` | builds the prototype's data file |
