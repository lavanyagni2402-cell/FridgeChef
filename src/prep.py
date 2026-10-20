"""
FridgeChef - data preparation
-----------------------------
Loads the Food.com "Recipes and User Interactions" dataset (Majumder et al., 2019),
cleans it and writes the processed files used by every experiment.

Steps
  1. Parse list columns (ingredients, tags, nutrition).
  2. Normalise ingredient names  ("garlic cloves" / "garlic clove" -> "garlic").
  3. Derive knowledge attributes per recipe: diet flags (from Food.com tags) and
     allergen flags (from keyword rules over ingredient names).
  4. Handle missing values: rating 0 = review without stars -> missing explicit rating,
     kept as an implicit "cooked it" signal.
  5. Iterative 10-core filtering (every user and recipe has >= 10 interactions).
  6. Temporal split per user: oldest 80% of interactions -> train, newest 20% -> test.
"""
import ast
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = ROOT / "processed"
OUT.mkdir(exist_ok=True)

K_CORE = 10
TEST_FRAC = 0.2

# ---------------------------------------------------------------- ingredients
DROP_WORDS = {
    "fresh", "freshly", "ground", "chopped", "minced", "diced", "sliced", "large", "small",
    "medium", "extra", "virgin", "unsalted", "salted", "all-purpose", "granulated", "boneless",
    "skinless", "lean", "low-fat", "fat-free", "reduced-fat", "light", "lite", "dried",
    "frozen", "canned", "shredded", "grated", "finely", "plain", "pure", "raw", "whole",
    "crushed", "fine", "coarse", "packed", "softened", "melted", "cold", "warm", "hot",
    "kosher", "sea", "organic", "lowfat", "nonfat", "low", "sodium", "fat", "free",
    "reduced", "boiling", "cooked", "uncooked", "peeled", "seedless", "ripe", "baby",
}
SYNONYMS = {
    "garlic clove": "garlic", "clove garlic": "garlic", "garlic cloves": "garlic",
    "egg": "egg", "eggs": "egg", "egg white": "egg", "egg yolk": "egg",
    "onions": "onion", "yellow onion": "onion", "white onion": "onion",
    "scallion": "green onion", "spring onion": "green onion",
    "flour": "flour", "plain flour": "flour", "white flour": "flour",
    "black pepper": "pepper", "salt and pepper": "salt", "pepper black": "pepper",
    "vanilla extract": "vanilla", "confectioners sugar": "powdered sugar",
    "icing sugar": "powdered sugar", "vegetable oil": "oil", "canola oil": "oil",
    "cooking oil": "oil", "olive oil": "olive oil", "tomato": "tomato", "potato": "potato",
    "cilantro": "coriander", "fresh cilantro": "coriander", "coriander leaves": "coriander",
    "parmesan": "parmesan cheese", "cheddar": "cheddar cheese", "unsalted butter": "butter",
    "margarine": "margarine", "chicken stock": "chicken broth", "beef stock": "beef broth",
}
STAPLES = {"salt", "pepper", "water", "ice", "cooking spray", "nonstick cooking spray", "ice cube"}


def singular(word: str) -> str:
    if len(word) <= 3 or word.endswith("ss") or word in {"hummus", "couscous", "asparagus", "molasses"}:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def normalise_ingredient(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z\s\-]", " ", s)
    if s in SYNONYMS:
        return SYNONYMS[s]
    words = [w for w in s.split() if w not in DROP_WORDS]
    if not words:
        words = s.split()
    words = [singular(w) for w in words]
    s = " ".join(words).strip()
    return SYNONYMS.get(s, s)


# ---------------------------------------------------------------- allergens / diet
ALLERGEN_RULES = {
    # keyword -> flagged unless an exception keyword is also present
    "nuts": (["almond", "walnut", "pecan", "cashew", "peanut", "hazelnut", "pistachio",
              "macadamia", "pine nut", "brazil nut", "praline", "nutella", "mixed nut", " nut"],
             ["nutmeg", "butternut", "coconut", "water chestnut", "doughnut", "donut"]),
    "dairy": (["milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "ghee", "whey",
               "buttermilk", "parmesan", "mozzarella", "ricotta", "half-and-half", "custard"],
              ["coconut milk", "almond milk", "soy milk", "rice milk", "peanut butter",
               "cream of tartar", "coconut cream", "butternut", "apple butter",
               "almond butter", "cocoa butter", "oat milk", "non-dairy", "dairy-free",
               "cashew butter", "sunflower butter", "butter bean", "creamed corn",
               "cream-style corn", "cream style corn", "ice cream cone", "creamer"]),
    "gluten": (["flour", "wheat", "bread", "pasta", "spaghetti", "noodle", "macaroni",
                "barley", "rye", "couscous", "cracker", "breadcrumb", "bread crumb",
                "biscuit", "tortilla", "pie crust", "cake mix", "beer", "soy sauce",
                "panko", "bulgur", "semolina", "orzo", "lasagna", "penne", "fettuccine",
                "linguine", "graham", "croissant", "bun", "pita", "bagel", "seitan"],
               ["rice flour", "almond flour", "coconut flour", "corn tortilla",
                "gluten-free", "gluten free", "tamari", "rice noodle", "cornflour",
                "corn flour", "potato flour", "tapioca flour", "chickpea flour",
                "buckwheat flour", "rice pasta", "root beer", "ginger beer"]),
    "egg": (["egg", "mayonnaise", "mayo", "meringue", "eggnog"],
            ["eggplant", "egg-free", "egg substitute", "egg replacer", "egg roll wrapper"]),
    "shellfish": (["shrimp", "prawn", "crab", "lobster", "scallop", "clam", "mussel",
                   "oyster", "crawfish", "crayfish"], ["oyster sauce", "oyster mushroom"]),
}
MEAT_WORDS = ["chicken", "beef", "pork", "bacon", "ham", "sausage", "turkey", "lamb", "veal",
              "steak", "fish", "shrimp", "tuna", "salmon", "anchovy", "anchovies", "gelatin",
              "prosciutto", "pepperoni", "salami", "chorizo", "crab", "lobster", "duck",
              "venison", "mutton", "meat", "cod", "tilapia", "halibut", "clam", "mussel",
              "scallop", "oyster", "lard", "pancetta", "hot dog", "frankfurter", "goat",
              "kielbasa", "bratwurst", "corned", "sardine", "trout", "catfish", "mahi",
              "squid", "calamari", "octopus", "brisket", "drumstick", "fish sauce",
              "chuck", "sirloin", "tenderloin", "pastrami", "bologna", "giblet", "liver",
              "worcestershire"]
MEAT_EXCEPTIONS = ["vegetarian ground beef", "vegetarian chicken", "vegetarian bacon",
                   "vegetarian sausage", "vegetarian", "veggie", "meatless", "imitation",
                   "soy chorizo", "vegan", "coconut meat", "beefsteak tomato", "plant-based",
                   "meat tenderizer", "almond meat", "nutmeat", "pecan meat", "walnut meat",
                   "no beef", "no chicken", "goat cheese", "goats cheese", "goat's cheese",
                   "goat milk", "goat's milk", "goats milk", "goat butter", "chicken-flavored",
                   "beef-flavored", "chicken flavored", "beef flavored", "vegetable broth"]
DAIRY_EGG_EXTRA = ["honey"]  # for vegan check


_PATTERNS = {}


def _pattern(keys):
    tup = tuple(keys)
    if tup not in _PATTERNS:
        alts = "|".join(re.escape(k.strip()) for k in keys)
        # whole-word match, optional plural: "almond" matches "almonds", not "slivered"->"liver"
        _PATTERNS[tup] = re.compile(r"\b(?:" + alts + r")(?:s|es)?\b")
    return _PATTERNS[tup]


def has_keyword(text, keys, excepts):
    t = " " + text + " "
    for e in excepts:
        t = t.replace(e, " ")
    return bool(_pattern(keys).search(t))


def allergen_flags(raw_ingredients):
    joined = " | ".join(i.lower() for i in raw_ingredients)
    out = {a: has_keyword(joined, keys, exc) for a, (keys, exc) in ALLERGEN_RULES.items()}
    out["meat"] = has_keyword(joined, MEAT_WORDS, MEAT_EXCEPTIONS)
    return out


def main():
    print("loading raw files ...")
    rec = pd.read_csv(RAW / "RAW_recipes.csv.gz")
    inter = pd.read_csv(RAW / "RAW_interactions.csv.gz",
                        usecols=["user_id", "recipe_id", "date", "rating"])
    stats = {"raw_recipes": len(rec), "raw_interactions": len(inter),
             "raw_users": int(inter.user_id.nunique()),
             "raw_rating_dist": inter.rating.value_counts().sort_index().to_dict(),
             "raw_density": len(inter) / inter.user_id.nunique() / len(rec)}

    # ---- recipes
    rec = rec.rename(columns={"id": "recipe_id"})
    rec["name"] = rec["name"].fillna("untitled recipe").str.replace(r"\s+", " ", regex=True).str.strip()
    rec["ingredients_raw"] = rec["ingredients"].apply(ast.literal_eval)
    rec["tags"] = rec["tags"].apply(ast.literal_eval)
    nut = rec["nutrition"].apply(ast.literal_eval)
    rec["calories"] = nut.str[0]
    rec["ingredients"] = rec["ingredients_raw"].apply(
        lambda L: sorted({normalise_ingredient(x) for x in L if normalise_ingredient(x)}))
    rec["n_ingredients_clean"] = rec["ingredients"].apply(len)
    rec["needs"] = rec["ingredients"].apply(lambda L: [x for x in L if x not in STAPLES])
    # outlier cooking times: cap absurd values (> 1 day) as missing
    rec["minutes"] = rec["minutes"].where(rec["minutes"].between(1, 24 * 60))
    stats["minutes_missing_after_cleaning"] = int(rec["minutes"].isna().sum())

    flags = rec["ingredients_raw"].apply(allergen_flags).apply(pd.Series)
    rec = pd.concat([rec, flags.add_prefix("has_")], axis=1)
    tagset = rec["tags"].apply(set)
    rec["tag_vegetarian"] = tagset.apply(lambda s: "vegetarian" in s or "vegan" in s)
    rec["tag_vegan"] = tagset.apply(lambda s: "vegan" in s)
    rec["tag_gluten_free"] = tagset.apply(lambda s: "gluten-free" in s)
    # diet rules: tag says so AND keyword rules agree (belt and braces)
    rec["is_vegetarian"] = rec["tag_vegetarian"] & ~rec["has_meat"]
    rec["is_vegan"] = rec["tag_vegan"] & ~rec["has_meat"] & ~rec["has_dairy"] & ~rec["has_egg"]

    # validation of the meat keyword rule against Food.com's own vegetarian tag
    veg = rec["tag_vegetarian"]
    stats["meat_rule_flags_tagged_veg_pct"] = float(rec.loc[veg, "has_meat"].mean() * 100)
    stats["meat_rule_flags_untagged_pct"] = float(rec.loc[~veg, "has_meat"].mean() * 100)
    stats["n_canonical_ingredients"] = int(len({i for L in rec["ingredients"] for i in L}))
    stats["n_raw_ingredients"] = int(len({i for L in rec["ingredients_raw"] for i in L}))

    # ---- interactions: missing ratings
    inter["date"] = pd.to_datetime(inter["date"])
    inter["has_rating"] = inter["rating"] > 0
    stats["missing_ratings"] = int((~inter.has_rating).sum())

    # ---- k-core
    d = inter.copy()
    while True:
        uc = d.user_id.value_counts()
        rc = d.recipe_id.value_counts()
        d2 = d[d.user_id.map(uc).ge(K_CORE) & d.recipe_id.map(rc).ge(K_CORE)]
        if len(d2) == len(d):
            break
        d = d2
    d = d.drop_duplicates(["user_id", "recipe_id"], keep="last")
    stats.update({"core_interactions": len(d), "core_users": int(d.user_id.nunique()),
                  "core_recipes": int(d.recipe_id.nunique()),
                  "core_density": len(d) / d.user_id.nunique() / d.recipe_id.nunique(),
                  "core_rating_dist": d.rating.value_counts().sort_index().to_dict()})

    # ---- temporal split per user
    d = d.sort_values(["user_id", "date", "recipe_id"])
    d["rank"] = d.groupby("user_id").cumcount()
    d["n"] = d.groupby("user_id")["recipe_id"].transform("size")
    d["n_test"] = np.maximum(1, np.round(d["n"] * TEST_FRAC)).astype(int)
    d["split"] = np.where(d["rank"] >= d["n"] - d["n_test"], "test", "train")
    train = d[d.split == "train"].drop(columns=["rank", "n", "n_test", "split"])
    test = d[d.split == "test"].drop(columns=["rank", "n", "n_test", "split"])
    # test recipes must exist in train (otherwise no CF model can know them)
    before = len(test)
    test = test[test.recipe_id.isin(set(train.recipe_id))]
    stats.update({"train_interactions": len(train), "test_interactions": len(test),
                  "test_dropped_unseen_recipe": before - len(test),
                  "split_cutoff_median_date": str(test.date.median().date())})

    core_ids = set(d.recipe_id)
    rec["in_core"] = rec["recipe_id"].isin(core_ids)
    keep_cols = ["recipe_id", "name", "minutes", "n_steps", "n_ingredients", "n_ingredients_clean",
                 "ingredients", "ingredients_raw", "needs", "tags", "calories", "description",
                 "has_nuts", "has_dairy", "has_gluten", "has_egg", "has_shellfish", "has_meat",
                 "tag_vegetarian", "tag_vegan", "tag_gluten_free", "is_vegetarian", "is_vegan",
                 "in_core", "submitted"]
    rec[keep_cols].to_pickle(OUT / "recipes_all.pkl")
    train.to_pickle(OUT / "train.pkl")
    test.to_pickle(OUT / "test.pkl")
    inter.to_pickle(OUT / "interactions_all.pkl")
    (OUT / "prep_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
