import pandas as pd

ALLERGEN_MAP = {
    "Milk": ["milk", "cheese", "butter", "cream","panner"],
    "Egg": ["egg","omlet"],
    "Peanut": ["peanut"],
    "Seafood": ["fish", "shrimp", "crab"],
    "Soy": ["soy", "soy sauce"],
    "Gluten": ["wheat", "flour", "bread"]
}

def recommend_recipes(allergies, preferred):
    df = pd.read_csv("data/Food_Recipes_Cleaned.csv")

    ingredient_col = next(
        col for col in df.columns if "ingredient" in col.lower()
    )

    allergies = [a.strip().lower() for a in allergies if a.strip()]
    preferred = [p.strip().lower() for p in preferred if p.strip()]

    results = []

    for _, row in df.iterrows():
        ingredients = str(row[ingredient_col]).lower()

        if any(word in ingredients
               for allergy in allergies
               for word in ALLERGEN_MAP.get(allergy.capitalize(), [])):
            continue

        score = sum(p in ingredients for p in preferred)

        results.append({
            "name": row.get("name", "Unknown"),
            "cuisine": row.get("cuisine", "Unknown"),
            "ingredients": row[ingredient_col],
            "score": score
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
