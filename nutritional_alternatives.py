import numpy as np

def recommend_nutritional_alternatives(
    allergic_food,
    detected_allergies,
    food_df,
    top_n=5
):
    detected_allergies = [a.lower().strip() for a in detected_allergies]

    allergic_row = food_df[
        food_df["food_name"].str.lower() == allergic_food.lower()
    ]

    if allergic_row.empty:
        return []

    target_nutrition = allergic_row[
        ["calories", "protein_g", "fat_g", "carbs_g"]
    ].values[0]

    safe_foods = food_df[
        ~food_df["contains_allergens"]
        .fillna("")
        .str.lower()
        .apply(lambda x: any(a in x for a in detected_allergies))
    ]

    safe_foods = safe_foods[
        safe_foods["food_name"].str.lower() != allergic_food.lower()
    ]

    if safe_foods.empty:
        return []

    nutrition_matrix = safe_foods[
        ["calories", "protein_g", "fat_g", "carbs_g"]
    ].values

    distances = np.linalg.norm(nutrition_matrix - target_nutrition, axis=1)

    safe_foods = safe_foods.copy()
    safe_foods["distance"] = distances

    recommendations = safe_foods.sort_values("distance").head(top_n)

    return recommendations[
        ["food_name", "calories", "protein_g", "fat_g", "carbs_g"]
    ].to_dict(orient="records")
