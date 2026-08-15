import streamlit as st

# ✅ PAGE CONFIG (ONLY ONCE AND AT TOP)
st.set_page_config(
    page_title="Food Allergy Detection System",
    layout="wide",  # Changed from "centered" to give more space for sidebar
    initial_sidebar_state="expanded"
)
import pickle
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mysql.connector
import hashlib
from recipe_recommender import recommend_recipes

# ================= DATABASE SETUP =================
MYSQL_PASSWORD = '2003'

def get_db_connection():
    return mysql.connector.connect(
        host='localhost', user='root', password=MYSQL_PASSWORD, database='food_allergy_db'
    )

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS allergy_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            food_eaten TEXT,
            symptoms TEXT,
            predicted_allergies TEXT,
            risk_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

create_tables()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(%s)", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return False
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", 
                      (username, password_hash))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except:
        return False

def login_user(username, password):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(%s) AND password_hash = %s", 
                      (username, password_hash))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except:
        return False


# ================= CLEAN CSS (FIXED SIDEBAR ISSUE) =================
import streamlit as st

# ✅ PAGE CONFIG - FORCE SIDEBAR TO SHOW
st.set_page_config(
    page_title="Food Allergy Detection System",
    layout="wide",  # Changed from "centered" to give more space for sidebar
    initial_sidebar_state="expanded"
)

# ✅ CSS TO MAKE SIDEBAR VISIBLE + ATTRACTIVE (replaces your old CSS)
st.markdown("""
<style>
/* Sidebar - Always visible and styled */
[data-testid="stSidebar"] {
    background-color: #1E3A8A !important;
    width: 300px !important;
    min-width: 300px !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main content */
.main .block-container {
    padding-left: 20px;
    padding-right: 20px;
}

/* Hide only menu/footer, NOT sidebar toggle */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.st-emotion-cache-1r4w99b {display: none;} /* Hide Streamlit watermark */

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "predicted_allergies" not in st.session_state:
    st.session_state.predicted_allergies = []

# ================= AUTH =================
def auth_page():
    st.title("🔐 Food Allergy Detection System")
    option = st.radio("Choose option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registered successfully. Please login.")
            else:
                st.error("User already exists")
    else:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")

if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ================= LOAD MODELS =================
rf_model = pickle.load(open("model/allergy_model.pkl", "rb"))
features = pickle.load(open("model/feature_columns.pkl", "rb"))
nlp_model = load_model("model/symptom_lstm_model.h5")
tokenizer = pickle.load(open("model/symptom_tokenizer.pkl", "rb"))
label_map = pickle.load(open("model/symptom_label_map.pkl", "rb"))

# ================= SIDEBAR =================
# ================= SIDEBAR =================
st.sidebar.title("🍽 Food Allergy System")
st.sidebar.write(f"👤 {st.session_state.username}")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧪 Allergy Risk Assessment",
        "🥗 Safer Food Alternatives",
        "🍽 Safe Recipe Suggestions"
    ]
)

if st.sidebar.button("🔓 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.predicted_allergies = []
    st.rerun()


# ================= HELPER FUNCTION =================
def get_risk_color(value):
    if value >= 85:
        return "#DC2626"
    elif value >= 60:
        return "#F59E0B"
    elif value >= 40:
        return "#FACC15"
    else:
        return "#16A34A"

# ================= ALLERGY DATA =================
ALLERGY_KEYWORDS = {
    "Peanut / Nuts": ["peanut", "nuts", "almond", "cashew","pista"],
    "Gluten": ["pizza", "bread", "noodle", "noodles", "burger", "chapati", "wheat",
              "bun","naan","maida","parotha","parota"],
    "Milk": ["milk", "cheese", "butter", "paneer","coffee","tea","milk-powder","milk powder",
           "butter milk","ghee"],
    "Egg": ["egg", "omelette","omelte","egg rice","egg puffs"],
    "Seafood": ["fish", "prawn", "shrimp","crab","crabs"],
    "Lentils / Pulses": ["dal", "lentil", "urad", "toor","moong dal","urad dal","moong",
                        "channa","channa dal","dal kichadi"],
    "Rice-based Foods": ["rice", "idli", "dosa", "pongal","appam","roti"]
}

PRIMARY_ALLERGY_MAP = {
    "skin_allergy": ["Egg", "Peanut / Nuts"],
    "gi_allergy": ["Milk", "Lentils / Pulses"],
    "oral_allergy": ["Peanut / Nuts"],
    "angioedema": ["Egg", "Seafood"],
    "severe_reaction": ["Seafood", "Peanut / Nuts"]
}

ALLERGY_BASE_WEIGHTS = {
    "Egg": 0.8,
    "Peanut / Nuts": 0.95,
    "Seafood": 0.9,
    "Milk": 0.85,
    "Gluten": 0.8,
    "Lentils / Pulses": 0.75,
    "Rice-based Foods": 0.6,
    "General Food Allergy": 0.5
}

SAFE_FOOD_SUGGESTIONS = {
    "Peanut / Nuts": [
        "Rice and rice-based meals",
        "Millet dishes (ragi, jowar, bajra)",
        "Seeds (sunflower, pumpkin)",
        "Lentils and pulses",
        "Green Salads"
    ],
    "Gluten": [
        "Rice, idli, dosa",
        "Millet rotis",
        "Corn-based foods",
        "Potatoes and vegetables"
    ],
    "Milk": [
        "Soy milk",
        "Oat milk",
        "Coconut milk",
    ],
    "Egg": [
        "Moong Dal",
        "Tofu",
        "Dal and legumes"
    ],
    "Lentils / Pulses": [
        "Rice and rice-based meals",
        "Vegetable curries",
        "Dairy products (if tolerated)",
        "Soy-based foods"
    ],
    "Rice-based Foods": [
        "Millet-based dishes",
        "Wheat-free breads (gluten-free)",
        "Potato-based meals",
        "Panner",
        "Quinoa and oats"
    ],
    "General Food Allergy": [
        "Fresh fruits",
        "Fresh vegetables",
        "Plain rice",
        "Boiled potatoes",
        "Panner",
        "Simple home-cooked meals"
    ]
}

# ================= HOME =================
if page == "🏠 Home":
    st.image(r"C:/Users/aksha/OneDrive/Pictures/a1.jpeg", use_container_width=True)
    st.caption("🥗 Eat Safe, Stay Healthy")

    st.markdown(
        f"""
        <div style="background-color:#F0F9FF; padding:30px; border-radius:18px; max-width:1100px; margin:auto; color:#0F172A;">
            <h2>Welcome to Your Dashboard</h2>
            <p>You are logged in as <b>{st.session_state.username}</b>.</p>
            <p>
                This system predicts food allergy risks, suggests safer food alternatives,
                and recommends allergy-safe recipes using AI models.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= Allergy Risk Assessment =================
elif page == "🧪 Allergy Risk Assessment":
    st.title("🧪 Allergy Risk Assessment")

    food = st.text_input("Food eaten")
    quantity = st.selectbox("Quantity of food consumed", ["Small", "Medium", "Large"])
    symptoms = st.text_input("Symptoms experienced")

    q1 = st.selectbox("When did symptoms start?", ["<1 hour", "1–6 hours", ">6 hours"])
    q2 = st.radio("Same symptoms every time?", ["Yes", "No"])
    q3 = st.radio("Family history of allergy?", ["Yes", "No"])
    q3_type = st.text_input("Which allergy in family?") if q3 == "Yes" else ""

    q4 = st.radio("Did you take medicine?", ["Yes", "No"])
    q5 = st.selectbox("Did the medicine resolve the symptoms?", ["Yes", "No", "Partially"]) if q4 == "Yes" else "No"

    q6 = st.radio("Similar symptoms with other foods?", ["Yes", "No"])
    q6_food = st.text_input("Which other food?") if q6 == "Yes" else ""
    q7 = st.radio("Consulted a doctor?", ["Yes", "No"])

    if st.button("Predict Allergy"):
        combined = f"{food} {symptoms} {q3_type} {q6_food}".lower()

        seq = tokenizer.texts_to_sequences([symptoms])
        padded = pad_sequences(seq, maxlen=10)
        pred = nlp_model.predict(padded)
        symptom_type = label_map[pred.argmax()]

        data = dict.fromkeys(features, 0)
        df = pd.DataFrame([data]).reindex(columns=features, fill_value=0)
        proba = rf_model.predict_proba(df)[0]
        risk = proba[1] * 100 if len(proba) == 2 else proba[0] * 100

        # ALL 7 QUESTIONS ACTIVE (NO BREAKDOWN DISPLAY)
        if quantity == "Large": risk += 12
        if q1 == "<1 hour": risk += 15
        if q2 == "Yes": risk += 8
        if q3 == "Yes": risk += 10
        if q4 == "Yes" and q5 != "Yes": risk += 12
        if q6 == "Yes": risk += 8
        if q7 == "No": risk += 6
        if symptom_type == "severe_reaction": risk += 18

        risk = min(risk, 100)

        detected = [a for a, keys in ALLERGY_KEYWORDS.items() if any(k in combined for k in keys)]
        if not detected:
            detected = ["General Food Allergy"]

        st.session_state.predicted_allergies = detected

        # SAVE TO DATABASE
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO allergy_predictions (username, food_eaten, symptoms, predicted_allergies, risk_score)
                VALUES (%s, %s, %s, %s, %s)
            """, (st.session_state.username, food, symptoms, ','.join(detected), float(risk)))
            conn.commit()
            cursor.close()
            conn.close()
        except:
            pass

        allergy_risks = {}
        for i, a in enumerate(detected):
            allergy_risks[a] = max(30, risk - i * 5)

        fig = go.Figure(
            go.Pie(
        labels=list(allergy_risks.keys()),
        values=list(allergy_risks.values())
         )
        )
        fig.update_layout(
            title="Allergy-wise Risk Prediction",
            yaxis=dict(range=[0, 110], title="Risk (%)")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("⚠️ AI-based prediction. Not a medical diagnosis.")

# ================= TAB 2 =================
elif page == "🥗 Safer Food Alternatives":
    st.title("🥗 Safer Food Alternatives")

    allergies = st.session_state.predicted_allergies

    if not allergies:
        st.info("Please complete the allergy prediction first.")
    else:
        st.markdown(
            f"""
            **You are likely allergic to:** {', '.join(allergies)}
            This means you should avoid foods that commonly contain these ingredients.
            Below are safer and nutritious alternatives you can include in your diet.
            """
        )

        st.subheader("🥗 Safer Food Options")

        shown = set()
        for allergy in allergies:
            for food in SAFE_FOOD_SUGGESTIONS.get(allergy, []):
                if food not in shown:
                    st.markdown(f"• 🍽️ **{food}**")
                    shown.add(food)

# ================= TAB 3 =================
elif page == "🍽 Safe Recipe Suggestions":
    st.title("🍽 Safe Recipe Suggestions")

    allergies = st.text_input("Your allergies (comma separated)")
    preferred = st.text_input("Preferred ingredients")

    if st.button("Get Recipes"):
        results = recommend_recipes(allergies.split(","), preferred.split(","))
        if not results:
            st.warning("No safe recipes found.")
        else:
            for r in results[:5]:
                st.subheader(r["name"])
                st.write("Cuisine:", r["cuisine"])
                st.write("Ingredients:", r["ingredients"])
