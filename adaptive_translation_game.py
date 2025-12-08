import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone
from skelo.model.glicko2 import Glicko2Model
from fsrs import Scheduler, Card, Rating

MIN_RATING = 600
MAX_RATING = 1800

def rating_to_10_scale(rating):
    scaled = 1 + 9 * (rating - MIN_RATING) / (MAX_RATING - MIN_RATING)
    result = round(min(10, max(1, scaled)), 2)
    print(f"[DEBUG] rating_to_10_scale: {rating} -> {result}")
    return result

def scale_10_to_rating(score_10):
    result = MIN_RATING + (score_10 - 1) * (MAX_RATING - MIN_RATING) / 9
    print(f"[DEBUG] scale_10_to_rating: {score_10} -> {result}")
    return result

USER_FILE = "user_rating.json"

def load_user():
    print(f"[DEBUG] Loading user from {USER_FILE}")
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            raw = json.load(f)
        history = {}
        for sid, obj in raw.get("history", {}).items():
            card = Card.from_dict(obj["card"])
            due = datetime.fromisoformat(obj["due"])
            history[int(sid)] = {"card": card, "due": due}
        raw["history"] = history
        print(f"[DEBUG] Loaded existing user: rating={raw['rating']}, history size={len(history)}")
        return raw
    rating, rd, sigma = 1500.0, 350.0, 0.06
    print(f"[DEBUG] Created new user: rating={rating}, rd={rd}, sigma={sigma}")
    return {
        "rating": rating,
        "rd": rd,
        "sigma": sigma,
        "rating_10": rating_to_10_scale(rating),
        "history": {}
    }

def save_user(user):
    print(f"[DEBUG] Saving user: rating={user['rating']}, history size={len(user['history'])}")
    out = {
        "rating": user["rating"],
        "rd": user["rd"],
        "sigma": user["sigma"],
        "rating_10": user["rating_10"],
        "history": {
            sid: {
                "card": obj["card"].to_dict(),
                "due": obj["due"].isoformat()
            } for sid, obj in user["history"].items()
        }
    }
    with open(USER_FILE, "w") as f:
        json.dump(out, f, indent=4)
    print(f"[DEBUG] User saved to {USER_FILE}")

def evaluate_answer(user_input, correct):
    result = user_input.strip().lower() == correct.strip().lower()
    print(f"[DEBUG] evaluate_answer: '{user_input}' vs '{correct}' -> {result}")
    return result

def select_sentence(df, user, user_elo_10, skill_window=1):
    print(f"[DEBUG] select_sentence: user_elo_10={user_elo_10}, skill_window={skill_window}")
    now = datetime.now(timezone.utc)

    # 1) FSRS due + skill filter
    due_list = []
    for sid, obj in user["history"].items():
        if obj["due"] <= now:
            row = df[df["id"] == sid]
            if row.empty:
                continue
            diff = float(row.iloc[0]["difficulty"])
            if abs(diff - user_elo_10) <= skill_window:
                due_list.append(row.iloc[0])
    
    print(f"[DEBUG] Found {len(due_list)} due sentences within skill window")
    if due_list:
        selected = pd.DataFrame(due_list).sample(1).iloc[0]
        print(f"[DEBUG] Selected due sentence: id={selected['id']}, difficulty={selected['difficulty']}")
        return selected

    # 2) New sampling by difficulty
    center = int(round(user_elo_10))
    min_d = max(1, center - skill_window)
    max_d = min(10, center + skill_window)
    print(f"[DEBUG] Sampling new sentence: difficulty range [{min_d}, {max_d}]")
    pool = df[(df["difficulty"] >= min_d) & (df["difficulty"] <= max_d)]
    if pool.empty:
        print(f"[DEBUG] Pool empty, using entire dataset")
        pool = df
    print(f"[DEBUG] Pool size: {len(pool)} sentences")
    weights = 1 / (1 + np.abs(pool["difficulty"] - user_elo_10))
    selected = pool.sample(1, weights=weights).iloc[0]
    print(f"[DEBUG] Selected new sentence: id={selected['id']}, difficulty={selected['difficulty']}")
    return selected

def start_training(csv_file):
    print(f"[DEBUG] Loading CSV from {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"[DEBUG] Loaded {len(df)} sentences")
    user = load_user()

    model = Glicko2Model()
    scheduler = Scheduler()

    print("Training start — skill:", user["rating_10"])

    while True:
        user_elo_10 = user["rating_10"]
        row = select_sentence(df, user, user_elo_10)
        sid = int(row["id"])
        sentence = row["sentence"]
        correct = row["translation"]
        diff = float(row["difficulty"])
        opp_rating = scale_10_to_rating(diff)

        print("\nTranslate:", sentence)
        print(f"(Difficulty {diff} vs your level {user_elo_10})")
        ans = input("Answer (or 'q' to quit): ").strip()
        if ans.lower() == 'q':
            save_user(user)
            print("Saved. Bye.")
            break

        correct_flag = evaluate_answer(ans, correct)
        print("Correct translation:", correct)
        print("--Correct--" if correct_flag else "--Incorrect--")

        # Glicko update
        r1 = (user["rating"], user["rd"], user["sigma"])
        r2 = (opp_rating, 200.0, 0.06)
        print(f"[DEBUG] Glicko update: r1={r1}, r2={r2}, outcome={1 if correct_flag else 0}")
        new_r, new_rd, new_sig = model.evolve_rating(r1, r2, 1 if correct_flag else 0)
        user["rating"], user["rd"], user["sigma"] = round(new_r,2), round(new_rd,2), round(new_sig,5)
        user["rating_10"] = rating_to_10_scale(user["rating"])

        # FSRS update
        if sid not in user["history"]:
            print(f"[DEBUG] Creating new FSRS card for sentence {sid}")
            user["history"][sid] = {"card": Card(), "due": datetime.now(timezone.utc)}

        card = user["history"][sid]["card"]
        rating = Rating.Good if correct_flag else Rating.Again
        print(f"[DEBUG] FSRS review: sid={sid}, rating={rating}")
        card, review_log = scheduler.review_card(card, rating)
        user["history"][sid]["card"] = card
        user["history"][sid]["due"] = card.due

        # print next due in seconds
        due_seconds = (card.due - datetime.now(timezone.utc)).total_seconds()
        print(f"[DEBUG] FSRS card updated: next due in {due_seconds:.2f} seconds")

        print("New skill:", user["rating_10"], "RD:", user["rd"])
        print("-"*40)

if __name__ == "__main__":
    start_training("output/test3_results.csv")
