import pandas as pd
INPUT_PATH = "data/ucl_fantasy_dataset.csv"
OUTPUT_PATH = "data/modeling_dataset.csv"

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(by=["player", "date"])

# List of past performance stats that we want to use for predicting the next game
rolling_cols = [
     # Overall fantasy output / playing time
    "fantasy_points",
    "minutes",

    # Attacking stats
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "xg",
    "npxg",
    "xag",
    "sca",
    "gca",

    # Defensive stats
    "tackles",
    "interceptions",
    "blocks",
    "ball_recoveries",

    # Discipline
    "yellow_cards",
    "red_cards",
    "fouls",

    # Goalkeeper / team defensive scoring
    "saves",
    "penalties_saved",
    "goals_conceded",
    "clean_sheet",

    # Other fantasy scoring events
    "own_goals",
    "penalties_won",
    "pens_made",
    "pens_att",
]

# Create past performance features for each player
# We want to use only stats from before the current match
for col in rolling_cols:

    # shift(1) moves each player's stats down one row, so current row sees previous match
    shifted = df.groupby("player")[col].shift(1)

    # tells the model what the player did in their previous match
    df[f"{col}_prev"] = shifted

    # tells the model the average of the player's previous 3 matches
    # gives model context to a player's current form  
    df[f"{col}_last3_avg"] = (
        shifted.groupby(df["player"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # similar to last 3, but for the last 5 matches
    df[f"{col}_last5_avg"] = (
        shifted.groupby(df["player"])
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# tells the model if the player played in the previous match
df["played_prev"] = (df["minutes_prev"] > 0).astype(int)
# tells the model if the player played for at least 60 minutes in the previous match
df["played_60_prev"] = (df["minutes_prev"] >= 60).astype(int)

# tells the model if they are a regular starter or barely played in the previous 3 matches
df["minutes_last3_sum"] = (
    df.groupby("player")["minutes"]
    .shift(1)
    .groupby(df["player"])
    .rolling(window=3, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)

# 1 means they played at least 120 total minutes over their previous 3 matches.
# 0 means they played less than that.
df["regular_recently"] = (df["minutes_last3_sum"] >= 120).astype(int)

# they describe the player, team, match, and season.
# just so we know what each row represents.
context_cols = [
    "player",
    "team",
    "opponent",
    "home_away",
    "fantasy_position",
    "season",
    "matchday",
    "date",
]

target_col = "fantasy_points"

feature_cols = [
    col for col in df.columns
    if col.endswith("_prev")
    or col.endswith("_last3_avg")
    or col.endswith("_last5_avg")
    or col in [
        "played_prev",
        "played_60_prev",
        "minutes_last3_sum",
        "regular_recently",
    ]
]

modeling_df = df[context_cols + feature_cols + [target_col]].copy()

# remove rows where the player has no previous-match history.
modeling_df = modeling_df.dropna(subset=["fantasy_points_prev", "minutes_prev"])

modeling_df = modeling_df.round(3)

modeling_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(modeling_df)} rows to {OUTPUT_PATH}")

print(modeling_df.head())