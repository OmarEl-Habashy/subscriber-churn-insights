import pandas as pd
import numpy as np

NORMALIZED_COLUMNS = [
    "user_id", "session_id", "ts", "registration", "page", "level", "gender", "artist", "song", "length"
]

def load_sparkify_events(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    df = df.dropna(subset=["userId", "sessionId"])
    df = df[df["userId"].astype(str).str.strip() != ""]
    df = df.rename(columns={
        "userId": "user_id",
        "sessionId": "session_id",
    })

    keep = [c for c in NORMALIZED_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)

def label_churn(events: pd.DataFrame, churn_page: str = "Cancellation Confirmation") -> pd.DataFrame:
    churned_users = events.loc[events["page"] == churn_page, "user_id"].unique()
    all_users = events["user_id"].unique()
    return pd.DataFrame({
        "user_id": all_users,
        "churn": np.isin(all_users, churned_users).astype(int),
    })

def build_feature_table(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["ts"] = pd.to_numeric(events["ts"], errors = "coerce")
    events["registration"] = pd.to_numeric(events["registration"], errors="coerce")
    labels = label_churn(events)
    grouped = events.groupby("user_id")

#--- base count ---

    tot_songs = grouped.apply(lambda g : (g["page"] == "NextSong").sum()).rename("tot_songs")
    listen_time = grouped.apply(lambda g : g.loc[g["page"] == "NextSong", "length"].sum()).rename("listen_time")
    num_thumb_up = grouped.apply(lambda g : (g["page"] == "Thumbs Up").sum()).rename("num_thumb_up")
    num_thumb_down = grouped.apply(lambda g : (g["page"] == "Thumbs Down").sum()).rename("num_thumb_down")
    add_to_playlist = grouped.apply(lambda g : (g["page"] == "Add to Playlist").sum()).rename("add_to_playlist")
    tot_friends = grouped.apply(lambda g : (g["page"] == "Add Friend").sum()).rename("tot_friends")
    tot_artist_played = grouped.apply(
        lambda g: g.loc[g["page"] == "NextSong", "artist"].nunique()
    ).rename("tot_artist_played")
    downgrade_flag = grouped.apply(lambda g: int((g["page"] == "Downgrade").any())).rename("downgrade_flag")
    ads_seen = grouped.apply(lambda g: (g["page"] == "Roll Advert").sum()).rename("ads_seen")    

    #Lifetime
    lt = grouped.apply(lambda g: g["ts"].max() - g["registration"].iloc[0]).rename("lt")

    #recency or recent events
    global_max_ts = events["ts"].max()
    last_seen = grouped["ts"].max()
    days_since_last_session = ((global_max_ts - last_seen) / (1000 * 60 * 60 * 24)).rename(
        "days_since_last_session"
    )

    #some Aggregations
    session_song_counts = (
        events[events["page"] == "NextSong"]
        .groupby(["user_id", "session_id"])
        .size()
    )
    avg_played_songs = session_song_counts.groupby("user_id").mean().rename("avg_played_songs")
 
    n_sessions = grouped["session_id"].nunique().rename("n_sessions")
    span_days = ((grouped["ts"].max() - grouped["ts"].min()) / (1000 * 60 * 60 * 24)).clip(lower=1)
    sessions_per_week = (n_sessions / span_days * 7).rename("sessions_per_week")

    #gender
    gender_raw = grouped["gender"].first()
    gender = gender_raw.map({"F": 1, "M": 0}).rename("gender")

    #some ratios
    negative_ratio = (num_thumb_down / (num_thumb_up + num_thumb_down + 1)).rename("negative_ratio")
    ads_per_session = (ads_seen / n_sessions.replace(0, np.nan)).fillna(0).rename("ads_per_session")

    features = pd.concat([
        gender, tot_songs, listen_time, num_thumb_up, num_thumb_down,
        add_to_playlist, tot_friends, lt, avg_played_songs, tot_artist_played,
        days_since_last_session, downgrade_flag, negative_ratio,
        sessions_per_week, ads_per_session,
    ], axis=1).reset_index()
 
    features = features.merge(labels, on="user_id", how="left")
    features = features.fillna(0)
 
    return features 


if __name__ == "__main__":
    import sys
 
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/mini_sparkify_event_data.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/sparkify_features.csv"
 
    print(f"Loading events from {input_path} ...")
    events = load_sparkify_events(input_path)
    print(f"  {len(events):,} events, {events['user_id'].nunique():,} unique users")
 
    print("Building feature table ...")
    table = build_feature_table(events)
    print(f"  {len(table):,} rows, churn rate: {table['churn'].mean():.1%}")
 
    table.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")