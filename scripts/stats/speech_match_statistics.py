import pandas as pd

df = pd.read_parquet("data/rixvox-alignments_bleu.parquet")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["year"] = df["date"].dt.year

df_riksdag = pd.read_parquet("data/riksdagen_speeches_new.parquet")
# Aggregate all dates for each speech_id as a list
df_riksdag["date"] = df_riksdag["date"].dt.strftime("%Y-%m-%d")
df_riksdag["dates"] = df_riksdag["speech_id"].map(
    df_riksdag.groupby("speech_id")["date"].agg(list)
)
df_riksdag["date"] = pd.to_datetime(df_riksdag["date"])
df_riksdag = df_riksdag[
    (df_riksdag["date"] > (pd.to_datetime(df["date"].min()) - pd.Timedelta(weeks=1)))
    & (df_riksdag["date"] < (pd.to_datetime(df["date"].max()) + pd.Timedelta(weeks=1)))
]
df_riksdag = df_riksdag.groupby("speech_id").first().reset_index()
df_riksdag["year"] = df_riksdag["date"].dt.year

stats_audio = (
    df.groupby("year")
    .agg(
        {
            "duration_segment": lambda x: x.sum() / 60 / 60,
            "speech_id": "count",
            "text_normalized": lambda x: sum([len(s.split()) for s in x]),
        }
    )
    .reset_index()
)

stats_speeches = (
    df_riksdag.groupby("year")
    .agg(
        {
            "speech_id": "count",
        }
    )
    .reset_index()
)

stats = stats_speeches.merge(stats_audio, on="year")

stats = stats.rename(
    columns={
        "duration_segment": "hours_matched",
        "speech_id_x": "nr_speeches_total",
        "speech_id_y": "nr_matched_speeches",
        "text_normalized": "nr_matched_words",
    }
)
stats_speeches = stats_speeches.rename(
    columns={"speech_id": "nr_speeches_total", "anftext_normalized": "nr_words_total"}
)

stats["match_fraction"] = stats["nr_matched_speeches"] / stats["nr_speeches_total"]

stats = stats[
    [
        "year",
        "hours_matched",
        "nr_speeches_total",
        "nr_matched_speeches",
        "match_fraction",
    ]
]
# Save year and nr_words and nr_speeches_columns columns as strings
stats[["year", "nr_speeches_total", "nr_matched_speeches", "hours_matched", "match_fraction"]] = (
    stats[
        ["year", "nr_speeches_total", "nr_matched_speeches", "hours_matched", "match_fraction"]
    ].astype(str)
)

# Stats to markdown with only 2 decimals for float columns, copy to clipboard
print(stats.to_markdown(floatfmt=".2f", index=False))
