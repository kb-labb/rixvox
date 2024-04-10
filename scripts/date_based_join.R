library(readr)
library(dplyr)
library(lubridate)
library(arrow)
library(purrr)
library(tidyr)
library(data.table)

# Make console output wider
options(width = 120)

# Load data
df <- arrow::read_parquet("data/riksdagen_transcriptions.parquet")
df_speeches <- arrow::read_parquet("data/riksdagen_speeches.parquet")

df <- df %>%
    mutate(
        date_start = lubridate::ymd(date_start),
        date_end = lubridate::ymd(date_end)
    )

df_speeches$date <- lubridate::ymd(df_speeches$date)

# df$text_timestamps is a column where every row is a dataframe
# with the following columns: end_time, start_time, word.
# We want to add the value of start/1000 and end/1000 to the
# columns in these dfs to get global timestamps.
df <- df %>%
    unnest(text_timestamps)

df <- df %>%
    mutate(
        start_time = start_time + start / 1000,
        end_time = end_time + start / 1000
    ) %>%
    nest(text_timestamps = c(start_time, end_time, word)) %>%
    group_by(audio_path) %>%
    summarise(
        text = paste0(text, collapse = " "),
        date_start = first(date_start),
        date_end = first(date_end),
        # Concatenate the tibbles
        text_timestamps = list(bind_rows(text_timestamps))
    )


df_final <- setDT(df_speeches)[df,
    .(name, start, end, speaker_id, speech_id, date, date_start, date_end),
    on = .(date >= date_start, date <= date_end)
]


df_speeches$date <- lubridate::ymd(df_speeches$date)

# Join by date
df %>%
    mutate(
        date_start = lubridate::ymd(date_start),
        date_end = lubridate::ymd(date_end)
    ) %>%
    left_join(df_speeches, by = join_by(
        date,
        between(date, date_start, date_end)
    ))

df_all <- df %>%
    full_join(df_speeches, by = character()) %>%
    filter(date >= date_start & date <= date_end)

# group by audio_path and concatenate all the text
# Choose the first date_start and end_date
df_speeches <- df %>%
    group_by(audio_path) %>%
    summarise(
        text = paste0(text, collapse = " "),
        date_start = first(date_start),
        date_end = first(date_end)
    )

library(fuzzyjoin)

df$text_timestamps[1]
