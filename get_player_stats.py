from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import Position
import pandas as pd
from typing import Tuple


def _download_player_stats():
    seasons = [2020, 2019, 2018, 2017]
    schedules = []
    for season in seasons:
        schedules.append(pd.DataFrame(client.season_schedule(season_end_year=season)))
    schedules = pd.concat(schedules)
    schedules["start_time"] = schedules["start_time"].dt.date
    game_days = pd.DataFrame(data={"game_day": schedules["start_time"].unique().astype(str)})
    game_days["day"] = (game_days["game_day"].str[-2:]).astype(int)
    game_days["month"] = (game_days["game_day"].str[-5:-3]).astype(int)
    game_days["year"] = (game_days["game_day"].str[:4]).astype(int)
    player_stats = []
    for _, row in game_days.iterrows():
        daily_data = pd.DataFrame(
            client.player_box_scores(day=row["day"], month=row["month"], year=row["year"])
        ).assign(game_date=pd.to_datetime(row["game_day"]))
        player_stats.append(daily_data)

    player_stats = pd.concat(player_stats)

    player_stats.to_pickle("player_stats.pkl")


def get_player_stats() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_stats = pd.read_pickle("player_stats.pkl")

    player_stats = _calculate_FD_points(player_stats)
    player_stats = _transform_player_data(player_stats)

    train_data = player_stats[(player_stats["game_date"] >= "2018-10-01") & (player_stats["game_date"] < "2019-10-01")]
    test_data = player_stats[(player_stats["game_date"] >= "2019-10-01") & (player_stats["game_date"] < "2021-01-16")]
    eval_data = pd.DataFrame()
    future_test_data = pd.DataFrame()

    return test_data, train_data, eval_data, future_test_data


def _calculate_FD_points(player_stats: pd.DataFrame) -> pd.DataFrame:
    FD_points = (
        player_stats["made_three_point_field_goals"]
        + (player_stats["assists"] * 1.5)
        + (player_stats["blocks"] * 3)
        + (player_stats["made_field_goals"] * 2)
        + (player_stats["made_free_throws"] * 1)
        + (player_stats["offensive_rebounds"] + player_stats["defensive_rebounds"]) * 1.2
        + (player_stats["steals"] * 3)
        + (player_stats["turnovers"] * -1)
    )
    return player_stats.assign(FD_points=FD_points)


def _transform_player_data(player_stats: pd.DataFrame) -> pd.DataFrame:
    output_frame = player_stats.copy(deep=True)[["game_date"]].drop_duplicates()
    for player in set(player_stats["name"]):
        temp_player_frame = player_stats[player_stats["name"] == player][["game_date", "FD_points", "seconds_played"]].rename(columns={"FD_points": f"{player}"})
        if (temp_player_frame[f"{player}"].dropna().median() < 38):
            continue
        output_frame = output_frame.merge(temp_player_frame.drop(columns="seconds_played"), how="outer", on="game_date").fillna(0)
    return output_frame
