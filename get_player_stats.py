from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import Position
import pandas as pd
import numpy as np
from typing import Tuple
import logging
import coloredlogs
from sklearn.preprocessing import LabelEncoder

rolling_window = 20


def download_player_stats():
    seasons = [2020, 2019, 2018, 2017]
    schedules = []
    season_stats = []
    for season in seasons:
        schedules.append(pd.DataFrame(client.season_schedule(season_end_year=season)))
        season_stats.append(pd.DataFrame(client.players_season_totals(season_end_year=season)))
    positions = pd.concat(season_stats)[["name", "positions"]]
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

    player_stats = pd.concat(player_stats).merge(positions, on="name", how="left")

    player_stats.to_pickle("player_stats.pkl")


def download_team_stats():
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
    team_stats = []
    for _, row in game_days.iterrows():
        daily_data = pd.DataFrame(client.team_box_scores(day=row["day"], month=row["month"], year=row["year"])).assign(
            game_date=pd.to_datetime(row["game_day"])
        )
        team_stats.append(daily_data)

    team_stats = pd.concat(team_stats)

    team_stats.to_pickle("team_stats.pkl")


def get_player_stats() -> Tuple[np.array, np.array, pd.DataFrame, pd.DataFrame]:
    player_stats = pd.read_pickle("player_stats.pkl").drop(
        columns=[
            "slug",
            "team",
            "location",
            "outcome",
            "attempted_field_goals",
            "attempted_three_point_field_goals",
            "attempted_free_throws",
            "personal_fouls",
            "game_score",
            "opponent",
        ]
    )

    player_stats = _calculate_FD_points(player_stats)
    train_data = player_stats.copy().pipe(_transform_player_data)
    test_data = train_data[:, :, -184:]
    eval_data = pd.DataFrame()
    future_test_data = pd.DataFrame()

    return train_data, test_data, eval_data, future_test_data


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
    game_dates = player_stats.copy(deep=True)[["game_date"]].drop_duplicates()
    player_set = set(player_stats["name"])
    logging.info(f"{len(player_set)} Players in the Model")
    logging.info(f"{len(game_dates)} Games in the Model")
    column_list = player_stats.columns
    output_frame = pd.DataFrame(columns=column_list[1:], index=player_set)
    output_array = []
    for player in player_set:
        temp_player_frame = (
            player_stats[player_stats["name"] == player]
            .merge(game_dates, on="game_date", how="right")
            .fillna(0)
            .sort_values(by="game_date")
        )
        for column in column_list[1:]:
            output_frame.loc[f"{player}", f"{column}"] = np.array(temp_player_frame[column])

    for _, row in output_frame.iterrows():
        output_array.append(np.array(row.to_list()))
    return np.array(output_array)


def build_NN_stats() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_stats = pd.read_pickle("player_stats.pkl").drop(
        columns=[
            "slug",
            "team",
            "location",
            "outcome",
            "attempted_three_point_field_goals",
            "personal_fouls",
            "game_score",
        ]
    )
    player_stats["positions"] = (
        player_stats["positions"]
        .astype(str)
        .map(
            {
                "[<Position.POINT_GUARD: 'POINT GUARD'>]": "point_guard",
                "[<Position.SHOOTING_GUARD: 'SHOOTING GUARD'>]": "shooting_guard",
                "[<Position.SMALL_FORWARD: 'SMALL FORWARD'>]": "small_forward",
                "[<Position.POWER_FORWARD: 'POWER FORWARD'>]": "power_forward",
                "[<Position.CENTER: 'CENTER'>]": "center",
            }
        )
    )
    team_stats = pd.read_pickle("team_stats.pkl")[
        ["team", "game_date", "points", "minutes_played", "attempted_field_goals", "attempted_free_throws", "turnovers", "defensive_rebounds"]
    ].rename(
        columns={
            "team": "opponent",
            "minutes_played": "team_minutes",
            "attempted_field_goals": "team_attempted_field_goals",
            "attempted_free_throws": "team_attempted_free_throws",
            "turnovers": "team_turnovers",
            "defensive_rebounds": "team_defensive_rebounds",
        }
    )
    team_stats["opponent"] = team_stats["opponent"].astype(str)
    player_stats["opponent"] = player_stats["opponent"].astype(str)
    team_stats = team_stats.sort_values(by="game_date").groupby("opponent", group_keys=False).apply(_team_averages)
    player_stats = player_stats.merge(team_stats, on=["opponent", "game_date"], how="left").drop_duplicates()
    player_stats = player_stats[player_stats["seconds_played"] > 500]

    player_stats = _calculate_FD_points(player_stats).pipe(_calculate_usage)
    player_stats = (
        player_stats.sort_values(["name", "game_date"]).groupby("name", group_keys=False).apply(_moving_averages)
    )
    player_stats = _calculate_salary(player_stats)
    train = player_stats[player_stats["game_date"] < "2020-01-01"]
    test = player_stats[player_stats["game_date"] > "2020-01-01"]

    y_train = train["FD_points"]
    x_train = train.drop(columns=["FD_points"])
    y_test = test["FD_points"]
    x_test = test.drop(columns=["FD_points"])

    return x_train, x_test, y_train, y_test


def _calculate_usage(player_stats: pd.DataFrame) -> pd.DataFrame:
    player_stats["usage"] = 100 * (
        (
            (
                (player_stats["attempted_field_goals"])
                + 0.44 * (player_stats["attempted_free_throws"])
                + (player_stats["turnovers"])
            )
        )
        * (player_stats["team_minutes"])
        / (
            (
                (player_stats["team_attempted_field_goals"])
                + 0.44 * (player_stats["team_attempted_free_throws"])
                + player_stats["team_turnovers"]
            )
            * 5
            * (player_stats["seconds_played"] / 60)
        )
    )
    return player_stats


def _calculate_salary(player_stats: pd.DataFrame) -> pd.DataFrame:
    player_stats["salary"] = (
        3258
        + 178 * player_stats["FD_points_avg"]
        + -33.7 * player_stats["FD_points_avg"] ** 2
        + 2.69 * player_stats["FD_points_avg"] ** 3
        + -0.0857 * player_stats["FD_points_avg"] ** 4
        + 1.23e-03 * player_stats["FD_points_avg"] ** 5
        + -6.68e-06 * player_stats["FD_points_avg"] ** 6
    )
    return player_stats


def _moving_averages(player_stats: pd.DataFrame) -> pd.DataFrame:
    return player_stats.assign(
        made_three_point_field_goals_avg=player_stats["made_three_point_field_goals"].rolling(rolling_window).mean().shift(1),
        assists_avg=player_stats["assists"].rolling(rolling_window).mean().shift(1),
        blocks_avg=player_stats["blocks"].rolling(rolling_window).mean().shift(1),
        made_field_goals_avg=player_stats["made_field_goals"].rolling(rolling_window).mean().shift(1),
        made_free_throws_avg=player_stats["made_free_throws"].rolling(rolling_window).mean().shift(1),
        offensive_rebounds_avg=player_stats["offensive_rebounds"].rolling(rolling_window).mean().shift(1),
        defensive_rebounds_avg=player_stats["defensive_rebounds"].rolling(rolling_window).mean().shift(1),
        steals_avg=player_stats["steals"].rolling(rolling_window).mean().shift(1),
        turnovers_avg=player_stats["turnovers"].rolling(rolling_window).mean().shift(1),
        usage_avg=player_stats["usage"].rolling(rolling_window).mean().shift(1),
        FD_points_avg=player_stats["FD_points"].rolling(rolling_window).mean().shift(1),
    )[
        [
            "name",
            "positions",
            "game_date",
            "made_three_point_field_goals_avg",
            "assists_avg",
            "blocks_avg",
            "made_field_goals_avg",
            "made_free_throws_avg",
            "offensive_rebounds_avg",
            "defensive_rebounds_avg",
            "steals_avg",
            "turnovers_avg",
            "FD_points_avg",
            "usage_avg",
            "FD_points",
            "pace",
            "team_defensive_rebounds_avg",
            "team_attempted_free_throws_avg",
            "team_attempted_field_goals_avg",

        ]
    ].dropna()


def _team_averages(team_stats: pd.DataFrame) -> pd.DataFrame:
    return team_stats.assign(
        pace=team_stats["points"].rolling(rolling_window).mean().shift(1),
        team_defensive_rebounds_avg=team_stats["team_defensive_rebounds"].rolling(rolling_window).mean().shift(1),
        team_attempted_free_throws_avg=team_stats["team_attempted_free_throws"].rolling(rolling_window).mean().shift(1),
        team_attempted_field_goals_avg=team_stats["team_attempted_field_goals"].rolling(rolling_window).mean().shift(1),
    )


if __name__ == "__main__":
    download_player_stats()
