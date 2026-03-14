import pytest
from pydantic import ValidationError

from load_excel_to_db import MatchRow, PlayerRow


def test_player_row_validation_ok() -> None:
    row = PlayerRow(
        player_name="  Shai Gilgeous-Alexander ",
        team_code="okc",
        age=26,
        points_total=2485,
    )
    assert row.player_name == "Shai Gilgeous-Alexander"
    assert row.team_code == "OKC"


def test_player_row_validation_rejects_invalid_age() -> None:
    with pytest.raises(ValidationError):
        PlayerRow(
            player_name="Player",
            team_code="MIA",
            age=10,
        )


def test_match_row_validation_normalizes_fields() -> None:
    row = MatchRow(team_code=" bos ", team_name=" Boston Celtics ")
    assert row.team_code == "BOS"
    assert row.team_name == "Boston Celtics"
