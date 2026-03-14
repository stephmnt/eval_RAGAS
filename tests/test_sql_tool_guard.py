from sql_tool import _is_safe_sql


def test_safe_sql_accepts_select_queries() -> None:
    assert _is_safe_sql("SELECT player_name FROM players LIMIT 5;")
    assert _is_safe_sql("WITH t AS (SELECT 1) SELECT * FROM t;")


def test_safe_sql_rejects_mutation_queries() -> None:
    assert not _is_safe_sql("DELETE FROM players;")
    assert not _is_safe_sql("UPDATE players SET team_code='OKC';")
    assert not _is_safe_sql("SELECT * FROM players; DROP TABLE players;")
