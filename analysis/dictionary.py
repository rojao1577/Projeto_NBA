import pandas as pd


COLUMNS = [
    
    ("Team_ID", "id_time", "Identificador numérico do time", "categórica"),
    ("Game_ID", "id_jogo", "Identificador do jogo (aparece 2 vezes: uma por time)", "categórica"),
    ("GAME_DATE", "data_jogo", "Data do jogo", "data"),
    ("MATCHUP", "confronto", "Confronto: 'vs.' = jogo em casa, '@' = jogo fora", "texto"),
    ("WL", "vitoria", "Resultado do jogo (1 = vitória, 0 = derrota)", "binária"),
    ("W", "vitorias_acumuladas", "Vitórias acumuladas na temporada até este jogo", "numérica"),
    ("L", "derrotas_acumuladas", "Derrotas acumuladas na temporada até este jogo", "numérica"),
    ("W_PCT", "aproveitamento_acumulado", "Aproveitamento acumulado: W / (W + L)", "numérica"),
    ("MIN", "minutos", "Minutos jogados pelo time (240 = normal; 265/290 = prorrogação)", "numérica"),
    ("FGM", "arremessos_convertidos", "Arremessos de quadra convertidos", "numérica"),
    ("FGA", "arremessos_tentados", "Arremessos de quadra tentados", "numérica"),
    ("FG_PCT", "aproveitamento_arremessos", "Aproveitamento nos arremessos de quadra: FGM / FGA", "numérica"),
    ("FG3M", "cestas_3_convertidas", "Cestas de 3 pontos convertidas", "numérica"),
    ("FG3A", "cestas_3_tentadas", "Cestas de 3 pontos tentadas", "numérica"),
    ("FG3_PCT", "aproveitamento_3pts", "Aproveitamento nas cestas de 3: FG3M / FG3A", "numérica"),
    ("FTM", "lances_livres_convertidos", "Lances livres convertidos", "numérica"),
    ("FTA", "lances_livres_tentados", "Lances livres tentados", "numérica"),
    ("FT_PCT", "aproveitamento_lances_livres", "Aproveitamento nos lances livres: FTM / FTA", "numérica"),
    ("OREB", "rebotes_ofensivos", "Rebotes ofensivos", "numérica"),
    ("DREB", "rebotes_defensivos", "Rebotes defensivos", "numérica"),
    ("REB", "rebotes", "Rebotes totais (OREB + DREB)", "numérica"),
    ("AST", "assistencias", "Assistências", "numérica"),
    ("STL", "roubos", "Roubos de bola", "numérica"),
    ("BLK", "tocos", "Tocos", "numérica"),
    ("TOV", "erros", "Erros (turnovers)", "numérica"),
    ("PF", "faltas", "Faltas pessoais", "numérica"),
    ("PTS", "pontos", "Pontos marcados", "numérica"),
    ("TEAM_NAME", "time", "Nome do time", "categórica"),
]


RENAME_MAP = {coluna: nome_novo for coluna, nome_novo, _, _ in COLUMNS}


def overview(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    return {"linhas": n_rows, "colunas": n_cols}


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(COLUMNS, columns=["coluna", "nome_novo", "significado", "tipo"])
    table["tipo_pandas"] = table["coluna"].map(df.dtypes.astype(str))
    return table
