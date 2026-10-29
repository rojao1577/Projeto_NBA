# NBA 2025-26: Cleveland Cavaliers e New York Knicks

Sistema web em Flask com a análise exploratória e de qualidade de uma base de jogos da temporada regular 2025-26 da
NBA. O foco são o **Cleveland Cavaliers** e o **New York Knicks**; os outros 28 times servem de comparação.

Projeto 1 · Laboratório de Ciência de Dados · UFAPE 2026.2 · Prof. Ryan Azevedo

**Site publicado:** _link a preencher após a publicação no Render_

## Como rodar localmente

Requisito: **Python 3.14** (testado com 3.14.5).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app.py
```

**Linux / macOS:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Depois de instalado, o site sobe com **um único comando** (`python app.py`, dentro do ambiente) e fica em
<http://127.0.0.1:5000>. A primeira carga leva alguns segundos: os 7 métodos de detecção de outliers são calculados uma
vez quando o site inicia.

## Seções do site

| Endereço | Conteúdo |
|---|---|
| `/` | Apresentação, os dois times principais e a grade com os 30 times |
| `/dados` | Tamanho da base, tipos das colunas e dicionário de dados |
| `/tratamento` | Conversão da data, novos nomes, colunas categóricas e exclusões para aprendizado de máquina |
| `/qualidade` | Valores especiais, correção dos percentuais, ausentes (MCAR/MAR/MNAR) e inconsistências |
| `/outliers` | Z-score, IQR, Grubbs, ECOD, K-NN, DBSCAN e LOF, coluna a coluna e em conjunto |
| `/descritiva` | Medidas descritivas, boxplot e gráficos de barras com z-score |
| `/assimetria` | Coeficiente de assimetria, histograma com KDE e boxplot |
| `/cv` | Coeficiente de variação e quadro-resumo |
| `/iqr` | Intervalo interquartil, robustez e conferência da regra de outliers |
| `/sintese` | Conclusões sobre o CLE e o NYK |
| `/time/<sigla>` | Resumo de cada time em relação à liga (ex.: `/time/CLE`) |

Nas páginas com seletores, escolha o time e/ou a variável e clique em “Ver”.

## Base de dados

`data/stats_times_regular_final.csv`: **2238 linhas × 28 colunas**, cada linha é um time em um jogo (30 times,
1119 jogos, de 21/10/2025 a 29/03/2026, 74 a 76 jogos por time). O arquivo original nunca é alterado: o tratamento
gera uma cópia em memória.

Variáveis analisadas: pontos (PTS), rebotes (REB), assistências (AST), roubos de bola (STL), tocos (BLK),
faltas (PF) e cestas de 3 convertidas (FG3M).

## Estrutura

```
projeto-nba/
├── app.py              # cria o site e as rotas (as rotas só chamam funções de analysis/)
├── config.py           # caminhos, times principais e variáveis analisadas
├── analysis/
│   ├── loader.py       # leitura do CSV
│   ├── dictionary.py   # dicionário de dados, jogos por time, siglas e logos
│   ├── cleaning.py     # tratamento: data, nomes, confronto, percentuais, exclusões
│   ├── quality.py      # valores especiais, domínio, ausentes e regras de consistência
│   ├── outliers.py     # os 7 métodos de detecção de outliers
│   ├── descriptive.py  # medidas descritivas, z-score e campanha dos times
│   ├── shape.py        # assimetria, CV e IQR
│   ├── charts.py       # gráficos Plotly e cores dos times
│   └── formatting.py   # números com 2 casas e vírgula decimal
├── templates/          # páginas HTML (Jinja2)
├── static/style.css
├── data/               # base original
├── requirements.txt
├── .python-version     # versão do Python usada na hospedagem
└── render.yaml         # configuração da hospedagem no Render
```

## Principais decisões

- **Percentuais:** muitos vieram sem o “0,” (554 em vez de 0,554). Valores acima de 1 foram divididos por 1000 e
  validados recalculando cada um a partir de convertidos ÷ tentados.
- **Convenções:** desvio padrão amostral (n − 1); quartis por interpolação linear (os mesmos na tabela, no boxplot e
  no IQR); outliers por 1,5 × IQR e |z| > 3; assimetria de Fisher-Pearson ajustada; CV em percentual; 2 casas
  decimais.
- **Outliers:** nenhum foi removido, porque são jogos reais (parte deles com prorrogação). Os métodos multivariados
  rodam na liga inteira e depois são filtrados por time.
- **z-score dos times:** (média do time − média dos 30 times) ÷ desvio padrão entre os 30 times.
- **Exclusões para aprendizado de máquina:** identificadores, colunas acumuladas (vazamento de dados), confronto e data
  bruta. A análise do site usa a base tratada completa.

## Hospedagem

O site está configurado para o [Render](https://render.com) (plano gratuito) pelo arquivo `render.yaml`:

- instalação: `pip install -r requirements.txt`
- execução: `gunicorn app:app --workers 1 --timeout 120`

Cada `git push` no ramo `main` atualiza o site publicado. No plano gratuito, o site “dorme” após cerca de 15 minutos
sem visitas, e o primeiro acesso depois disso leva até 1 minuto.

## Créditos

- Logos dos times e fotos de Donovan Mitchell e Jalen Brunson: marcas registradas e imagens da NBA e dos times,
  carregadas de cdn.nba.com e usadas apenas com fins educacionais.
- Cores dos times: [teamcolorcodes.com](https://teamcolorcodes.com/nba-team-color-codes/).
