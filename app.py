import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import pandas as pd
from ucimlrepo import fetch_ucirepo
import dash_bootstrap_components as dbc
from functools import lru_cache
import time

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "/assets/style.css"]
)
server = app.server
app.title = "Bank Marketing Campaign Analysis Dashboard"

# Modern color palette
COLORS = {
    "primary": "#3498db",
    "secondary": "#2c3e50",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# Cache global para dados
_data_cache = {}
_processed_cache = {}


def get_cached_data():
    """Cache dos dados originais do UCI"""
    if "raw_data" not in _data_cache:
        print("Carregando dados do UCI...")
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        df = pd.concat([X, y], axis=1)

        # Processamento básico
        categorical_vars = ["job", "education", "marital", "contact", "poutcome"]
        df[categorical_vars] = df[categorical_vars].fillna("Missing")
        df["was_contacted_before"] = df["pdays"].apply(
            lambda x: False if x == -1 else True
        )
        df["y_numeric"] = df["y"].map({"yes": 1, "no": 0})

        _data_cache["raw_data"] = df
        print("Dados carregados e processados!")

    return _data_cache["raw_data"]


@lru_cache(maxsize=32)
def get_frequency_data(column):
    """Cache para dados de frequência"""
    df = get_cached_data()
    return df[column].value_counts()


@lru_cache(maxsize=32)
def get_adhesion_rate(column):
    """Cache para taxa de adesão"""
    df = get_cached_data()
    return df.groupby(column)["y_numeric"].mean().sort_values(ascending=False)


@lru_cache(maxsize=8)
def get_correlation_matrix():
    """Cache para matriz de correlação"""
    df = get_cached_data()
    numeric_vars = [
        "age",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "y_numeric",
    ]
    return df[numeric_vars].corr()


@lru_cache(maxsize=16)
def get_crosstab_data(column):
    """Cache para dados de crosstab"""
    df = get_cached_data()
    ct = pd.crosstab(df[column], df["y"], normalize="index") * 100
    return ct.sort_values(by="yes", ascending=False)


@app.callback(
    Output("data-store", "data"), Input("data-store", "id"), prevent_initial_call=False
)
def load_data(_):
    """Carrega dados uma única vez"""
    df = get_cached_data()
    return df.to_dict("records")


# Layout permanece o mesmo
app.layout = dbc.Container(
    [
        dcc.Store(id="data-store"),
        # Header Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Bank Marketing Campaign Analysis Dashboard",
                                    className="header-title text-center mb-3",
                                ),
                                html.P(
                                    "Análise exploratória interativa de campanhas de marketing bancário para depósitos a prazo",
                                    className="header-subtitle text-center",
                                ),
                            ],
                            className="header-container",
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # Stats Overview Section
        dbc.Row([dbc.Col([html.Div(id="stats-overview")])], className="mb-4"),
        # Section 1: Descriptive Analysis
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            "📊 1. Análise Descritiva Geral",
                                            className="card-title mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        # Controls and Categorical Chart Row
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Selecione a variável categórica:",
                                                            className="control-label fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="categorical-dropdown",
                                                            options=[
                                                                {
                                                                    "label": "Ocupação (job)",
                                                                    "value": "job",
                                                                },
                                                                {
                                                                    "label": "Escolaridade (education)",
                                                                    "value": "education",
                                                                },
                                                                {
                                                                    "label": "Estado Civil (marital)",
                                                                    "value": "marital",
                                                                },
                                                                {
                                                                    "label": "Tipo de Contato (contact)",
                                                                    "value": "contact",
                                                                },
                                                                {
                                                                    "label": "Resultado Campanha Anterior (poutcome)",
                                                                    "value": "poutcome",
                                                                },
                                                            ],
                                                            value="job",
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="categorical-frequency-chart",
                                                            className="main-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=8,
                                                ),
                                            ],
                                            className="mb-4",
                                        ),
                                        # Numeric Variable Analysis Row
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Selecione a variável numérica:",
                                                            className="control-label fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="numeric-dropdown",
                                                            options=[
                                                                {
                                                                    "label": "Idade (age)",
                                                                    "value": "age",
                                                                },
                                                                {
                                                                    "label": "Saldo Médio (balance)",
                                                                    "value": "balance",
                                                                },
                                                                {
                                                                    "label": "Número de Contatos (campaign)",
                                                                    "value": "campaign",
                                                                },
                                                                {
                                                                    "label": "Duração do Contato (duration)",
                                                                    "value": "duration",
                                                                },
                                                            ],
                                                            value="age",
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="numeric-distribution-chart",
                                                            className="main-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=8,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="dashboard-card",
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # Section 2: Target Variable Analysis
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            "🎯 2. Relações com a Variável Alvo (y)",
                                            className="card-title mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        # Control and Adhesion Rate Chart
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Selecione a variável para análise de taxa de adesão:",
                                                            className="control-label fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="adhesion-dropdown",
                                                            options=[
                                                                {
                                                                    "label": "Ocupação (job)",
                                                                    "value": "job",
                                                                },
                                                                {
                                                                    "label": "Escolaridade (education)",
                                                                    "value": "education",
                                                                },
                                                                {
                                                                    "label": "Tipo de Contato (contact)",
                                                                    "value": "contact",
                                                                },
                                                                {
                                                                    "label": "Resultado Campanha Anterior (poutcome)",
                                                                    "value": "poutcome",
                                                                },
                                                                {
                                                                    "label": "Mês do Contato (month)",
                                                                    "value": "month",
                                                                },
                                                            ],
                                                            value="job",
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="adhesion-rate-chart",
                                                            className="main-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=8,
                                                ),
                                            ],
                                            className="mb-4",
                                        ),
                                        # Two Charts Side by Side
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="balance-boxplot",
                                                            className="side-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="contact-before-chart",
                                                            className="side-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="dashboard-card",
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # Section 3: Advanced Analysis
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            "🔬 3. Análises Avançadas",
                                            className="card-title mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        # First Row of Charts
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="correlation-heatmap",
                                                            className="advanced-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="scatter-age-balance",
                                                            className="advanced-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        # Second Row of Charts
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="violin-balance",
                                                            className="advanced-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="density-age",
                                                            className="advanced-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="dashboard-card",
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # Section 4: Missing Values Analysis
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            "🔍 4. Análise de Valores Ausentes",
                                            className="card-title mb-0",
                                        )
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Selecione a variável para análise de missing values:",
                                                            className="control-label fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="missing-dropdown",
                                                            options=[
                                                                {
                                                                    "label": "Ocupação (job)",
                                                                    "value": "job",
                                                                },
                                                                {
                                                                    "label": "Escolaridade (education)",
                                                                    "value": "education",
                                                                },
                                                                {
                                                                    "label": "Tipo de Contato (contact)",
                                                                    "value": "contact",
                                                                },
                                                                {
                                                                    "label": "Resultado Campanha Anterior (poutcome)",
                                                                    "value": "poutcome",
                                                                },
                                                            ],
                                                            value="job",
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="missing-impact-chart",
                                                            className="main-chart",
                                                        )
                                                    ],
                                                    width=12,
                                                    lg=8,
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            className="dashboard-card",
                        )
                    ]
                )
            ]
        ),
    ],
    fluid=True,
    className="px-4 py-3",
)


# Callbacks otimizados
@app.callback(
    Output("stats-overview", "children"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_stats_overview(data):
    """Stats overview otimizado usando cache"""
    df = get_cached_data()

    total_records = len(df)
    adhesion_rate = df["y_numeric"].mean() * 100
    avg_age = df["age"].mean()
    avg_balance = df["balance"].mean()

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H2(
                                        f"{total_records:,}",
                                        className="stat-value text-primary mb-1",
                                    ),
                                    html.P(
                                        "Total de Registros",
                                        className="stat-label text-muted mb-0",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="stat-card",
                    )
                ],
                width=12,
                sm=6,
                lg=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H2(
                                        f"{adhesion_rate:.1f}%",
                                        className="stat-value text-success mb-1",
                                    ),
                                    html.P(
                                        "Taxa de Adesão",
                                        className="stat-label text-muted mb-0",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="stat-card",
                    )
                ],
                width=12,
                sm=6,
                lg=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H2(
                                        f"{avg_age:.0f}",
                                        className="stat-value text-info mb-1",
                                    ),
                                    html.P(
                                        "Idade Média",
                                        className="stat-label text-muted mb-0",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="stat-card",
                    )
                ],
                width=12,
                sm=6,
                lg=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H2(
                                        f"€{avg_balance:,.0f}",
                                        className="stat-value text-warning mb-1",
                                    ),
                                    html.P(
                                        "Saldo Médio",
                                        className="stat-label text-muted mb-0",
                                    ),
                                ],
                                className="text-center",
                            )
                        ],
                        className="stat-card",
                    )
                ],
                width=12,
                sm=6,
                lg=3,
            ),
        ],
        className="g-3",
    )


@app.callback(
    Output("categorical-frequency-chart", "figure"),
    Input("categorical-dropdown", "value"),
    prevent_initial_call=True,
)
def update_categorical_chart(selected_var):
    """Gráfico categórico otimizado"""
    freq_data = get_frequency_data(selected_var)

    fig = px.bar(
        x=freq_data.index,
        y=freq_data.values,
        title=f"Frequência das categorias em {selected_var}",
        labels={"x": selected_var, "y": "Frequência"},
        color=freq_data.values,
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=16,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision=selected_var,
        transition_duration=300,
    )

    return fig


@app.callback(
    Output("numeric-distribution-chart", "figure"),
    Input("numeric-dropdown", "value"),
    prevent_initial_call=True,
)
def update_numeric_chart(selected_var):
    """Gráfico numérico otimizado"""
    df = get_cached_data()

    fig = px.histogram(
        df,
        x=selected_var,
        nbins=30,
        title=f"Distribuição da variável {selected_var}",
        marginal="box",
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_layout(
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=16,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision=selected_var,
        transition_duration=300,
    )

    return fig


@app.callback(
    Output("adhesion-rate-chart", "figure"),
    Input("adhesion-dropdown", "value"),
    prevent_initial_call=True,
)
def update_adhesion_chart(selected_var):
    """Gráfico de taxa de adesão otimizado"""
    adh_rate = get_adhesion_rate(selected_var)

    fig = px.bar(
        x=adh_rate.index,
        y=adh_rate.values,
        title=f"Taxa de Adesão por {selected_var}",
        labels={"x": selected_var, "y": "Taxa de Adesão"},
        color=adh_rate.values,
        color_continuous_scale="RdYlBu_r",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        yaxis_tickformat=".1%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=16,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision=selected_var,
        transition_duration=300,
    )

    return fig


@app.callback(
    Output("balance-boxplot", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_balance_boxplot(data):
    """Boxplot otimizado"""
    df = get_cached_data()

    fig = px.box(
        df,
        x="y",
        y="balance",
        title="Distribuição do Saldo Médio por Adesão ao Produto",
        color="y",
        color_discrete_map={"yes": COLORS["success"], "no": COLORS["danger"]},
    )

    fig.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=14,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision="balance-boxplot",
    )

    return fig


@app.callback(
    Output("contact-before-chart", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_contact_before_chart(data):
    """Gráfico de contato anterior otimizado"""
    df = get_cached_data()

    # Cache do cálculo
    if "contact_before_rates" not in _processed_cache:
        _processed_cache["contact_before_rates"] = (
            df.groupby("was_contacted_before")["y_numeric"].mean().reset_index()
        )

    adhesion_rates = _processed_cache["contact_before_rates"]

    fig = px.bar(
        adhesion_rates,
        x="was_contacted_before",
        y="y_numeric",
        title="Taxa de Adesão por Contato Anterior",
        labels={
            "was_contacted_before": "Já foi contatado anteriormente?",
            "y_numeric": "Taxa de Adesão",
        },
        color="y_numeric",
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        height=320,
        yaxis_tickformat=".1%",
        xaxis=dict(
            tickmode="array",
            tickvals=[False, True],
            ticktext=["Não Contatado", "Já Contatado"],
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=14,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision="contact-before",
    )

    return fig


@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_correlation_heatmap(data):
    """Heatmap de correlação otimizado"""
    corr_matrix = get_correlation_matrix()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlação das Variáveis Numéricas",
        color_continuous_scale="RdBu_r",
    )

    fig.update_layout(
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=16,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision="correlation-heatmap",
    )

    return fig


@app.callback(
    Output("scatter-age-balance", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_scatter_chart(data):
    """Scatter plot otimizado com sampling"""
    df = get_cached_data()

    # Sampling para melhor performance em scatter plots grandes
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=42)
    else:
        df_sample = df

    fig = px.scatter(
        df_sample,
        x="age",
        y="balance",
        color="y",
        title="Relação entre Idade e Saldo por Adesão ao Produto",
        opacity=0.6,
        color_discrete_map={"yes": COLORS["success"], "no": COLORS["danger"]},
    )

    fig.update_layout(
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=16,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision="scatter-age-balance",
    )

    return fig


@app.callback(
    Output("violin-balance", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_violin_chart(data):
    """Gráfico de violino otimizado"""
    df = get_cached_data()

    fig = px.violin(
        df,
        x="y",
        y="balance",
        title="Distribuição de Saldo por Adesão (Gráfico de Violino)",
        color="y",
        color_discrete_map={"yes": COLORS["success"], "no": COLORS["danger"]},
    )

    fig.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=14,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision="violin-balance",
    )

    return fig


@app.callback(
    Output("density-age", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_density_chart(data):
    """Gráfico de densidade otimizado"""
    df = get_cached_data()

    fig = go.Figure()
    colors = {"yes": COLORS["success"], "no": COLORS["danger"]}

    for category in ["yes", "no"]:
        subset = df[df["y"] == category]
        fig.add_trace(
            go.Histogram(
                x=subset["age"],
                name=f"Adesão: {category}",
                opacity=0.7,
                histnorm="probability density",
                marker_color=colors[category],
                nbinsx=25,  # Reduzir bins para melhor performance
            )
        )

    fig.update_layout(
        title="Distribuição de Idade por Adesão ao Produto (Densidade)",
        xaxis_title="Idade",
        yaxis_title="Densidade",
        barmode="overlay",
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=14,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision="density-age",
    )

    return fig


@app.callback(
    Output("missing-impact-chart", "figure"),
    Input("missing-dropdown", "value"),
    prevent_initial_call=True,
)
def update_missing_chart(selected_var):
    """Gráfico de missing values otimizado"""
    ct = get_crosstab_data(selected_var)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(name="Não Aderiu", x=ct.index, y=ct["no"], marker_color=COLORS["danger"])
    )

    fig.add_trace(
        go.Bar(name="Aderiu", x=ct.index, y=ct["yes"], marker_color=COLORS["success"])
    )

    fig.update_layout(
        title=f"Distribuição de Adesão por Categoria em {selected_var} (incluindo Missing)",
        xaxis_title=selected_var,
        yaxis_title="Porcentagem (%)",
        barmode="stack",
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=16,
        title_font_color=COLORS["secondary"],
        # Otimizações de performance
        uirevision=selected_var,
        transition_duration=300,
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
