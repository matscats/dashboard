import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import pandas as pd
from ucimlrepo import fetch_ucirepo

app = dash.Dash(__name__)
app.title = "Bank Marketing Campaign Analysis Dashboard"


@app.callback(Output("data-store", "data"), Input("data-store", "id"))
def load_data(_):
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    df = pd.concat([X, y], axis=1)

    categorical_vars = ["job", "education", "marital", "contact", "poutcome"]
    df[categorical_vars] = df[categorical_vars].fillna("Missing")
    df["was_contacted_before"] = df["pdays"].apply(lambda x: False if x == -1 else True)
    df["y_numeric"] = df["y"].map({"yes": 1, "no": 0})

    return df.to_dict("records")


app.layout = html.Div(
    [
        dcc.Store(id="data-store"),
        html.Div(
            [
                html.H1(
                    "Bank Marketing Campaign Analysis Dashboard",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "30px",
                    },
                ),
                html.P(
                    "Análise exploratória interativa de campanhas de marketing bancário para depósitos a prazo",
                    style={
                        "textAlign": "center",
                        "fontSize": "18px",
                        "color": "#7f8c8d",
                        "marginBottom": "40px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.H2("1. Análise Descritiva Geral", style={"color": "#2c3e50"}),
                html.Div(
                    [
                        html.Label(
                            "Selecione a variável categórica:",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="categorical-dropdown",
                            options=[
                                {"label": "Ocupação (job)", "value": "job"},
                                {
                                    "label": "Escolaridade (education)",
                                    "value": "education",
                                },
                                {"label": "Estado Civil (marital)", "value": "marital"},
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
                            style={"marginBottom": "20px"},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                dcc.Graph(id="categorical-frequency-chart"),
                html.H3(
                    "Distribuição de Variáveis Numéricas",
                    style={"color": "#2c3e50", "marginTop": "30px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Selecione a variável numérica:",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="numeric-dropdown",
                            options=[
                                {"label": "Idade (age)", "value": "age"},
                                {"label": "Saldo Médio (balance)", "value": "balance"},
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
                            style={"marginBottom": "20px"},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                dcc.Graph(id="numeric-distribution-chart"),
            ],
            style={"marginBottom": "40px"},
        ),
        html.Div(
            [
                html.H2(
                    "2. Relações com a Variável Alvo (y)", style={"color": "#2c3e50"}
                ),
                html.Div(
                    [
                        html.Label(
                            "Selecione a variável para análise de taxa de adesão:",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="adhesion-dropdown",
                            options=[
                                {"label": "Ocupação (job)", "value": "job"},
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
                                {"label": "Mês do Contato (month)", "value": "month"},
                            ],
                            value="job",
                            style={"marginBottom": "20px"},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                dcc.Graph(id="adhesion-rate-chart"),
                html.Div(
                    [
                        dcc.Graph(
                            id="balance-boxplot",
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        dcc.Graph(
                            id="contact-before-chart",
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                ),
            ],
            style={"marginBottom": "40px"},
        ),
        html.Div(
            [
                html.H2("3. Análises Avançadas", style={"color": "#2c3e50"}),
                html.Div(
                    [
                        dcc.Graph(
                            id="correlation-heatmap",
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        dcc.Graph(
                            id="scatter-age-balance",
                            style={"width": "50%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="violin-balance",
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        dcc.Graph(
                            id="density-age",
                            style={"width": "50%", "display": "inline-block"},
                        ),
                    ]
                ),
            ],
            style={"marginBottom": "40px"},
        ),
        html.Div(
            [
                html.H2("4. Análise de Valores Ausentes", style={"color": "#2c3e50"}),
                html.Div(
                    [
                        html.Label(
                            "Selecione a variável para análise de missing values:",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="missing-dropdown",
                            options=[
                                {"label": "Ocupação (job)", "value": "job"},
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
                            style={"marginBottom": "20px"},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                dcc.Graph(id="missing-impact-chart"),
            ]
        ),
    ],
    style={"padding": "20px", "fontFamily": "Arial, sans-serif"},
)


@app.callback(
    Output("categorical-frequency-chart", "figure"),
    [Input("categorical-dropdown", "value"), Input("data-store", "data")],
)
def update_categorical_chart(selected_var, data):
    df = pd.DataFrame(data)

    freq_data = df[selected_var].value_counts()

    fig = px.bar(
        x=freq_data.index,
        y=freq_data.values,
        title=f"Frequência das categorias em {selected_var}",
        labels={"x": selected_var, "y": "Frequência"},
        color=freq_data.values,
        color_continuous_scale="viridis",
    )

    fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)

    return fig


@app.callback(
    Output("numeric-distribution-chart", "figure"),
    [Input("numeric-dropdown", "value"), Input("data-store", "data")],
)
def update_numeric_chart(selected_var, data):
    df = pd.DataFrame(data)

    fig = px.histogram(
        df,
        x=selected_var,
        nbins=30,
        title=f"Distribuição da variável {selected_var}",
        marginal="box",
        color_discrete_sequence=["#2E86AB"],
    )

    fig.update_layout(height=500)

    return fig


@app.callback(
    Output("adhesion-rate-chart", "figure"),
    [Input("adhesion-dropdown", "value"), Input("data-store", "data")],
)
def update_adhesion_chart(selected_var, data):
    df = pd.DataFrame(data)

    adh_rate = df.groupby(selected_var)["y_numeric"].mean().sort_values(ascending=False)

    fig = px.bar(
        x=adh_rate.index,
        y=adh_rate.values,
        title=f"Taxa de Adesão por {selected_var}",
        labels={"x": selected_var, "y": "Taxa de Adesão"},
        color=adh_rate.values,
        color_continuous_scale="RdYlBu_r",
    )

    fig.update_layout(xaxis_tickangle=-45, height=500, yaxis_tickformat=".1%")

    return fig


@app.callback(Output("balance-boxplot", "figure"), Input("data-store", "data"))
def update_balance_boxplot(data):
    df = pd.DataFrame(data)

    fig = px.box(
        df,
        x="y",
        y="balance",
        title="Distribuição do Saldo Médio por Adesão ao Produto",
        color="y",
        color_discrete_map={"yes": "#2E86AB", "no": "#A23B72"},
    )

    fig.update_layout(height=400)

    return fig


@app.callback(Output("contact-before-chart", "figure"), Input("data-store", "data"))
def update_contact_before_chart(data):
    df = pd.DataFrame(data)

    adhesion_rates = (
        df.groupby("was_contacted_before")["y_numeric"].mean().reset_index()
    )

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
        height=400,
        yaxis_tickformat=".1%",
        xaxis=dict(
            tickmode="array",
            tickvals=[False, True],
            ticktext=["Não Contatado", "Já Contatado"],
        ),
    )

    return fig


@app.callback(Output("correlation-heatmap", "figure"), Input("data-store", "data"))
def update_correlation_heatmap(data):
    df = pd.DataFrame(data)

    numeric_vars = [
        "age",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "y_numeric",
    ]
    corr_matrix = df[numeric_vars].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlação das Variáveis Numéricas",
        color_continuous_scale="RdBu_r",
    )

    fig.update_layout(height=500)

    return fig


@app.callback(Output("scatter-age-balance", "figure"), Input("data-store", "data"))
def update_scatter_chart(data):
    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="age",
        y="balance",
        color="y",
        title="Relação entre Idade e Saldo por Adesão ao Produto",
        opacity=0.6,
        color_discrete_map={"yes": "#2E86AB", "no": "#A23B72"},
    )

    fig.update_layout(height=500)

    return fig


@app.callback(Output("violin-balance", "figure"), Input("data-store", "data"))
def update_violin_chart(data):
    df = pd.DataFrame(data)

    fig = px.violin(
        df,
        x="y",
        y="balance",
        title="Distribuição de Saldo por Adesão (Gráfico de Violino)",
        color="y",
        color_discrete_map={"yes": "#2E86AB", "no": "#A23B72"},
    )

    fig.update_layout(height=400)

    return fig


@app.callback(Output("density-age", "figure"), Input("data-store", "data"))
def update_density_chart(data):
    df = pd.DataFrame(data)

    fig = go.Figure()

    for category in ["yes", "no"]:
        subset = df[df["y"] == category]
        fig.add_trace(
            go.Histogram(
                x=subset["age"],
                name=f"Adesão: {category}",
                opacity=0.7,
                histnorm="probability density",
            )
        )

    fig.update_layout(
        title="Distribuição de Idade por Adesão ao Produto (Densidade)",
        xaxis_title="Idade",
        yaxis_title="Densidade",
        barmode="overlay",
        height=400,
    )

    return fig


@app.callback(
    Output("missing-impact-chart", "figure"),
    [Input("missing-dropdown", "value"), Input("data-store", "data")],
)
def update_missing_chart(selected_var, data):
    df = pd.DataFrame(data)

    ct = pd.crosstab(df[selected_var], df["y"], normalize="index") * 100
    ct = ct.sort_values(by="yes", ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(name="Não Aderiu", x=ct.index, y=ct["no"], marker_color="#A23B72")
    )

    fig.add_trace(
        go.Bar(name="Aderiu", x=ct.index, y=ct["yes"], marker_color="#2E86AB")
    )

    fig.update_layout(
        title=f"Distribuição de Adesão por Categoria em {selected_var} (incluindo Missing)",
        xaxis_title=selected_var,
        yaxis_title="Porcentagem (%)",
        barmode="stack",
        height=500,
        xaxis_tickangle=-45,
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
