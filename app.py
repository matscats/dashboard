import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from ucimlrepo import fetch_ucirepo
from functools import lru_cache
import time

# Configuração da página
st.set_page_config(
    page_title="Bank Marketing Campaign Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de cores moderna
COLORS = {
    "primary": "#3498db",
    "secondary": "#2c3e50",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# CSS customizado
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cache global para dados
@st.cache_data
def get_cached_data():
    """Cache dos dados originais do UCI"""
    print("Carregando dados do UCI...")
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    df = pd.concat([X, y], axis=1)

    # Processamento básico
    categorical_vars = ["job", "education", "marital", "contact", "poutcome"]
    df[categorical_vars] = df[categorical_vars].fillna("Missing")
    df["was_contacted_before"] = df["pdays"].apply(lambda x: False if x == -1 else True)
    df["y_numeric"] = df["y"].map({"yes": 1, "no": 0})

    print("Dados carregados e processados!")
    return df


@st.cache_data
def get_frequency_data(column):
    """Cache para dados de frequência"""
    df = get_cached_data()
    return df[column].value_counts()


@st.cache_data
def get_adhesion_rate(column):
    """Cache para taxa de adesão"""
    df = get_cached_data()
    return df.groupby(column)["y_numeric"].mean().sort_values(ascending=False)


@st.cache_data
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


@st.cache_data
def get_crosstab_data(column):
    """Cache para dados de crosstab"""
    df = get_cached_data()
    ct = pd.crosstab(df[column], df["y"], normalize="index") * 100
    return ct.sort_values(by="yes", ascending=False)


# Header Principal
st.markdown(
    """
<div class="main-header">
    <h1>📊 Bank Marketing Campaign Analysis Dashboard</h1>
    <p>Análise exploratória interativa de campanhas de marketing bancário para depósitos a prazo</p>
</div>
""",
    unsafe_allow_html=True,
)

# Carregamento dos dados
df = get_cached_data()

# Stats Overview
st.markdown(
    '<h2 class="section-header">📈 Visão Geral dos Dados</h2>', unsafe_allow_html=True
)

total_records = len(df)
adhesion_rate = df["y_numeric"].mean() * 100
avg_age = df["age"].mean()
avg_balance = df["balance"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div class="stat-card">
        <div class="stat-value">{total_records:,}</div>
        <div class="stat-label">Total de Registros</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="stat-card">
        <div class="stat-value">{adhesion_rate:.1f}%</div>
        <div class="stat-label">Taxa de Adesão</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="stat-card">
        <div class="stat-value">{avg_age:.0f}</div>
        <div class="stat-label">Idade Média</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div class="stat-card">
        <div class="stat-value">€{avg_balance:,.0f}</div>
        <div class="stat-label">Saldo Médio</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Seção 1: Análise Descritiva Geral
st.markdown(
    '<h2 class="section-header">📊 1. Análise Descritiva Geral</h2>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Variável Categórica")
    categorical_var = st.selectbox(
        "Selecione a variável categórica:",
        options=["job", "education", "marital", "contact", "poutcome"],
        format_func=lambda x: {
            "job": "Ocupação (job)",
            "education": "Escolaridade (education)",
            "marital": "Estado Civil (marital)",
            "contact": "Tipo de Contato (contact)",
            "poutcome": "Resultado Campanha Anterior (poutcome)",
        }[x],
    )

with col2:
    freq_data = get_frequency_data(categorical_var)
    fig_cat = px.bar(
        x=freq_data.index,
        y=freq_data.values,
        title=f"Frequência das categorias em {categorical_var}",
        labels={"x": categorical_var, "y": "Frequência"},
        color=freq_data.values,
        color_continuous_scale="Blues",
    )
    fig_cat.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# Análise de variáveis numéricas
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Variável Numérica")
    numeric_var = st.selectbox(
        "Selecione a variável numérica:",
        options=["age", "balance", "campaign", "duration"],
        format_func=lambda x: {
            "age": "Idade (age)",
            "balance": "Saldo Médio (balance)",
            "campaign": "Número de Contatos (campaign)",
            "duration": "Duração do Contato (duration)",
        }[x],
    )

with col2:
    fig_num = px.histogram(
        df,
        x=numeric_var,
        nbins=30,
        title=f"Distribuição da variável {numeric_var}",
        marginal="box",
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig_num.update_layout(
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_num, use_container_width=True)

# Seção 2: Relações com a Variável Alvo
st.markdown(
    '<h2 class="section-header">🎯 2. Relações com a Variável Alvo (y)</h2>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Taxa de Adesão")
    adhesion_var = st.selectbox(
        "Selecione a variável para análise de taxa de adesão:",
        options=["job", "education", "contact", "poutcome", "month"],
        format_func=lambda x: {
            "job": "Ocupação (job)",
            "education": "Escolaridade (education)",
            "contact": "Tipo de Contato (contact)",
            "poutcome": "Resultado Campanha Anterior (poutcome)",
            "month": "Mês do Contato (month)",
        }[x],
    )

with col2:
    adh_rate = get_adhesion_rate(adhesion_var)
    fig_adh = px.bar(
        x=adh_rate.index,
        y=adh_rate.values,
        title=f"Taxa de Adesão por {adhesion_var}",
        labels={"x": adhesion_var, "y": "Taxa de Adesão"},
        color=adh_rate.values,
        color_continuous_scale="RdYlBu_r",
    )
    fig_adh.update_layout(
        xaxis_tickangle=-45,
        height=400,
        yaxis_tickformat=".1%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_adh, use_container_width=True)

# Gráficos lado a lado
col1, col2 = st.columns(2)

with col1:
    fig_box = px.box(
        df,
        x="y",
        y="balance",
        title="Distribuição do Saldo Médio por Adesão ao Produto",
        color="y",
        color_discrete_map={"yes": COLORS["success"], "no": COLORS["danger"]},
    )
    fig_box.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    adhesion_rates = (
        df.groupby("was_contacted_before")["y_numeric"].mean().reset_index()
    )
    fig_contact = px.bar(
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
    fig_contact.update_layout(
        height=320,
        yaxis_tickformat=".1%",
        xaxis=dict(
            tickmode="array",
            tickvals=[False, True],
            ticktext=["Não Contatado", "Já Contatado"],
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_contact, use_container_width=True)

# Seção 3: Análises Avançadas
st.markdown(
    '<h2 class="section-header">🔬 3. Análises Avançadas</h2>', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    corr_matrix = get_correlation_matrix()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlação das Variáveis Numéricas",
        color_continuous_scale="RdBu_r",
    )
    fig_corr.update_layout(
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col2:
    # Sampling para melhor performance em scatter plots grandes
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=42)
    else:
        df_sample = df

    fig_scatter = px.scatter(
        df_sample,
        x="age",
        y="balance",
        color="y",
        title="Relação entre Idade e Saldo por Adesão ao Produto",
        opacity=0.6,
        color_discrete_map={"yes": COLORS["success"], "no": COLORS["danger"]},
    )
    fig_scatter.update_layout(
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_violin = px.violin(
        df,
        x="y",
        y="balance",
        title="Distribuição de Saldo por Adesão (Gráfico de Violino)",
        color="y",
        color_discrete_map={"yes": COLORS["success"], "no": COLORS["danger"]},
    )
    fig_violin.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_violin, use_container_width=True)

with col2:
    fig_density = go.Figure()
    colors = {"yes": COLORS["success"], "no": COLORS["danger"]}

    for category in ["yes", "no"]:
        subset = df[df["y"] == category]
        fig_density.add_trace(
            go.Histogram(
                x=subset["age"],
                name=f"Adesão: {category}",
                opacity=0.7,
                histnorm="probability density",
                marker_color=colors[category],
                nbinsx=25,
            )
        )

    fig_density.update_layout(
        title="Distribuição de Idade por Adesão ao Produto (Densidade)",
        xaxis_title="Idade",
        yaxis_title="Densidade",
        barmode="overlay",
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_density, use_container_width=True)

# Seção 4: Análise de Valores Ausentes
st.markdown(
    '<h2 class="section-header">🔍 4. Análise de Valores Ausentes</h2>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Missing Values")
    missing_var = st.selectbox(
        "Selecione a variável para análise de missing values:",
        options=["job", "education", "contact", "poutcome"],
        format_func=lambda x: {
            "job": "Ocupação (job)",
            "education": "Escolaridade (education)",
            "contact": "Tipo de Contato (contact)",
            "poutcome": "Resultado Campanha Anterior (poutcome)",
        }[x],
    )

with col2:
    ct = get_crosstab_data(missing_var)

    fig_missing = go.Figure()
    fig_missing.add_trace(
        go.Bar(name="Não Aderiu", x=ct.index, y=ct["no"], marker_color=COLORS["danger"])
    )
    fig_missing.add_trace(
        go.Bar(name="Aderiu", x=ct.index, y=ct["yes"], marker_color=COLORS["success"])
    )

    fig_missing.update_layout(
        title=f"Distribuição de Adesão por Categoria em {missing_var} (incluindo Missing)",
        xaxis_title=missing_var,
        yaxis_title="Porcentagem (%)",
        barmode="stack",
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_missing, use_container_width=True)

# Informações adicionais na sidebar
st.sidebar.markdown("## 📋 Informações do Dataset")
st.sidebar.info(
    f"""
**Total de registros:** {len(df):,}  
**Variáveis:** {len(df.columns)}  
**Taxa de adesão geral:** {df['y_numeric'].mean()*100:.1f}%  
**Período:** Campanhas de marketing bancário  
**Fonte:** UCI Machine Learning Repository
"""
)

st.sidebar.markdown("## 🎯 Principais Insights")
st.sidebar.success(
    """
- Análise completa de campanhas de marketing
- Identificação de fatores de sucesso
- Segmentação de clientes
- Otimização de estratégias
"""
)
