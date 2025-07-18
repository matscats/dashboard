import streamlit as st
import plotly.express as px
import pandas as pd
from ucimlrepo import fetch_ucirepo
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- Configuração da Página e Estilos ---
st.set_page_config(
    page_title="Bank Marketing Campaign Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de cores
COLORS = {
    "primary": "#27ae60", # Verde para combinar com Random Forest
    "secondary": "#2c3e50",
    "success": "#2ecc71",
    "danger": "#e74c3c",
}

# CSS customizado
st.markdown(
    """
<style>
    .main-header { text-align: center; color: #2c3e50; margin-bottom: 2rem; }
    .stat-card { background: linear-gradient(135deg, #27ae60 0%, #2c3e50 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem; }
    .stat-value { font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; }
    .stat-label { font-size: 0.9rem; opacity: 0.8; }
    .section-header { color: #2c3e50; border-bottom: 2px solid #27ae60; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0; }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Funções de Cache e Carregamento de Dados ---
@st.cache_data
def get_data():
    """Carrega e pré-processa os dados originais do UCI."""
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    df = pd.concat([X, y], axis=1)
    df["y_numeric"] = df["y"].map({"yes": 1, "no": 0})
    df.drop(columns=['contact', 'poutcome'], inplace=True)
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_and_predict(_df):
    """Treina o modelo e adiciona as previsões ao dataframe original."""
    X = _df.drop(["y", "y_numeric"], axis=1)
    y = _df["y_numeric"]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)
    
    # Adiciona as previsões ao dataframe original para a análise de perfil
    _df['prediction'] = model_pipeline.predict(X)
    
    return model_pipeline, X_train, X_test, y_train, y_test, _df

# Carrega os dados e treina o modelo
df_original = get_data()
model_pipeline, X_train, X_test, y_train, y_test, df_with_predictions = train_and_predict(df_original.copy())


# --- UI da Barra Lateral ---
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione uma página:", ["Análise Exploratória", "Simulação Interativa"])
st.sidebar.markdown("---")

# --- Lógica da Página ---
if page == "Análise Exploratória":
    st.markdown('<h1 class="main-header">📊 Análise de Campanha de Marketing Bancário</h1>', unsafe_allow_html=True)

    # --- SEÇÃO DE FILTRO DE PERFIL ---
    st.markdown('<h2 class="section-header">🔍 Análise de Perfil de Cliente</h2>', unsafe_allow_html=True)
    profile_filter = st.selectbox(
        "Filtrar Perfil de Cliente:",
        options=["Todos os Clientes", "Clientes com Potencial de Adesão", "Clientes com Baixo Potencial"]
    )

    if profile_filter == "Clientes com Potencial de Adesão":
        df_display = df_with_predictions[df_with_predictions['prediction'] == 1]
    elif profile_filter == "Clientes com Baixo Potencial":
        df_display = df_with_predictions[df_with_predictions['prediction'] == 0]
    else:
        df_display = df_with_predictions

    st.markdown(f"Exibindo dados para **{len(df_display)}** clientes.")

    # Visão Geral (baseada no df_display filtrado)
    st.markdown('<h2 class="section-header">📈 Visão Geral dos Dados</h2>', unsafe_allow_html=True)
    total_records, adhesion_rate, avg_age, avg_balance = len(df_display), df_display["y_numeric"].mean() * 100, df_display["age"].mean(), df_display["balance"].mean()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{total_records:,}</div><div class="stat-label">Total de Registros</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{adhesion_rate:.1f}%</div><div class="stat-label">Taxa de Adesão</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg_age:.0f}</div><div class="stat-label">Idade Média</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value">€{avg_balance:,.0f}</div><div class="stat-label">Saldo Médio</div></div>', unsafe_allow_html=True)

    # Gráficos (baseados no df_display filtrado)
    st.markdown('<h2 class="section-header">🎨 Visualizações de Distribuição</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        var_cat = st.selectbox("Análise por Variável Categórica:", options=["job", "education", "marital", "housing", "loan", "month"])
        fig_cat = px.bar(df_display[var_cat].value_counts(), title=f"Distribuição por {var_cat}", color_discrete_sequence=[COLORS["primary"]])
        st.plotly_chart(fig_cat, use_container_width=True)
    with c2:
        var_num = st.selectbox("Análise por Variável Numérica:", options=["age", "balance", "duration", "campaign"])
        fig_num = px.histogram(df_display, x=var_num, marginal="box", title=f"Distribuição de {var_num}", color_discrete_sequence=[COLORS["success"]])
        st.plotly_chart(fig_num, use_container_width=True)

    st.markdown('<h2 class="section-header">🎯 Análise da Variável Alvo e Correlações</h2>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Contagem de Adesão (Sim/Não)")
        fig_countplot = plt.figure(figsize=(8, 6))
        sns.countplot(x='y', data=df_display, palette=[COLORS["danger"], COLORS["success"]])
        plt.xlabel("Adesão ao Depósito a Prazo")
        plt.ylabel("Contagem")
        st.pyplot(fig_countplot)
    with c4:
        st.subheader("Proporção de Adesão (Sim/Não)")
        fig_pie, ax = plt.subplots()
        pie_data = df_display['y'].value_counts()
        if not pie_data.empty:
            labels = list(pie_data.index)
            ax.pie(pie_data, labels=labels, autopct='%.2f%%', colors=[COLORS["primary"], COLORS["success"]], explode=[0.1] * min(2, len(pie_data)))
        st.pyplot(fig_pie)

    st.subheader("Mapa de Calor de Correlações")
    numeric_df = df_display.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig_heatmap = plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig_heatmap)

elif page == "Simulação Interativa":
    st.markdown('<h1 class="main-header">🤖 Simulação Interativa com Perfis de Exemplo</h1>', unsafe_allow_html=True)

    @st.cache_data
    def get_personas(_model, _X_test, _y_test):
        results_df = _X_test.copy()
        results_df['y_real'] = _y_test
        results_df['y_predito'] = _model.predict(_X_test)
        results_df['prob_adesao'] = _model.predict_proba(_X_test)[:, 1]
        
        maior_potencial = results_df.loc[results_df[(results_df['y_real'] == 1) & (results_df['y_predito'] == 1)]['prob_adesao'].idxmax()]
        menor_potencial = results_df.loc[results_df[(results_df['y_real'] == 0) & (results_df['y_predito'] == 0)]['prob_adesao'].idxmin()]
        results_df['dist_do_meio'] = abs(results_df['prob_adesao'] - 0.5)
        indeciso = results_df.loc[results_df['dist_do_meio'].idxmin()]
        
        return {"maior": maior_potencial, "menor": menor_potencial, "indeciso": indeciso}

    personas = get_personas(model_pipeline, X_test, y_test)
    
    st.sidebar.header("Parâmetros do Cliente")
    st.sidebar.subheader("Carregar Perfis de Exemplo:")
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("Ideal", help="Carrega o perfil de um cliente real com a maior probabilidade de adesão."):
        st.session_state.persona = personas["maior"]
    if col2.button("Ruim", help="Carrega o perfil de um cliente real com a menor probabilidade de adesão."):
        st.session_state.persona = personas["menor"]
    if col3.button("Indeciso", help="Carrega o perfil de um cliente real sobre o qual o modelo está mais em dúvida."):
        st.session_state.persona = personas["indeciso"]
    
    if st.sidebar.button("Limpar Campos", type="primary"):
        st.session_state.persona = {}

    def user_input_features():
        p = st.session_state.get("persona", {})
        
        age = st.sidebar.slider('Idade', 18, 95, int(p.get("age", 40)))
        
        job_options = sorted(df_original['job'].unique())
        job_index = job_options.index(p.get("job", "admin.")) if p.get("job") in job_options else 0
        job = st.sidebar.selectbox('Ocupação', job_options, index=job_index)
        
        marital_options = sorted(df_original['marital'].unique())
        marital_index = marital_options.index(p.get("marital", "married")) if p.get("marital") in marital_options else 0
        marital = st.sidebar.selectbox('Estado Civil', marital_options, index=marital_index)

        education_options = sorted(df_original['education'].unique())
        education_index = education_options.index(p.get("education", "secondary")) if p.get("education") in education_options else 0
        education = st.sidebar.selectbox('Escolaridade', education_options, index=education_index)
        
        default_options = ('no', 'yes')
        default_index = default_options.index(p.get("default", "no")) if p.get("default") in default_options else 0
        default = st.sidebar.selectbox('Inadimplência?', default_options, index=default_index)

        balance = st.sidebar.number_input('Saldo (em €)', value=int(p.get("balance", 1000)))

        housing_options = ('no', 'yes')
        housing_index = housing_options.index(p.get("housing", "no")) if p.get("housing") in housing_options else 0
        housing = st.sidebar.selectbox('Empréstimo Imobiliário?', housing_options, index=housing_index)
        
        loan_options = ('no', 'yes')
        loan_index = loan_options.index(p.get("loan", "no")) if p.get("loan") in loan_options else 0
        loan = st.sidebar.selectbox('Empréstimo Pessoal?', loan_options, index=loan_index)

        day_of_week = st.sidebar.slider('Dia do Contato', 1, 31, int(p.get("day_of_week", 15)))
        
        month_options = sorted(df_original['month'].unique(), key=lambda m: list(df_original['month'].unique()).index(m))
        month_index = month_options.index(p.get("month", "may")) if p.get("month") in month_options else 0
        month = st.sidebar.select_slider('Mês do Contato', options=month_options, value=month_options[month_index])

        duration = st.sidebar.slider('Duração da Ligação (segundos)', 0, 5000, int(p.get("duration", 300)))
        campaign = st.sidebar.slider('Nº de Contatos na Campanha', 1, 63, int(p.get("campaign", 2)))
        pdays = st.sidebar.slider('Dias Desde Último Contato (pdays)', -1, 871, int(p.get("pdays", -1)))
        previous = st.sidebar.slider('Nº de Contatos Anteriores', 0, 275, int(p.get("previous", 0)))
        
        data = {
            'age': age, 'job': job, 'marital': marital, 'education': education, 
            'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
            'day_of_week': day_of_week, 'month': month, 'duration': duration, 
            'campaign': campaign, 'pdays': pdays, 'previous': previous
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    if st.sidebar.button('Executar Previsão'):
        prediction = model_pipeline.predict(input_df)[0]
        prediction_proba = model_pipeline.predict_proba(input_df)[0][1]

        st.subheader("Resultado da Previsão")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success("✅ **Previsão:** Cliente propenso a aderir ao depósito.")
            else:
                st.error("❌ **Previsão:** Cliente não propenso a aderir ao depósito.")
        with col2:
            st.info(f"💡 **Probabilidade de Adesão:** {prediction_proba*100:.2f}%")

    st.markdown('<h2 class="section-header">⚙️ Painel de Performance do Modelo Random Forest</h2>', unsafe_allow_html=True)
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, xticklabels=['Não Aderiu', 'Aderiu'], yticklabels=['Não Aderiu', 'Aderiu'])
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        st.pyplot(fig_cm)

    with c2:
        st.subheader("Importância das Features")
        try:
            ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = list(ohe.get_feature_names_out(X_train.select_dtypes(include=['object']).columns))
            num_feature_names = list(X_train.select_dtypes(include=['int64', 'float64']).columns)
            all_feature_names = num_feature_names + cat_feature_names
            importances = model_pipeline.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Importância': importances}, index=all_feature_names).sort_values(by='Importância', ascending=False)
            top_features = feature_importance_df.head(10)
            fig_importance = px.bar(top_features, x='Importância', y=top_features.index, orientation='h', title='Top 10 Features Mais Influentes')
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico de importância das features. Erro: {e}")

    with c3:
        st.subheader("Curva ROC e AUC")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:0.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim((0.0, 1.0))
        ax_roc.set_ylim((0.0, 1.05))
        ax_roc.set_xlabel('Taxa de Falsos Positivos')
        ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
        ax_roc.set_title('Característica de Operação do Receptor')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    with c4:
        st.subheader("Distribuição das Probabilidades")
        proba_df = pd.DataFrame({'Probabilidade': y_pred_proba, 'Classe Real': y_test.map({0: 'Não Aderiu', 1: 'Aderiu'})})
        fig_proba = px.histogram(proba_df, x='Probabilidade', color='Classe Real', marginal="box", nbins=50, title="Distribuição das Probabilidades por Classe", color_discrete_map={'Não Aderiu': COLORS['danger'], 'Aderiu': COLORS['success']})
        st.plotly_chart(fig_proba, use_container_width=True)