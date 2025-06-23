import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro, norm, probplot, binom

def carregar_dados():
    sex_offenders = pd.read_csv('datas/Sex_Offenders.csv')
    crimes = pd.read_csv('datas/Crimes_2001_to_Present_Cut_Version.csv', low_memory=False)
    # ---- TRATAMENTO DOS DADOS ---- #
    sex_offenders['GENDER'] = sex_offenders['GENDER'].astype('category')
    sex_offenders['RACE'] = sex_offenders['RACE'].astype('category')
    sex_offenders['BIRTH DATE'] = pd.to_datetime(sex_offenders['BIRTH DATE'], format='%m/%d/%Y', errors='coerce')
    sex_offenders['VICTIM MINOR'] = sex_offenders['VICTIM MINOR'].astype('category')
    sex_offenders['BLOCK'] = sex_offenders['BLOCK'].astype('category')
    crimes['Date'] = pd.to_datetime(crimes['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    crimes['Primary Type'] = crimes['Primary Type'].astype('category')
    crimes['Block'] = crimes['Block'].astype('category')
    crimes['IUCR'] = crimes['IUCR'].astype('category')
    crimes['Location Description'] = crimes['Location Description'].astype('category')
    crimes['District'] = crimes['District'].astype('category')
    crimes['Community Area'] = crimes['Community Area'].astype('category')
    print(sex_offenders.head())
    print(sex_offenders.info())
    print(sex_offenders.columns)
    print(crimes.head())
    print(crimes.info())

    #Tratamento de valores nulos - Remoção das linhas com valores nulos ou que puderam ser convertidos pros tipos definidos
    sex_offenders_colunas_criticas = ['BIRTH DATE', 'BLOCK', 'GENDER', 'RACE']
    sex_offenders.dropna(subset=sex_offenders_colunas_criticas, inplace=True)
    #
    crimes_colunas_criticas = ['Date', 'Block', 'Community Area', 'Primary Type']
    crimes.dropna(subset=crimes_colunas_criticas, inplace=True)

    return sex_offenders, crimes

#Análise CSV - "Sex Offenders"
def calcular_estatisticas(sex_offenders):
    contagem_criminosos_sexuais_block = sex_offenders['BLOCK'].value_counts()
    quantidade_blocos_distintos = contagem_criminosos_sexuais_block.shape[0]

    print(f"Quantidade de blocos distintos: {quantidade_blocos_distintos}")
    media_criminosos_sexuais_block = contagem_criminosos_sexuais_block.mean()
    print(f"Média de criminosos sexuais por BLOCK: {media_criminosos_sexuais_block:.2f}")
    mediana_criminosos_sexuais_block = contagem_criminosos_sexuais_block.median()
    print(f"Mediana de criminosos sexuais por BLOCK: {mediana_criminosos_sexuais_block}")
    variancia_criminosos_sexuais_block = contagem_criminosos_sexuais_block.var()
    print(f"Variância de criminosos sexuais por BLOCK: {variancia_criminosos_sexuais_block}")
    moda_criminosos_sexuais_block = contagem_criminosos_sexuais_block.mode().iloc[0]
    print(f"Moda de criminosos sexuais por BLOCK:{moda_criminosos_sexuais_block}")

    total_blocos = contagem_criminosos_sexuais_block.shape[0]
    
    return contagem_criminosos_sexuais_block, total_blocos

def analisar_data_nascimento(sex_offenders):
    anos_nascimento = sex_offenders['BIRTH DATE'].dt.year

    plt.figure(figsize=(12, 6))
    sns.histplot(anos_nascimento, bins=30, kde=False, color='skyblue', edgecolor='black')
    # Configurações do histograma
    plt.title('Distribuição de Ano de Nascimento dos Criminosos Sexuais')
    plt.xlabel('Ano de Nascimento')
    plt.ylabel('Quantidade de Criminosos Sexuais')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 

def analisar_genero(sex_offenders):
    contagem_criminosos_sexuais = sex_offenders['GENDER'].value_counts()
    percentual_genero = sex_offenders['GENDER'].value_counts(normalize=True) * 100
    # Tabela com os percentuais por gênero
    tabela = pd.DataFrame({
        'Quantidade de Agressores': contagem_criminosos_sexuais,
        'Percentual (%)': percentual_genero.round(2)
    })
    print("Tabela de Distribuição Percentual por Gênero dos Criminosos Sexuais:")
    print(tabela)

def analisar_concentracao_por_faixas(contagem_criminosos_sexuais_block, total_blocos):
    faixas = [(1, 4), (5, 10), (11, 20), (21, 50), (51, contagem_criminosos_sexuais_block.max())]
    labels = ['1-4 (Baixa)', '5-10 (Média)', '11-20 (Alta)', '21-50 (Muito Alta)', f'>50 (Extrema)']
    quantidades = []
    
    for inicio, fim in faixas:
        blocos = contagem_criminosos_sexuais_block[
            (contagem_criminosos_sexuais_block >= inicio) & (contagem_criminosos_sexuais_block <= fim)]
        quantidades.append(len(blocos))
        print(f"Quantidade de blocos com entre {inicio} e {fim} criminosos sexuais: {len(blocos)}")
    
    percentuais = [qtd / total_blocos * 100 for qtd in quantidades]
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=labels, y=percentuais, palette="Reds_d")
    
    for bar, pct in zip(bars.patches, percentuais):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.title('Distribuição percentual da concentração de criminosos sexuais por bloco em Chicago')
    plt.xlabel('Faixa de concentração (nº de criminosos sexuais por bloco)')
    plt.ylabel('Porcentagem de blocos (%)')
    plt.ylim(0, max(percentuais) + 5)  # Adiciona espaço para os rótulos
    plt.tight_layout()
    plt.show()
# CSV- Crimes

def analisar_crimes_sexuais_ano(crimes):
    tipos_sexuais = ['CRIM SEXUAL ASSAULT', 'SEX OFFENSE']
    crimes_sexuais = crimes[crimes['Primary Type'].isin(tipos_sexuais)]
    total_crimes_sexuais = crimes_sexuais.shape[0]
    crimes_sexual_por_ano = crimes_sexuais['Year'].value_counts().sort_index()
    #Gráfico ano x  n° de crimes
    plt.figure(figsize=(12, 6))
    crimes_sexual_por_ano.plot(kind='bar', color='salmon')
    plt.title('Número de Crimes Sexuais por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Quantidade de Crimes Sexuais')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    return crimes_sexual_por_ano

''' Cruzamento dos CSVs - Relacionando os blocks que aparecem no CSv "Sex Offenders" com as community areas
de Chicago (usando o CSV "Crimes- 2001 to Present")
'''

'''  Função responsável por: mapear block para community area
 usando o dataset de 'crimes-2001 to present' e aplica
 ao DataFrame de criminosos sexuais 
 Exclui blocos registrados como 'Homeless'
 ''' 
def mapear_blocks_para_community(crimes, sex_offenders):
    mapping_block_community_from_crimes = crimes.groupby('Block')['Community Area'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else None
    ).to_dict()
    mapeamento_manual = {
        '12XXX S GREEN ST': 53.0,
        '11XXX S STATE ST': 49.0,
        '12XXX S LAFAYETTE AVE': 53.0,
        '11XXX S PERRY AVE': 49.0,
        '10XXX S LAFAYETTE AVE': 49.0,
        '10XXX S WABASH AVE': 49.0,
        '12XXX S WENTWORTH AVE': 53.0,
        '11XXX S EMERALD AVE': 49.0,
        '10XXX S AVENUE M': 52.0,
        '12XXX S WALLACE ST': 53.0,
    }
    mapa_block_para_community = {**mapping_block_community_from_crimes, **mapeamento_manual}
    sex_offenders['Community Area'] = sex_offenders['BLOCK'].map(mapa_block_para_community)

    return sex_offenders, mapa_block_para_community

def analisar_community_mais_criminosos_sexuais(contagem_criminosos_sexuais_block, crimes,sex_offenders):
    #Amostra: de blocos com +5 agressores (por causa do alto numero de 'blocks' com menos de 5 criminosos
    #sexuais)
    blocos_mais_5 = contagem_criminosos_sexuais_block[contagem_criminosos_sexuais_block >= 5].index
    criminosos_blocos_mais_5 = sex_offenders[
        sex_offenders['BLOCK'].isin(blocos_mais_5) &
        (sex_offenders['Community Area'] != 'Homeless')
    ]
    contagem_community_blocks_criminosos_sexuais = criminosos_blocos_mais_5.groupby('Community Area')['BLOCK'].nunique()
    contagem_community_blocks_criminosos_sexuais = contagem_community_blocks_criminosos_sexuais[contagem_community_blocks_criminosos_sexuais > 0]
    plt.figure(figsize=(12, 6))
    contagem_community_blocks_criminosos_sexuais.plot(kind='bar')
    plt.title('Quantidade de Blocos com mais de 5 criminosos sexuais por Community Area')
    plt.xlabel('Community Area')
    plt.ylabel('Quantidade de Blocos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
   
    tipos_sexuais = ['CRIM SEXUAL ASSAULT', 'SEX OFFENSE']
    crimes_sexuais = crimes[crimes['Primary Type'].isin(tipos_sexuais)]
    contagem_criminosos_nao_homeless = sex_offenders[sex_offenders['Community Area'] != 'Homeless']['Community Area'].value_counts()
    contagem_crimes_sexuais_community = crimes_sexuais['Community Area'].value_counts()

    df_resultado = pd.DataFrame({
        'Agressores Registrados': contagem_criminosos_nao_homeless,
        'Crimes Sexuais Registrados': contagem_crimes_sexuais_community
    }).fillna(0)

    # Filtrar Community Areas com números acima da média para ambos
    media_criminosos_sexuais_community = df_resultado['Criminosos Sexuais Registrados'].mean()
    media_crimes_sexuais_community = df_resultado['Crimes Sexuais Registrados'].mean()

    df_filtrado = df_resultado[
        (df_resultado['Criminosos Sexuais Registrados'] > media_criminosos_sexuais_community) &
        (df_resultado['Crimes Sexuais Registrados'] > media_crimes_sexuais_community)
    ]
    df_filtrado['Total'] = df_filtrado['Criminosos Sexuais Registrados'] + df_filtrado['Crimes Sexuais Registrados']
    df_filtrado = df_filtrado.sort_values(by='Total', ascending=False)

    # Gráfico das Community Areas de destaque
    plt.figure(figsize=(14, 6))
    df_filtrado[['Criminosos Sexuais Registrados', 'Crimes Sexuais Registrados']].plot(
        kind='bar', figsize=(14, 6), color=['salmon', 'skyblue']
    )
    plt.title('Community Areas com Altos Números de Criminosos Sexuais e Crimes Sexuais Registrados')
    plt.xlabel('Community Area')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
    

#Cálculo de crimes registrados/ community
def calcular_crimes_community(crimes):
    crimes_validos = crimes.dropna(subset=['Community Area'])
    contagem_crimes_por_community = crimes_validos['Community Area'].value_counts()
    print("Contagem de crimes por 'Community Area':")
    print(contagem_crimes_por_community)

    media_crimes_por_community = contagem_crimes_por_community.mean()
    print(f"Média de crimes gerais por 'Community Area': {media_crimes_por_community:.2f}")
    mediana_crimes_por_community = contagem_crimes_por_community.median()
    print(f"Mediana de crimes gerais por 'Community Area': {mediana_crimes_por_community}")
    variancia_crimes_por_community = contagem_crimes_por_community.var()
    print(f"Variância de crimes gerais por 'Community Area': {variancia_crimes_por_community:.2f}")
    moda_crimes_por_community = contagem_crimes_por_community.mode().iloc[0]
    print(f"Moda de crimes gerais por 'Community Area':{moda_crimes_por_community}")

    return contagem_crimes_por_community, crimes_validos


def analisar_crimes_por_community(contagem_crimes_por_community):
    total_crimes = contagem_crimes_por_community.sum()
    percentuais = contagem_crimes_por_community / total_crimes * 100
  
    plt.figure(figsize=(12, 6))
    sns.barplot(x=contagem_crimes_por_community.index.astype(str), y=contagem_crimes_por_community.values, palette="Reds_d")
    plt.title('Número de crimes por Community Area em Chicago')
    plt.xlabel('Community Area')
    plt.ylabel('Número de crimes')
    plt.xticks(rotation=90)  # Gira os rótulos para não sobrepor
    plt.tight_layout()
    plt.show()
    return percentuais

#Existe alguma community que se destaca simultaneamente com altos números de crimes e agressores sexuais?
def analisar_agressores_community(mapa_block_para_community,sex_offenders,crimes):
   #Selecionar crimes sexuais
    mapping_block_community=sex_offenders['Community Area'] = sex_offenders['BLOCK'].map(mapa_block_para_community)
    contagem_agressores = sex_offenders['Community Area'].value_counts()
    tipos_sexuais = ['CRIM SEXUAL ASSAULT', 'SEX OFFENSE']
    crimes_sexuais = crimes[crimes['Primary Type'].isin(tipos_sexuais)]
    contagem_crimes_sexuais = crimes_sexuais['Community Area'].value_counts()
    df_resultado = pd.DataFrame({
        'Criminosos Sexuais Registrados': contagem_agressores,
        'Crimes Sexuais Registrados': contagem_crimes_sexuais
    }).fillna(0) 
    #Gráfico
    media_agressores = df_resultado['Criminosos Sexuais Registrados'].mean()
    media_crimes = df_resultado['Crimes Sexuais Registrados'].mean()
    df_filtrado = df_resultado[
        (df_resultado['Criminosos Sexuais Registrados'] > media_agressores) &
        (df_resultado['Crimes Sexuais Registrados'] > media_crimes)
    ]
    df_filtrado['Total'] = df_filtrado['Criminosos Sexuais Registrados'] + df_filtrado['Crimes Sexuais Registrados']
    df_filtrado = df_filtrado.sort_values(by='Total', ascending=False)
    # Gráfico
    plt.figure(figsize=(14, 6))
    df_filtrado[['Criminosos Sexuais Registrados', 'Crimes Sexuais Registrados']].plot(
        kind='bar', figsize=(14, 6), color=['salmon', 'skyblue']
    )
    plt.title('Community Areas com Altos Números de Criminosos Sexuais e Crimes Sexuais Registrados')
    plt.xlabel('Community Area')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

def tabela_frequencia_relativa_crimes_sexuais(crimes):
    crimes = crimes.dropna(subset=['Community Area'])
    tipos_sexuais = ['CRIM SEXUAL ASSAULT', 'SEX OFFENSE']
    crimes_sexuais = crimes[crimes['Primary Type'].isin(tipos_sexuais)]
    contagem_crimes_sexuais = crimes_sexuais['Community Area'].value_counts().sort_index()
    contagem_total = crimes['Community Area'].value_counts().sort_index()
    frequencia_relativa = (contagem_crimes_sexuais / contagem_total)
    tabela_frequencia = pd.DataFrame({
        'Crimes Totais': contagem_total,
        'Crimes Sexuais': contagem_crimes_sexuais,
        'Frequência Relativa (%)': frequencia_relativa
    }).fillna(0).round(2)
    return tabela_frequencia

def calcular_probabilidades(crimes, sex_offenders):
    crimes['A'] = crimes['Primary Type'].isin(['CRIM SEXUAL ASSAULT', 'SEX OFFENSE']).astype(int)
    community_com_agressores = sex_offenders['Community Area'].dropna().unique()
    crimes['B'] = crimes['Community Area'].isin(community_com_agressores).astype(int)
    # Distribuição das probabilidades com o uso das variáveis aleatórias
    p_a = crimes['A'].mean()
    p_b = crimes['B'].mean()
    p_b_dado_a = crimes[crimes['A'] == 1]['B'].mean()
    p_a_dado_b = (p_b_dado_a * p_a) / p_b if p_b > 0 else 0
    # Distribuição por área (como antes)
    crimes_sexuais, p_a_por_community = calcular_probabilidade_crime_sexual_por_community(crimes)
    # Prints informativos
    print(f"Probabilidade média de um crime ser sexual (P(A)): {p_a:.4f}")
    print(f"Probabilidade de um crime ocorrer em 'Community Area' com agressores (P(B)): {p_b:.4f}")
    print(f"Probabilidade de um crime sexual ocorrer em 'Community Area' com agressores (P(B|A)): {p_b_dado_a:.4f}")
    print(f"Probabilidade condicional (Bayes - P(A|B)): {p_a_dado_b:.4f} ({p_a_dado_b * 100:.2f}%)")

    return p_a_por_area, p_b, p_b_dado_a, p_a_dado_b

def calcular_probabilidade_crime_sexual_por_community(crimes):
    tipos_sexuais = ['CRIM SEXUAL ASSAULT', 'SEX OFFENSE']
    crimes_sexuais = crimes[crimes['Primary Type'].isin(tipos_sexuais)]

    total_por_community = crimes['Community Area'].value_counts()
    sexuais_por_community = crimes_sexuais['Community Area'].value_counts()
    p_a_por_community = (sexuais_por_community / total_por_community).fillna(0)
    print(f"Probabilidade de ocorrência de um crime sexual por Community Area:\n{p_a_por_community}")
    return crimes_sexuais, p_a_por_community

def calcular_probabilidade_community_com_agressores(crimes, sex_offenders):
    community_com_agressores = sex_offenders['Community Area'].dropna().unique()
    crimes_em_community_agressores = crimes[crimes['Community Area'].isin(community_com_agressores)]
    total_crimes = crimes.shape[0]
    p_b = crimes_em_community_agressores.shape[0] / total_crimes
    print(f"Probabilidade de um crime ocorrer em uma Community Area com agressores sexuais registrados: {p_b:.4f}")
    return community_com_agressores, total_crimes, p_b
'''
Distribuição das probabilidades
Variáveis aleatórias: 
A --> P(A)= 1, crime é sexual. 
      P(A)=0, criem não é sexual.
B --> P(B)= 1, o crime ocorreu em uma área com agressores registrados.
      P(B)= 0, o crime não ocorreu em uma área com agressores registrados.

'''
def analisar_distribuicao_normal(p_a_por_area):
    dados = p_a_por_area[p_a_por_area > 0]
    # Histograma e curva de densidade normal teórica
    plt.figure(figsize=(12, 6))
    sns.histplot(dados, kde=True, stat='density', bins=20,
                 color='lightblue', edgecolor='black', label='Dados')
    media = dados.mean()
    desvio = dados.std()
    x = np.linspace(dados.min(), dados.max(), 100)
    y = norm.pdf(x, media, desvio)
    plt.plot(x, y, 'r', label='Distribuição Normal Teórica')
    plt.title('Verificação de Normalidade das Probabilidades por Área')
    plt.xlabel('Probabilidade de Crime Sexual')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Gráfico Q-Q 
    plt.figure(figsize=(6, 6))
    probplot(dados, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.grid(True)
    plt.show()
    # Teste de Shapiro-Wilk
    stat, p_valor = shapiro(dados)
    print(f"Shapiro-Wilk: estatística={stat:.4f}, p-valor={p_valor:.4f}")
    if p_valor > 0.05:
        print("→ Os dados seguem uma distribuição normal (nível de 5%).")
    else:
        print("→ Os dados NÃO seguem uma distribuição normal (nível de 5%).")
    #Cálculo Acumulado
    cdf_teorica = norm.cdf(dados.sort_values(), loc=media, scale=desvio)
    plt.figure(figsize=(10, 6))
    plt.plot(dados.sort_values(), cdf_teorica, color='red', label='Cálculo Acumulado da Normal Teórica')
    plt.step(dados.sort_values(), np.arange(1, len(dados) + 1) / len(dados),
             where='post', label='Cálculo Acumulado da Normal Empírica', color='blue')
    plt.title("Distribuição Acumulada")
    plt.xlabel("Probabilidade de Crime Sexual")
    plt.ylabel("P(X ≤ x)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Qual a probabilidade de ocorrer exatamente k crimes sexuais em n crimes, se a probabilidade de um crime ser sexual é 𝑝?
def analisar_distribuicao_binomial(crimes, n=500, k=50):
    p = crimes['A'].mean()
    prob = binom.pmf(k, n, p)
    print(f"P(X = {k}) com n={n}, p={p:.4f}: {prob:.4f}")
    x = range(0, n + 1)
    y = binom.pmf(x, n, p)
    plt.figure(figsize=(12, 6))
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.title(f'Distribuição Binomial - P(X=k), n={n}, p={p:.4f}')
    plt.xlabel('Número de Crimes Sexuais')
    plt.ylabel('Probabilidade')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    sex_offenders, crimes = carregar_dados()
    contagem_criminosos_sexuais_block, total_blocos = calcular_estatisticas(sex_offenders)
    analisar_data_nascimento(sex_offenders)
    analisar_genero(sex_offenders)
    analisar_concentracao_por_faixas(contagem_criminosos_sexuais_block, total_blocos)  
    crime_sexual_por_ano= analisar_crimes_sexuais_ano(crimes)
    sex_offenders, mapa_block_para_community= mapear_blocks_para_community(crimes, sex_offenders)
    analisar_agressores_community(mapa_block_para_community,sex_offenders,crimes)

if __name__ == '__main__':
    main()

