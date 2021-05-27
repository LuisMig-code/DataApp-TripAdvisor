# importando os dados:
from deep_translator import GoogleTranslator


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output , State
import dash_bootstrap_components as dbc

# lendo os dados:
df = pd.read_csv("data/tripadvisor_hotel_reviews.csv")

# traduz textos:
def translate(frase_pt):
    text_traduzido = GoogleTranslator(source='portuguese', target='en').translate(frase_pt)

    return text_traduzido

# criando o modelo:
def modelo():
    data = df

    vectorizer = CountVectorizer(analyzer="word")

    reviews = data['Review']
    vect_reviews = vectorizer.fit_transform(reviews)
    notas = data['Rating']

    modelo = MultinomialNB()
    modelo.fit(vect_reviews, notas)

    return modelo



########################################################################################################################
############################################      APP    ###############################################################
########################################################################################################################

# Criando o app da dashboard setando um tema do Bootstrap
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.MINTY],
                                        # Responsivilidade para mobile layout
                                        meta_tags=[ {'name': 'viewport',
                                                    'content': 'width=device-width, initial-scale=1.0'}  ] )



# Criando a Dashboard em si:
app.layout = dbc.Container([
    # Cabeçalho:
    dbc.Row([
        dbc.Col([
            # título principal
            html.H1("DataApp - TripAdvisor" ,
                    className = "text-center text-primary, display-2 shadow") ,

            # subtítulo
            html.H2("DataApp criada para sugerir classificações a partir de comentários" ,
                    className = "text-sm-center text-primary display-3 font-weight-bold" ,
                    style = {"font-size":24})

        ] , width=12)
    ]) ,

    html.Br(),

    # Main :
    dbc.Row([
            # coluna com a textarea
            dbc.Col([
                html.H3("Digite sua Review:"),
                html.Br(),
                dcc.Textarea(
                        id='texto-review',
                        placeholder = 'Digite aqui sua review do quarto de hotel',
                        value = "",
                        style={'width': '100%', 'height': 280},
                    ),
                html.Button('Classificar', id='botao_classificar' , className = "btn btn-primary" , n_clicks=0)
            ] , width = {'size' : 4 , 'order':1}) ,

            # Coluna do resultado
            dbc.Col([
                html.H3("Resultado:"),
                html.Img(src="" , className="align-self-center" , id="img_classificacao")
            ] , width = {'size' : 8 , 'order':2})
        ]) ,



] , fluid = True)

##### CallBacks #####
@app.callback(
    Output("img_classificacao","src"),
    Input("botao_classificar","n_clicks"),
    State("texto-review","value")
)
def classifica_texto(n_clicks , value):
    if n_clicks is not None:
        # guardando o texto em português:
        texto_pt = value

        #traduxindo o texto para classificação:
        review = translate(texto_pt)

        # instanciando o vetorizador :
        vectorizer = CountVectorizer(analyzer="word")

        # vetorizando os textos:
        reviews = df['Review']
        vect_reviews = vectorizer.fit_transform(reviews)

        # classificações (notas):
        notas = df['Rating']

        # instanciando e treinando o modelo:
        modelo = MultinomialNB()
        modelo.fit(vect_reviews, notas)

        # vetorizando o texto digitado pelo usuario:
        vect_texto = vectorizer.transform([review])

        # fazendo a predição da nota:
        predicao = modelo.predict(vect_texto)

        return "assets/nota-{}.png".format(predicao[0])

    else:
        return ""

# Instanciando a aplicação:
if __name__ == '__main__':
    app.run_server(debug=True , port=8000)