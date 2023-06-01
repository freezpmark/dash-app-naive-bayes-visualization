import dash, pickle, json, re, os
from dash import dcc
from dash import html
from dash import dash_table as dt
import numpy as np
import dashCLASSIFIER as cf
import collections as coll

stopWord = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                                    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def nestFunc():
    return coll.defaultdict(list)

# calculate percentual probability from logarithm values
def probMap(probList, prior = False):
    avgPerc = sum(probList)/3
    if prior:
        for ith, i in enumerate(probList):          # calc % for positive values
            probList[ith] = (i * 33.33) / avgPerc
    else:                                           # calc % for negative values
        for ith, i in enumerate(probList):
            probList[ith] = (((2*avgPerc) - i) * 33.33) / avgPerc
    return probList

# creating axis properties for ternary graph
def makeAxis(title, minPerc): 
    return {
      'min': minPerc,
      'title': title,
      'titlefont': { 'size': 15 },
      'tickangle': 0,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(20,40,100,50)',
      'ticklen': 5,
      'showgrid': True
    }

import base64
with open(os.path.join("images", "NBclassifier.png"), "rb") as handle:
    encoded_image = base64.b64encode(handle.read())

with open(os.path.join("datasets", "dashjupy-likelihood"), "rb") as handle:
    likelihood = pickle.loads(handle.read())            # likelihood[className][word]   {counted}
with open(os.path.join("datasets", "dashjupy-priors"), "rb") as handle:
    priors = pickle.loads(handle.read())                # prior[className]              {counted}
with open(os.path.join("datasets", "dashjupy-content"), "rb") as handle:
    content = pickle.loads(handle.read())               # content[className][sampleName][words(list)]

app = dash.Dash()

app.layout = html.Div([
    # title of the application
    html.H1('Textova klasifikacia s Naivnym Bayesovskym klasifikatorom',style={'textAlign': 'center'}),
    
    # Naive Bayes intro text
    html.Div([
        dcc.Markdown('''**Naivny Bayesovsky klasifikator** je algoritmus strojoveho ucenia vyuzivajuci Bayesove pravidlo zalozene na teorii pravdepodobnosti pre klasifikovanie kategorii vzoriek.  
                **Proces ucenia** spociva v dodavani trenovacich dat - vzorky (textove dokumenty) s priradenou kategoriou. 
                Spocitaju sa frekvencie kazdeho slova vo vzorkach pre ich priradenu kategoriu a taktiez sa spocitaju aj vzorky pri kazdej kategorii.  
                **Proces klasifikacie** je mozne vyjadrit nasledujucou vypoctovou formulou:''', 
                containerProps={'style': {'text-align': 'justify', 'font-family': 'Helvetica'}}),
    
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
            style={
                'width': '30%',
                'heights': '15%',
                'display': 'inline-block',
                'text-align': 'center',
                'marginLeft': '250px'}),
        dcc.Markdown("""C_k - aktualne spracovavana kategoria  
        w_i - aktualne vypocitavane slovo actually being computed  
        K - pocet kategorii  
        n - pocet slov v neznamej vzorke""", 
        containerProps={'style': {'display': 'inline-block', 'font-family': 'Helvetica'}}),

        dcc.Markdown("""**Zhrnutie celeho procesu:** Hladame argmax najpravdepodobnejsiu kategoriu (y^) do ktorej moze patrit neznama vzorka. 
        Pre kazdu z K moznych kategorii sa vypocita pravdepodobnost ktora pozostava z vynasobenia tychto dvoch pravdepodobnosti:
- **prior** P(C_k) - pravdepodobnost vybratia vzorky s kategoriou C_k zo vsetkych vzoriek v trenovacej mnozine (#vzoriek v C_k kategorii / #vzoriek vo vsetkych kategoriach)  
- **likelihood** -||-P(w_i | C_k) - pravdepodobnost pozostavajuca z vynasobeni pravdepodobnosti medzi vsetkymi n slovami vyskytujuce sa vo vzorke ktora sa predikuje. 
        Tieto pravdepodobnosti slov su vypocitavane tymto vztahom: (#slov w_i v kategorii C_k) / (celkovy # slov v kategorii C_k)
    + pri likelihoode moze dojst k pripadu slova s nulovou pocetnostou, co sposobi celkovu nulovu pravdepodobnost likelihoodu. 
    Preto je potrebne bud nahradit tieto pravdepodobnosti racionalnym cislom alebo pridat jeden vyskyt pre vsetky existujuce slova (Laplace smoothing).""", 
    containerProps={'style': {'text-align': 'justify', 'font-family': 'Helvetica'}}),
    ],
    style={
        'paddingLeft': '100px',
        'paddingRight': '100px',
        'paddingBottom': '30px'
        #'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)',
    }),
    
    # left side section (intro, classification, report)
    html.Div([
        dcc.Markdown('## Klasifikacne parametre a ich vysledky', containerProps={'style': {'text-align': 'center'}}),

        dcc.Markdown("""Pouzite datasety (news, ohsu, revs) su rozdelene do **trenovacieho(75%)** a **testovacieho(25%)** datasetu.  
            **Vyberte tri kategorie** z ktorych sa vyberu trenovacie vzorky na trenovanie a testovacie vzorky pre klasifikaciu.""",
            containerProps={'style': {'font-family': 'Helvetica'}}),
        dcc.Dropdown(
            id='dropdown-categorySelection',
            options= [{'label': str(category),'value': str(category)} for category in priors],
            placeholder="Vyber 3 kategorie pre klasifikaciu",
            value=['news-Graphics', 'news-Forsale', 'news-Baseball'],
            multi=True
        ),

        # zero-fix selection
        dcc.Markdown('### Moznosti riesenia nulovych frekvencii slov'),
        dcc.RadioItems(
            id = "radio-zeroFixSelection",
            options=[
                {'label': 'Vyber cislo', 'value': 'RTN'},
                {'label': 'Laplace smoothing', 'value': 'LAP'},
            ],
            value='RTN',
            labelStyle={'display': 'inline-block'},
            style={
                'padding': '5px',
                'display': 'inline-block'
            }
        ),
        # tooltip about zero fix selection
        html.Abbr("[?]", 
            title="Vyber cislo - vyber cislo zo slideru nizsie pre nahradenie vsetkych nulovych frekvencii slov v kategoriach\
                        \nLaplace smoothing - pridanie jedneho vyskytu slova do vsetkych atributov kategorie",
            style={
                'display': 'inline-block',
                'marginLeft': '15px',
                'cursor': 'help',
                'text-decoration': 'none'
            }
        ),

        # range slider for rational number selection
        dcc.RangeSlider(
            id='rangeSlider-rationalNum',
            marks={i: '1e-{}'.format(i) for i in range(4,13)},
            max=12,
            min=4,
            value=[8],
            dots=False,
            step=0.01,
            updatemode='drag'
        ),
        html.Br(),
        html.Br(),

        # output for exact value of selected rational number
        html.Div(id='div-outputZeroF', style={'display': 'inline-block', 'width': '200px'}),

        html.Button('Klasifikuj', 
            id='button-classify',
            style={
                'font-size': '15px',
                'cursor': 'pointer',
                'color': 'black',
                'width': '100px',
                'height': '25px',
                'display': 'inline-block',
                'marginLeft': '50px'
            }
        ),
        html.Br(),

        # classification report of 3 chosen categories
        dcc.Markdown('### Klasifikacny report ', containerProps={'style': {'display': 'inline-block', 'font-family': 'Helvetica'}}),
        # tooltip about score metrics
        html.Abbr("[?]", 
            title="Precision - Pravdepodobnost, ze klasifikovana vzorka do prislusnej kategorie bola spravne klasifikovana\
                \nRecall - Pravdepodobnost, ze vzorka prislusnej kategorie bola spravne klasifikovana\
                \nAverage - (Precision + Recall) / 2\
                \nF1-Score - 2 * (Precision * Recall) / (Precision + Recall)\
                \nSupport - Pocet vzoriek",
            style={
                'display': 'inline-block',
                'marginLeft': '15px',
                'cursor': 'help',
                'text-decoration': 'none'
            }
        ),

        html.Div([
            dt.DataTable(
                rows = [{},{},{}],
                #sortable=True,
                editable=False,
                selected_row_indices=[3],
                #row_selectable=False,
                #enable_drag_and_drop=False,
                id='dataTable-scoreMetrics',
                column_widths=[200, 75, 75, 75, 75, 75]
            )],
            style={'width': '600'}
        ),
            
        # accuracy value output
        html.Br(),
        html.Div(id='div-accuracy', 
            style={
                'font-size': '20px',
                'white-space': 'pre',
                'textAlign': 'center',
            }
        )],style={
            'padding': '25px',
            'marginLeft': '10px',
            'width': "600",
            'height': "675",
            'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)',
            'float': 'left'
        }
    ),

    # right side section for prior and word frequencies/likelihood graphs
    html.Div([
        dcc.Markdown('## Vlastnosti trenovacej mnoziny', containerProps={'style': {'text-align': 'center'}}),

        # bar graph of priors
        dcc.Graph(
            id='graph-prior',
            figure={
                'data': [{
                    'x': [str(category) for category in priors], 
                    'y': [str(priors[category]) for category in priors], 
                    'type': 'bar', 
                    'name': 'Vzorky'
                },{
                    'x': [str(category) for category in likelihood],
                    'y': [str(sum(likelihood[category].values())) for category in likelihood],
                    'type': 'bar',
                    'name': 'Slova',
                    'visible': 'legendonly'
                },{
                    'x': [str(category) for category in likelihood],
                    'y': [str(sum(likelihood[category].values())/priors[category]) for category in likelihood],
                    'type': 'bar',
                    'name': 'Priemer #slov',
                    'visible': 'legendonly'
                }],
                'layout': {
                    'title': 'Pocty vzoriek/slov v kategoriach',
                    'titlefont': {"size": 20},
                    'height': '550',
                    'legend': {
                        "font": {"size": 9}
                    },
                }
            },
        )],
        style={
            'padding': '25px',              # space between div start and content inside 
            'marginLeft': '665px',          # position placing for not getting overlayed
            "width": "600",                 # width of the block
            "height": "675",
            'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)',
        }
    ),

    html.Div(id='moreCatsSpace'),

    # div for word freq/imp graph
    html.Div([
        dcc.Markdown('## Atributy trenovacieho datasetu', containerProps={'style': {'text-align': 'center'}}),
            
        # word freq/imp selecion
        dcc.RadioItems(
            id = 'radio-wordPreference',
            options=[
                {'label': 'Frekvencie slov', 'value': 'wFreq'},
                {'label': 'Pravdepodobnost slov', 'value': 'wImp'},
            ],
            value='wFreq',
            style={'display': 'inline-block', 'marginLeft': '460px'}
        ),
        # tooltip about freq/imp selection
        html.Abbr("[?]", 
            title="Frekvencie slov - spocita frekvencie slov vo vybratych kategoriach\
                    \nPravdepodobnost slov - vypocita pravdepodobnost slova vzhladom na celkovy pocet slov v kategorii\
                    \n(frekvencia urciteho slova v kategorii trenovacieho datasetu je vydeleny celkovym poctom slov v kategorii)\
                    \nPoznamka: Stop-slova niesu pouzite v trenovacom datasete, su vsak vyfiltrovane v tomto grafe lebo nam nedavaju podstatne informacie",
            style={
                'display': 'inline-block',
                'marginLeft': '15px',
                'cursor': 'help',
                'text-decoration': 'none'
            },
        ),
        html.Br(), html.Br(),

        # selection of category with sorted frequent/importance words
        dcc.Dropdown(
            id='dropdown-categoryPreference',
            options=[{'label': str(category),'value': str(category)} for category in priors],
            placeholder="Vyberte kategoriu pri ktorej chcete vidiet najviac frekventovane slova",
            searchable=False
        ),

        # tab selection for different view of the graph
        dcc.Tabs(
            tabs=[{'label': 'Tab {}'.format(i), 'value': i} for i in range(1, 3)],
            value=2,
            id='tabs-select'
        ),

        # graph of word freqs/imps
        html.Div(id='div-graph-bar'),
        ],
        style={'paddingLeft': '40px', 'paddingRight': '40px'}
    ),

    # section for ternaries, process, uniqs and sample text
    html.Div([
        dcc.Graph(id='graph-ternarySamples', style={'display': 'inline-block', 'marginLeft': '30px'}),
        dcc.Graph(id='graph-ternaryWords', style={'display': 'inline-block', 'white-space': 'pre'}),

        # range picker of words
        html.Div([
            dcc.RangeSlider(
                id='slider-wordRange',
                step=1,
            )],
            style={
                'marginLeft': '25px',
                'display': 'inline-block',
                'width': '1200'
            }
        ),
        # tooltip for word selection
        html.Abbr("[?]", 
                title="Vyber rozsahu slov pri aktualne vybratej vzorke\n" + 
                    "   - aktualizuje klasifikacny proces\n" +
                    "   - aktualizuje klasifikaciu kategorie (graf pravdepodobnosti slov cez farbu)",
                style={
                    'display': 'inline-block',
                    'marginLeft': '25px',
                    'cursor': 'help',
                    'text-decoration': 'none'
                }
        ),
        # output with boundaries of selected word range
        html.Div(id='div-wordRange', style={'textAlign': 'center'}),
        html.Br(),

        # range slider for reset point selection
        html.Div([
            dcc.Slider(
                id='slider-chunkSum',
                min=1,
                max=100,
                step=1,
                value=25,
                marks={
                    10: '10', 20: '20', 30: '30', 40: '40', 50: '50',
                    60: '60', 70: '70', 80: '80', 90: '90', 100: '100'
                }
            )],
            style={
                'marginLeft': '25px',
                'display': 'inline-block',
                'width': '1200'
            }
        ),
        # tooltip about reset points
        html.Abbr("[?]", 
            title="Celkova pravdepodobnost je kazdym vypocitanym slovom menej ovplyvnena,\n" +
                "na zobrazenie chodu cez cely proces, nastav resetujuci pravdepodobnostny bod pre\n" +
                "rozdelenie do viacerich segmentov",
            style={
                'display': 'inline-block',
                'marginLeft': '25px',
                'cursor': 'help',
                'text-decoration': 'none'
            }
        ),
        html.Br(),
        html.Br(),
        # reset point output index
        html.Div(id='div-sumRange', style={'textAlign': 'center',}),

        # graph of probability computation process
        dcc.Graph(id='graph-process'),

        html.Br(),
        # graph of uniqueness of words for each category
        html.Div(id='div-wordImportances3', style={'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'}),

        dcc.Markdown('### Text vzorky', containerProps={'style': {'textAlign': 'center'}}),

        # text of selected sample also influenced with range selection
        html.Div(id='div-sampleText',
            style={
                'padding': '30px',
                'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)',
                'text-align': 'justify'
            }),
    ]),
    
    # preprocessing hidden values
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id ='selected-sample', style={'display': 'none'}),
    ],
    style={
        'padding': '5px',
		'width': '1300px',
		'margin': 'auto',
    	'box-shadow': '0px 0px 20px #00070E',
    	'border-style': 'solid',
    	'border-width': '2px'
    }
)

################ CALLBACK FUNCTIONS ################
#app.config.supress_callback_exceptions = True

# updating category preference dropdown
@app.callback(
    dash.dependencies.Output(component_id='dropdown-categoryPreference', component_property='options'),
    [dash.dependencies.Input(component_id='dropdown-categorySelection', component_property='value')])
def updatePreference(selectedCategories):
    return [{'label': category, 'value': category} for category in selectedCategories]

# displaying graph of word frequencies 
@app.callback(
    dash.dependencies.Output('div-graph-bar', 'children'), 
    [dash.dependencies.Input('tabs-select', 'value'),
     dash.dependencies.Input('dropdown-categorySelection', 'value'),
     dash.dependencies.Input('dropdown-categoryPreference', 'value'),
     dash.dependencies.Input('radio-wordPreference', 'value')])
def displayFreqGraph(tabSelection, chosenCategories, preferedCategory, wordPref):
    if len(chosenCategories) != 3:
        return
    secCats = [category for category in chosenCategories if category != preferedCategory]
    if len(secCats) == 3:                                   # changing chosenCategories doesnt change preferedCategory (color fix correction)
        preferedCategory = secCats.pop()
    newLikelihood1 = dict(likelihood[preferedCategory])     # assigning global list into new local one without id reference via list()

    colors = [0, 0, 0]
    for ith, i in enumerate(chosenCategories):
        if i == preferedCategory:
            colors[0] = ith
        elif i == secCats[0]:
            colors[1] = ith
        else:
            colors[2] = ith

    for word in stopWord:                                       # delete stopWords from new likelihood
        newLikelihood1.pop(word, None)
    sortedValues = sorted(newLikelihood1.items(), key=lambda x: x[1], reverse=True) # save number of word frequencies in sorted order
    sortedWords = [word[0] for word in sortedValues[:150]]      # save first 150 most frequent words 

    if wordPref == 'wImp':
        newLikelihood2 = {}
        newLikelihood3 = {}
        n1 = float(sum(likelihood[preferedCategory].values()))  # sum of all words
        n2 = float(sum(likelihood[secCats[0]].values()))
        n3 = float(sum(likelihood[secCats[1]].values()))
        for i in sortedValues:                                  # probability calculations
            newLikelihood1[i[0]] = (newLikelihood1[i[0]] / n1)
            newLikelihood2[i[0]] = (likelihood[secCats[0]][i[0]] / n2)
            newLikelihood3[i[0]] = (likelihood[secCats[1]][i[0]] / n3)
        tmpData = [{
            'x': sortedWords,
            'y': [newLikelihood1[x] for x in sortedWords],
            'name': preferedCategory,
            'marker': { 'color': ['red', 'green', 'blue'][colors[0]]},
            'outlinewidth': 10,
            'type': ['bar', 'scatter'][int(tabSelection) % 2]
        },{
            'x': sortedWords,
            'y': [newLikelihood2[x] for x in sortedWords],
            'name': secCats[0],
            'marker': { 'color': ['red', 'green', 'blue'][colors[1]]},
            'outlinewidth': 10,
            'type': ['bar', 'bar'][int(tabSelection) % 2]
        },{
            'x': sortedWords,
            'y': [newLikelihood3[x] for x in sortedWords],
            'name': secCats[1],
            'marker': {'color': ['red', 'green', 'blue'][colors[2]]},
            'outlinewidth': 10,
            'type': ['bar', 'bar'][int(tabSelection) % 2]
        }]
    else:
        tmpData = [{
            'x': sortedWords,
            'y': [likelihood[preferedCategory][x] for x in sortedWords],
            'name': preferedCategory,
            'marker': { 'color': ['red', 'green', 'blue'][colors[0]]},
            'type': ['bar', 'scatter'][int(tabSelection) % 2]
        },{
            'x': sortedWords,
            'y': [likelihood[secCats[0]][x] for x in sortedWords],
            'name': secCats[0],
            'marker': { 'color': ['red', 'green', 'blue'][colors[1]]},
            'type': ['bar', 'bar'][int(tabSelection) % 2]
        },{
            'x': sortedWords,
            'y': [likelihood[secCats[1]][x] for x in sortedWords],
            'name': secCats[1],
            'marker': {'color': ['red', 'green', 'blue'][colors[2]]},
            'type': ['bar', 'bar'][int(tabSelection) % 2]
        }]
        
    return dcc.Graph(
        id='graph',
        figure={
            'data': tmpData,
            'layout': {
                'margin': {
                    'l': 30,
                    'r': 0,
                    'b': 30,
                    't': 0,
                },
                'barmode': ['stack', 'group'][int(tabSelection) % 2],
                'legend': {'x': 0.9, 'y': 0.9},
                'xaxis': {'rangeslider': {}}
            }
        }
    ),

# displaying exact zero fix rational number
@app.callback( #component_id, component_property
    dash.dependencies.Output('div-outputZeroF', 'children'),
    [dash.dependencies.Input('rangeSlider-rationalNum', 'value'),
     dash.dependencies.Input('radio-zeroFixSelection', 'value')])
def displayZeroRTN(value, radio):
    if radio == "RTN":
        return 'Value: 1 / {}'.format(round(10 ** value[0]))
    else:
        return ""

# displaying score metrics table
@app.callback(
    dash.dependencies.Output('dataTable-scoreMetrics', 'rows'),
    [dash.dependencies.Input('intermediate-value', 'children')])
def displayMetrics(intermediate):
    data = json.loads(intermediate)
    metrics = [{'Category': category,
                    'Precision': data[category]['report'][0], 
                    'Recall': data[category]['report'][1],
                    'Average': data[category]['report'][2],
                    'F1-score': data[category]['report'][3],
                    'Support': data[category]['report'][4]} for category in data if category != 'Avg/total' and category != 'accuracy']
    metrics.append({'Category': 'Avg/total', 'Precision': data['Avg/total'][0], 'Recall': data['Avg/total'][1], 'Average': data['Avg/total'][2], 'F1-score': data['Avg/total'][3], 'Support': data['Avg/total'][4]})
    return metrics

# displaying ternary graph of samples (HARD)
@app.callback(
    dash.dependencies.Output('graph-ternarySamples', 'figure'),
    [dash.dependencies.Input('intermediate-value', 'children')])
def displaySampleGraph(intermediate):
    data = json.loads(intermediate) # data[className]['testSampsProbs'] [sampleName][className][2d-row=cat, col=word, na konci vysl p, minP]
    data.pop('Avg/total', None)
    data.pop('accuracy', None)

    # getting right data format for visualization
    vizData = {}                    # vizData['class(sampleBelonging)] [0(list of samples)] [probs of 3 chosen categories, sampleLabel]
    chosenCats = []                 # discovering selected categories (no need another input for callback)
    minZoom = [33, 33, 33]          # min probability values for visualization with zoom
    for i in data:                                                      # i actual className
        chosenCats.append(i)
        vizData[i] = []
        for jth, j in enumerate(data[i]['testSampsProbs']):             # j sample name
            vizData[i].append({})
            scProbs = []                                                # sample class probabilities
            for k in data[i]['testSampsProbs'][j]:                      # k predicted className 
                scProbs.append(data[i]['testSampsProbs'][j][k][-1])     # -1th value is overall probability of class k
            percProbs = probMap(scProbs)                                # mapping log probabilities of sample into percentual probabilities
            for kth, k in enumerate(data[i]['testSampsProbs'][j]):
                vizData[i][jth][k] = percProbs[kth]
                if vizData[i][jth][k] < minZoom[kth]:
                    minZoom[kth] = vizData[i][jth][k]
            vizData[i][jth]['label'] = j
            vizData[i][jth]['size'] = len(data[i]['testSamps'][j])      # word count
    
    figure = {
        'data': [{ 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[chosenCats[0]], vizData[chosenCats[0]])],
            'b': [i for i in map(lambda x: x[chosenCats[1]], vizData[chosenCats[0]])],
            'c': [i for i in map(lambda x: x[chosenCats[2]], vizData[chosenCats[0]])],
            'text': [i for i in map(lambda x: x['label'] + ", pocet slov: " + str(x['size']), vizData[chosenCats[0]])],
            'marker': {'color': 'red',},
            'name': chosenCats[0],
            'customdata': [chosenCats[0] for i in range(0, len(vizData[chosenCats[0]]))]
        },{ 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[chosenCats[0]], vizData[chosenCats[1]])],
            'b': [i for i in map(lambda x: x[chosenCats[1]], vizData[chosenCats[1]])],
            'c': [i for i in map(lambda x: x[chosenCats[2]], vizData[chosenCats[1]])],
            'text': [i for i in map(lambda x: x['label'] + ", pocet slov: " + str(x['size']), vizData[chosenCats[1]])],
            'marker': {'color': 'green',},
            'name': chosenCats[1],
            'customdata': [chosenCats[1] for i in range(0, len(vizData[chosenCats[1]]))]
        },{ 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[chosenCats[0]], vizData[chosenCats[2]])],
            'b': [i for i in map(lambda x: x[chosenCats[1]], vizData[chosenCats[2]])],
            'c': [i for i in map(lambda x: x[chosenCats[2]], vizData[chosenCats[2]])],
            'text': [i for i in map(lambda x: x['label'] + ", pocet slov: " + str(x['size']), vizData[chosenCats[2]])],
            'marker': {'color': 'blue',},
            'name': chosenCats[2],
            'customdata': [chosenCats[2] for i in range(0, len(vizData[chosenCats[2]]))]
        },{   # probability separation lines
            'a': [0, 33],
            'b': [50, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            'showlegend': False
        },{
            'a': [50, 33],
            'b': [0, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            'showlegend': False
        },{
            'a': [50, 33],
            'b': [50, 33],
            'c': [0, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            'showlegend': False
        }],
        'layout': {
            'title': "<b>Pravdepodobnosti vzoriek</b><br>(<b>KLIKNI</b> na vzorku pre zobrazenie klasifikacneho procesu)",
            'height': 600,
            'width': 600,
            'hoverdistance': 3,
            'ternary': {
                'sum': 100,
                'aaxis': makeAxis(chosenCats[0], minZoom[0]),
                'baxis': makeAxis(chosenCats[1], minZoom[1]),
                'caxis': makeAxis(chosenCats[2], minZoom[2])
            },
            'legend': {'x': 0.65, 'y': 0.9}
        }
    }
    return figure

# sample data sender (HARD)
@app.callback(
    dash.dependencies.Output('selected-sample', 'children'),
    [dash.dependencies.Input('graph-ternarySamples', 'clickData')],
    [dash.dependencies.State('intermediate-value', 'children')])
def getSelectedData(clickData, intermediate):
    data = json.loads(intermediate)
    # click -> {'points': [{'curveNumber': 2, 'pointNumber': 18, 'customdata': 'news-Baseball', 'a': 35.965673392081925, 'b': 29.978185948456115, 'c': 34.046140659461955, 'text': '104359'}]}
    # wData -> wData[class][0(probList)], last classes are sampleText(list of words), sampleName
    #                      # first value is prior, last is probability sum of logs
    wData = data[clickData['points'][0]['customdata']]['testSampsProbs'][re.search(r'\d+', clickData['points'][0]['text']).group()]            # probs (search for first number sample id)
    wData['sampleText'] = data[clickData['points'][0]['customdata']]['testSamps'][re.search(r'\d+', clickData['points'][0]['text']).group()]   # words
    wData['sampleName'] = clickData['points'][0]['text']
    return json.dumps(wData)

# update rangeslider of words based on selected sample
@app.callback(
    dash.dependencies.Output('slider-wordRange', 'value'),
    [dash.dependencies.Input('selected-sample', 'children')])
def getWordSum(wData):
    data = json.loads(wData)
    return [0, len(data['sampleText'])]

# rangeslider of words (setting max value because 100 is default)
@app.callback(
    dash.dependencies.Output('slider-wordRange', 'max'),
    [dash.dependencies.Input('selected-sample', 'children')])
def getWordSum(wData):
    data = json.loads(wData)
    return len(data['sampleText'])

# displaying ternary graph of words
@app.callback(
    dash.dependencies.Output('graph-ternaryWords', 'figure'),
    [dash.dependencies.Input('selected-sample', 'children'),
     dash.dependencies.Input('slider-wordRange', 'value')])
def displayWordGraph(wData, wRange):
    data = json.loads(wData)        # ['class1', 'class2', 'class3', 'sampleText', 'sampleName']

    # getting right data format for visualization
    chosenCats = []                 # discovering selected categories (no need another input for callback) and class with biggest probability for color setting
    minZoom = [33, 33, 33]          # min probability values for visualization with zoom
    wRawProbs = []                  # sum class log values
    wPercProbs = []                 # wPercProbs[0(prob list of words)][classProb]

    for i in data:
        chosenCats.append(i)                    # chosenCats['class1', 'class2', 'class3']

    for i in range(1+wRange[0], 1+wRange[1]):   # +1 for prior ignore
        wPercProbs.append(probMap([data[chosenCats[0]][i], data[chosenCats[1]][i], data[chosenCats[2]][i]])) # first value data[chosenCats[0]][i] is prior, last is probability sum of logs
        wRawProbs.append(         [data[chosenCats[0]][i], data[chosenCats[1]][i], data[chosenCats[2]][i]])
    sumRawProbs = [0, 0, 0]

    for i in range(0, len(wRawProbs)):
        sumRawProbs[0] += wRawProbs[i][0]
        sumRawProbs[1] += wRawProbs[i][1]
        sumRawProbs[2] += wRawProbs[i][2]
    priorPerc = probMap([data[chosenCats[0]][0], data[chosenCats[1]][0], data[chosenCats[2]][0]], prior = True)
    nPriorProb = probMap(list(sumRawProbs))
    yPriorProb = probMap([sumRawProbs[0] + data[chosenCats[0]][0], 
                          sumRawProbs[1] + data[chosenCats[1]][0], 
                          sumRawProbs[2] + data[chosenCats[2]][0]])

    maxPiC = [-1E6, 0, '']                          # prob, index, className
    for i in range(0, 3):                           # finding max probability class for color index and name for legend
        if yPriorProb[i] > maxPiC[0]:
            maxPiC[0] = yPriorProb[i]
            maxPiC[1] = i
            maxPiC[2] = chosenCats[i]

    for i in range(0, (wRange[1]-wRange[0])):       # finding zoom word outliers
        for j in range(0, 3):
            if wPercProbs[i][j] < minZoom[j]:
                minZoom[j] = wPercProbs[i][j]

    for i in range(0, 3):                           # dont miss general predictions out of zoom
        if priorPerc[i] < minZoom[i]:
            minZoom[i] = priorPerc[i]
        if nPriorProb[i] < minZoom[i]:
            minZoom[i] = nPriorProb[i]
        if yPriorProb[i] < minZoom[i]:
            minZoom[i] = yPriorProb[i]

    for ith, i in enumerate(data['sampleText'][wRange[0]:wRange[1]]):    # word ordering in sample
        data['sampleText'][ith+wRange[0]] = (str(wRange[0]+ith+1) + ". " + i)

    figure = {
        'data': [{ 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[0], wPercProbs)],
            'b': [i for i in map(lambda x: x[1], wPercProbs)],
            'c': [i for i in map(lambda x: x[2], wPercProbs)],
            'text': [i for i in map(lambda x: x, data['sampleText'][wRange[0]:wRange[1]])],
            'marker': {'color': ['red', 'green', 'blue'][maxPiC[1]]},
            'name': maxPiC[2],
        },{
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [priorPerc[0]],
            'b': [priorPerc[1]],
            'c': [priorPerc[2]],
            'text': 'pravdepodobnost prioru',
            'marker': {
                'color': 'black',
                'size': 15,
                'opacity': 0.5
            },
            'name': 'pravdepodobnost prioru',
        },{
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [nPriorProb[0]],
            'b': [nPriorProb[1]],
            'c': [nPriorProb[2]],
            'text': 'vysl. pravdepodobnost bez priorov',
            'marker': {
                'color': 'orange',
                'size': 15,
                'opacity': 0.5
            },
            'name': 'vysl. pravdepodobnost bez priorov',
        },{
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [yPriorProb[0]],
            'b': [yPriorProb[1]],
            'c': [yPriorProb[2]],
            'text': 'vysl. pravdepodobnost s priormi',
            'marker': {
                'color': 'purple',
                'size': 15,
                'opacity': 0.5
            },
            'name': 'vysl. pravdepodobnost s priormi',
        },{ # probability separation lines
            'a': [0, 33],
            'b': [50, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            'showlegend': False
        },{
            'a': [50, 33],
            'b': [0, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            'showlegend': False
        },{
            'a': [50, 33],
            'b': [50, 33],
            'c': [0, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            'showlegend': False
        }],
        'layout': {
            'title': "<b>Pravdepodobnosti slov</b><br>" + "Meno vzorky: " + data['sampleName'],
            'height': 600,
            'width': 600,
            'hoverdistance': 3,
            'ternary': {
                'sum': 100,
                'aaxis': makeAxis(chosenCats[0], minZoom[0]),
                'baxis': makeAxis(chosenCats[1], minZoom[1]),
                'caxis': makeAxis(chosenCats[2], minZoom[2])
            },
            'legend': {'x': 0.65, 'y': 0.9}
        }
    }
    return figure

# displaying range selection of words
@app.callback(
    dash.dependencies.Output('div-wordRange', 'children'),
    [dash.dependencies.Input('slider-wordRange', 'value')])
def displayRangePerc(wRange):
    wRange[0] += 1
    return 'Selekcia slov vzorky {}'.format(wRange)

# display text according to range selection
@app.callback(
    dash.dependencies.Output('div-sampleText', 'children'),
    [dash.dependencies.Input('slider-wordRange', 'value'),
     dash.dependencies.Input('selected-sample', 'children')])
def displayText(wRange, wData):
    data = json.loads(wData)
    wString = ''
    for i in range(wRange[0], wRange[1]):
        wString += data['sampleText'][i] + " "
    return wString

# display probability calculation process with selected words
@app.callback(
    dash.dependencies.Output('graph-process', 'figure'),
    [dash.dependencies.Input('slider-wordRange', 'value'),
     dash.dependencies.Input('selected-sample', 'children'),
     dash.dependencies.Input('slider-chunkSum', 'value')])
def displayProcess(wRange, wData, chunk):
    data = json.loads(wData)        # ['class1', 'class2', 'class3', 'sampleText', 'sampleName']
    chosenCats = []         # discovering selected categories (no need another input for callback) and class with biggest probability for color setting
    wRawProbs = []          # sum class probabilities for visualization
    wPercProbs = []         # wPercProbs[0(prob list of words)][classProb]
    wSumProbs = []
    dotSums = []
    dotWords = []

    for i in data:
        chosenCats.append(i)            # chosenCats['class1', 'class2', 'class3']
    sumRawProbs = np.array([.0, .0, .0])
    for ith, i in enumerate(range(1+wRange[0], 1+wRange[1])):   # +1 for prior ignore
        wPercProbs.append(probMap([data[chosenCats[0]][i], data[chosenCats[1]][i], data[chosenCats[2]][i]])) # first value data[chosenCats[0]][i] is prior, last is probability sum of logs
        wRawProbs.append(         [data[chosenCats[0]][i], data[chosenCats[1]][i], data[chosenCats[2]][i]])
        sumRawProbs += wRawProbs[ith]
        wSumProbs.append(probMap(np.copy(sumRawProbs)))
        if ith % chunk == 0 and ith != 0:               # we dont want first dot!
            dotSums.append(wSumProbs[ith])              # need to save for big dots
            dotWords.append(str(i) + ". " + data['sampleText'][i-1])    # saving words (x coords)
            sumRawProbs = np.array([.0, .0, .0])

    for ith, i in enumerate(data['sampleText']):
        data['sampleText'][ith] = (str(ith+1) + ". " + i)

    # need index, value
    maxPiC = []
    for cValues in dotSums:
        maxVal = 0
        for ith, value in enumerate(cValues):
            if value > maxVal:
                maxInd = ith
                maxVal = value
        maxPiC.append([maxVal, maxInd])
    
    figure={
        'data': [{
            'x': data['sampleText'][wRange[0]:wRange[1]],
            'y': [i for i in map(lambda x: x[0], wSumProbs)],
            'name': "Sum: " + chosenCats[0],
            'line': {
                'color': 'red',
                'width': 4,
            }
        },{
            'x': data['sampleText'][wRange[0]:wRange[1]],
            'y': [i for i in map(lambda x: x[1], wSumProbs)],
            'name': "Sum: " + chosenCats[1],
            'line': {
                'color': 'green',
                'width': 4,
            }
        },{
            'x': data['sampleText'][wRange[0]:wRange[1]],
            'y': [i for i in map(lambda x: x[2], wSumProbs)],
            'name': "Sum: " + chosenCats[2],
            'line': {
                'color': 'blue',
                'width': 4,
            }
        },{
            'x': data['sampleText'][wRange[0]:wRange[1]],
            'y': [i for i in map(lambda x: x[0], wPercProbs)],
            'name': "Word: " + chosenCats[0],
            'line': {
                'color': 'red',
                'width': 1,
                'dash': 'dash'          # dot / dashdot
            },
            'opacity': 0.5
        },{
            'x': data['sampleText'][wRange[0]:wRange[1]],
            'y': [i for i in map(lambda x: x[1], wPercProbs)],
            'name': "Word: " + chosenCats[1],
            'line': {
                'color': 'green',
                'width': 1,
                'dash': 'dash'
            },
            'opacity': 0.5
        },{
            'x': data['sampleText'][wRange[0]:wRange[1]],
            'y': [i for i in map(lambda x: x[2], wPercProbs)],
            'name': "Word: " + chosenCats[2],
            'line': {
                'color': 'blue',
                'width': 1,
                'dash': 'dash'
            },
            'opacity': 0.5
        },{
            'x': dotWords,
            'y': [i for i in map(lambda x: x[0], maxPiC)],
            'mode': 'markers',
            'marker': {
                'size': '20',
                'color': [i for i in map(lambda x: ['red', 'green', 'blue'][x[1]], maxPiC)]
            },
            'hoverinfo': 'skip'
        }],
        'layout': {
            'title': "<b>Klasifikacny proces pravdepodobnosti</b><br>Meno vzorky: " + data['sampleName'],
            'xaxis': {'rangeslider': {}},
            'yaxis': {'title': 'Probability'}
        }
    }
    return figure

# displaying reset points index for probability summaries
@app.callback(
    dash.dependencies.Output('div-sumRange', 'children'),
    [dash.dependencies.Input('slider-chunkSum', 'value')])
def displayRangePerc(wcRange):
    return '{} - Resetovy bod sumarizacnych pravdepodobnosti'.format(wcRange)

# display uniqueness of words graph for each class
@app.callback(
    dash.dependencies.Output('div-wordImportances3', 'children'),
    [dash.dependencies.Input('slider-wordRange', 'value'),
     dash.dependencies.Input('selected-sample', 'children')])
def displayWordImp(wRange, wData):
    data = json.loads(wData)        # ['class1', 'class2', 'class3', 'sampleText', 'sampleName']
    chosenCats = []
    wPercProbs = []
    maxDiff = {}                    # maxDiff[0,1,2<classes>][diffVal]
    wCount = {}

    for i in data:
        chosenCats.append(i)
        maxDiff[i] = []
        wCount[i] = []
    for ith, i in enumerate(range(1+wRange[0], 1+wRange[1])):
        wPercProbs.append(probMap([data[chosenCats[0]][i], data[chosenCats[1]][i], data[chosenCats[2]][i]]))
        
        # now we want to discover which value has the biggest prob and his minMax difference
        maxIVW = [0, '', 0]                      # index, value, word
        secMax = 0
        for jth, j in enumerate(wPercProbs[ith]):
            if j > maxIVW[0]:
                secMax = maxIVW[0]              # save sec. max value
                maxIVW[0] = j                   # save max value
                index = jth                     # save class index
            else:
                if secMax == 0 or secMax < j:
                    secMax = j

        maxIVW[0] = maxIVW[0] - secMax
        maxIVW[1] = data['sampleText'][i-1]
        maxIVW[2] = 1
        if maxIVW[1] in map(lambda x: x[1], maxDiff[chosenCats[index]]):    # check if this word is already in maxDiff
            for ith, i in enumerate(maxDiff[chosenCats[index]]):            # if yes, we need to find index of it
                if maxIVW[1] == i[1]:
                    maxDiff[chosenCats[index]][ith][2] += 1                 # add another appearance
            continue
        if maxIVW[0] != 0:
            maxDiff[chosenCats[index]].append(maxIVW)                       # if no, we append new word with occurence of 1

    return html.Div([
        html.Div([dcc.Markdown('### Jedinecnost slov pre kazdu kategoriu')],
            style={
                'display': 'inline-block',
                'marginLeft': '15px',
            }
        ),
        html.Abbr("[?]", title="Percentualny rozdiel medzi najvyssou a druhou najvyssou kategorickou pravdepodobnostou slova\n" +
                "   - frekvencie slov vo vzorke su indikovane v () zatvorkach",
            style={
                'display': 'inline-block',
                'marginLeft': '15px',
                'cursor': 'help',
                'text-decoration': 'none'
            }
        ),
        html.Br(),

        dcc.Graph(
            id='imp1',
            figure={
                'data': [{
                    'x': [i for i in map(lambda x: str(x[1] + "(" + str(x[2]) + ")"), sorted(maxDiff[chosenCats[0]], reverse=True))],
                    'y': [i for i in map(lambda x: x[0], sorted(maxDiff[chosenCats[0]], reverse=True))],
                    'marker': {'color': 'red' },
                    'type': 'bar',
                }],
                'layout': {
                    'title': chosenCats[0],
                    'xaxis': {'rangeslider': {}},
                }
            },
            style={
                'width': 425,
                'height': 500,
                'display': 'inline-block',
            }
        ),
        dcc.Graph(
            id='imp2',
            figure={
                'data': [{
                    'x': [i for i in map(lambda x: str(x[1] + "(" + str(x[2]) + ")"), sorted(maxDiff[chosenCats[1]], reverse=True))],
                    'y': [i for i in map(lambda x: x[0], sorted(maxDiff[chosenCats[1]], reverse=True))],
                    'name': chosenCats[1],
                    'marker': {'color': 'green' },
                    'type': 'bar',
                }],
                'layout': {
                    'title': chosenCats[1],
                    'xaxis': {'rangeslider': {}},
                }
            },
            style={
                'width': 425,
                'height': 500,
                'display': 'inline-block',
            },
        ),
        dcc.Graph(
            id='imp3',
            figure={
                'data': [{
                    'x': [i for i in map(lambda x: str(x[1] + "(" + str(x[2]) + ")"), sorted(maxDiff[chosenCats[2]], reverse=True))],
                    'y': [i for i in map(lambda x: x[0], sorted(maxDiff[chosenCats[2]], reverse=True))],
                    'name': chosenCats[2],
                    'marker': {'color': 'blue' },
                    'type': 'bar',
                }],
                'layout': {
                    'title': chosenCats[2],
                    'xaxis': {'rangeslider': {}}
                }
            },
            style={
                'width': 425,
                'height': 500,
                'display': 'inline-block',
            },
        ),
    ],
    style={'textAlign': 'center',})

# add space for classification alternative with more than 3 categories
@app.callback(
    dash.dependencies.Output('moreCatsSpace', 'children'),
    [dash.dependencies.Input('dropdown-categorySelection', 'value')])
def addSpace(chosenCats):
    if len(chosenCats) > 3:
        return html.Div([
            html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
            html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
        ])
    else:
        return

# classifying and sending data for other callbacks
@app.callback(
    dash.dependencies.Output('intermediate-value', 'children'),
    [dash.dependencies.Input('button-classify', 'n_clicks')],
    [dash.dependencies.State('dropdown-categorySelection', 'value'),
     dash.dependencies.State('radio-zeroFixSelection', 'value'),
     dash.dependencies.State('rangeSlider-rationalNum', 'value')])
def updateMultipleOutputs(n_clicks, chosenCategories, radio, fixNum):
    # creating new local priors, likelihood and content corresponding to chosen categories
    newPriors = {}; newLikelihood = {}; newContent = {}
    for i in chosenCategories:
        newPriors[i] = priors[i]                    # priors[className]                 {counted}
        newLikelihood[i] = likelihood[i]            # likelihood[className][word]       {counted}
        newContent[i] = content[i]                  # content[className][sampleName][words(list)]

    if radio == 'RTN':
        accuracy, skewInfo, contProb = cf.classifyTestSet(newLikelihood, newPriors, newContent, 1/(10 ** fixNum[0]))
    else:
        accuracy, skewInfo, contProb = cf.classifyTestSet(newLikelihood, newPriors, newContent, 1) # Laplace smoothing
    metricsReport = cf.calcReport(skewInfo)         # metricsReport -> [support/TN/FN/TP/FP] 
    metricsReportAVG = metricsReport['Avg/total']

    data = {}
    for i in chosenCategories:                      # data[className]
        data[i] = {'sampleAmount': priors[i],       # data[className]['sampleAmount']                       {counted}
                    'wordFreqs': likelihood[i],     # data[className]['wordFreqs']      ['<word>s']         {counted}
                    'testSamps': content[i],        # data[className]['testSamps']      [sampleName][words(list)]
                    'testSampsProbs': contProb[i],  # data[className]['testSampsProbs'] [sampleName][className][2d(row=cat, col=word), minP]
                    'report': metricsReport[i]}     # data[className]['report']         [prec,rec..]
    data['Avg/total'] = metricsReport['Avg/total']
    data['accuracy'] = accuracy

    return json.dumps(data)

# displaying overall accuracy
@app.callback(
    dash.dependencies.Output('div-accuracy', 'children'),
    [dash.dependencies.Input('intermediate-value', 'children')])
def displayAccuracy(intermediate):
    data = json.loads(intermediate)
    return str(data['accuracy'][0]) + " spravnych klasifikacii z celkovych " + str(data['accuracy'][1]) + " vzoriek\nUspesnost: " + str(round(data['accuracy'][0] / data['accuracy'][1], 3))

if __name__ == '__main__':
    app.run_server(debug=True)