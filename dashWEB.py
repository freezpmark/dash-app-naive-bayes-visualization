import dash, pickle, jsonpickle, json, re
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table_experiments as dt
import pandas as pd
import numpy as np
import dashCLASSIFIER as cf

stopWord = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                                    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def nestFunc():
    return coll.defaultdict(list)

# calculate percentual probability from classification values
def probMap(probList, prior = False):
    avgPerc = sum(probList)/3
    if prior:
        for ith, i in enumerate(probList):          # calc % for positive values
            probList[ith] = (i * 33.33) / avgPerc
    else:                                           # calc % for negative values
        for ith, i in enumerate(probList):
            probList[ith] = (((2*avgPerc) - i) * 33.33) / avgPerc
    return probList

with open('dash-likelihood', 'rb') as handle:
    likelihood = pickle.loads(handle.read())            # likelihood[className][word]   {counted}
with open('dash-priors', 'rb') as handle:
    priors = pickle.loads(handle.read())                # prior[className]              {counted}
with open('dash-content', 'rb') as handle:
    content = pickle.loads(handle.read())               # content[className][sampleName][words(list)]

cPriors = {}; cLikelihood = {}; cContent = {}
app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#000000'
}

def makeAxis(title, minPerc): 
    return {
      'min': minPerc,
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': 0,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(20,40,100,50)',
      'ticklen': 5,
      'showgrid': True
    }

def transform_value(value):
    return 10 ** value

# Edit the layout
app.layout = html.Div(children=[
    # Title of application
    html.H1('Naive Bayesian Classifier',
        style={
            'textAlign': 'center',
            'color': colors['text'],
        }
    ),
    html.Abbr("???", title="Hello, I am hover-enabled helpful information\nAAAA."),

    html.Div(
        [
            # bar graph of priors
            dcc.Graph(
                id='graph-prior',
                figure={
                    'data': [
                        {
                            'x': [str(category) for category in priors], 
                            'y': [str(priors[category]) for category in priors], 
                            'type': 'bar', 
                            'name': 'Total samples'},
                        {
                            'x': [str(category) for category in likelihood],
                            'y': [str(sum(likelihood[category].values())) for category in likelihood],
                            'type': 'bar',
                            'name': 'Total words',
                            'visible': 'legendonly'
                        },
                        {
                            'x': [str(category) for category in likelihood],
                            'y': [str(sum(likelihood[category].values())/priors[category]) for category in likelihood],
                            'type': 'bar',
                            'name': 'Avg words per sample',
                            'visible': 'legendonly'
                        },
                    ],
                    'layout': {
                        'title': 'Number of samples and words in each category'
                    }
                }
            ),

            # dropdown from which we can choose categories to classify
            dcc.Dropdown(
                id='dropdown-categorySelection',
                options= [{'label': str(category),'value': str(category)} for category in priors],
                placeholder="Select three categories for classification",
                value=['news-Graphics', 'news-Forsale', 'news-Baseball'],
                multi=True
            ),

            # dropdown from which we can choose category with preferred word frequencies
            dcc.Dropdown(
                id='dropdown-categoryPreference',
                options=[{'label': str(category),'value': str(category)} for category in priors],
                placeholder="Select one category to see most frequent used words",
                searchable=False
            ),

            # tabs for graph of word frequencies
            dcc.Tabs(
                tabs=[{'label': 'Tab {}'.format(i), 'value': i} for i in range(1, 3)],
                value=2,
                id='tabs-select'
            ),

            # graph of word frequencies
            html.Div(id='div-graph-bar'),

            dcc.RadioItems(
                id = 'radio-wordPreference',
                options=[
                    {'label': 'Word frequency', 'value': 'wFreq'},
                    {'label': 'Word importance', 'value': 'wImp'},
                ],
                value='wFreq'
            ),

            # radio for zeroFix selection (rational number / laplace smoothing) 
            dcc.RadioItems(
                id = "radio-zeroFixSelection",
                options=[
                    {'label': 'Replace number:', 'value': 'RTN'},
                    {'label': 'Laplace smoothing', 'value': 'LAP'},
                ],
                value='RTN',
                labelStyle={'display': 'inline-block'}
            ),

            # rangeSlider for rational number selection
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

            # exact value of rational number
            html.Div(id='div-outputZeroF'),

            html.Br(),
            html.Br(),

            html.Button('Classify', id='button-classify'),

            # classification report of 3 chosen categories
            html.H4('Classification report'),
            dt.DataTable(
                rows = [{},{},{}],
                #row_selectable=True,
                #filterable=True,
                sortable=True,
                selected_row_indices=[],
                id='dataTable-scoreMetrics'
            ),
            
            # exact accuracy value
            html.Div(id='div-accuracy', style={'color': 'green'}),

        ],
        style={
            'width': '75%',
            'fontFamily': 'Sans-Serif',
            'margin-left': 'auto',
            'margin-right': 'auto'
        }
    ),

    dcc.Graph(id='graph-ternarySamples'),
    dcc.Graph(id='graph-ternaryWords'),

    # selected text with range selection
    html.Div(id='div-sampleText'),

    html.Hr(),          # line
    html.Br(),          # space

    # range picker of words
    dcc.RangeSlider(
        id='slider-wordRange',
        step=1,
    ),
    html.Div(id='div-wordRange'),
    
    html.Br(),
    html.Hr(),

    # graph of probability calculations process with selected words
    dcc.Graph(id='graph-process'),

    html.Br(),
    html.Br(),

    # range slider for selection of chunk sums
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
    ),
    html.Div(id='div-sumRange'),

    html.Br(),
    html.Br(),
    # graph for each class of most important word probability influences
    html.Div(id='div-wordImportances3'),


    # preprocessing
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id ='selected-sample', style={'display': 'none'}),
])





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
    secCats = [category for category in chosenCategories if category != preferedCategory] # secondary categories are chosen categories without preferedCategory
    newLikelihood1 = dict(likelihood[preferedCategory])          # assigning global list into new local one without id reference via list()

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
    sortedValues = sorted(newLikelihood1.items(), key=lambda x: x[1], reverse=True)    # save number of word frequencies in sorted order
    sortedWords = [word[0] for word in sortedValues[:150]]      # save first 150 most frequent words 

    if wordPref == 'wImp':
        newLikelihood2 = {}
        newLikelihood3 = {}
        n1 = float(sum(likelihood[preferedCategory].values()))
        n2 = float(sum(likelihood[secCats[0]].values()))
        n3 = float(sum(likelihood[secCats[1]].values()))
        for i in sortedValues:
            newLikelihood1[i[0]] = (newLikelihood1[i[0]] / n1)
            newLikelihood2[i[0]] = (likelihood[secCats[0]][i[0]] / n2)
            newLikelihood3[i[0]] = (likelihood[secCats[1]][i[0]] / n3)
        tmpData = [{
            'x': sortedWords,
            'y': [newLikelihood1[x] for x in sortedWords],
            'name': preferedCategory,
            'marker': { 'color': ['red', 'green', 'blue'][colors[0]]},
            'type': ['bar', 'scatter'][int(tabSelection) % 2]
        },{
            'x': sortedWords,
            'y': [newLikelihood2[x] for x in sortedWords],
            'name': secCats[0],
            'marker': { 'color': ['red', 'green', 'blue'][colors[1]]},
            'type': ['bar', 'bar'][int(tabSelection) % 2]
        },{
            'x': sortedWords,
            'y': [newLikelihood3[x] for x in sortedWords],
            'name': secCats[1],
            'marker': {'color': ['red', 'green', 'blue'][colors[2]]},
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
                    'legend': {'x': 0, 'y': 1},
                    'xaxis': {
                        'rangeselector': {
                            'buttons': 'step'
                        },
                        'rangeslider': {}
                    },
                    
                    'type': 'date'
                }
            }
        ),
    

app.config.supress_callback_exceptions = True

# displaying exact zero fix rational number
@app.callback( #component_id, component_property
    Output('div-outputZeroF', 'children'),
    [Input('rangeSlider-rationalNum', 'value'),
     Input('radio-zeroFixSelection', 'value')])
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
                    'F1': data[category]['report'][3],
                    'Support': data[category]['report'][4]} for category in data if category != 'Avg/total' and category != 'accuracy']
    metrics.append({'Category': 'Avg/total', 'Precision': data['Avg/total'][0], 'Recall': data['Avg/total'][1], 'Average': data['Avg/total'][2], 'F1': data['Avg/total'][3], 'Support': data['Avg/total'][4]})
    return metrics

# displaying ternary graph of samples
@app.callback(
    dash.dependencies.Output('graph-ternarySamples', 'figure'),
    [dash.dependencies.Input('intermediate-value', 'children')])
def displaySampleGraph(intermediate):
    data = json.loads(intermediate) # data[className]['testSampsProbs'] [sampleName][className(vyp pravdepodobnosti pre vsetky classy)][2d-row=cat, col=word na konci vysl p, minP]
    data.pop('Avg/total', None)
    data.pop('accuracy', None)

    # getting right data format for visualization
    vizData = {}                    # vizData['class(sampleBelonging)] [0(list of samples)] [probs of 3 chosen categories, sampleLabel]
    chosenCats = []                 # discovering selected categories (no need another input for callback)
    minZoom = [33, 33, 33]          # min probability values for visualization with zoom
    for i in data:                                          # i actual className
        chosenCats.append(i)
        vizData[i] = []
        for jth, j in enumerate(data[i]['testSampsProbs']): # j sample name
            vizData[i].append({})
            scProbs = []                                    # sample class probabilities
            for k in data[i]['testSampsProbs'][j]:          # k predicted className 
                scProbs.append(data[i]['testSampsProbs'][j][k][-1]) # -1th value is overall probability of class k
            percProbs = probMap(scProbs)                         # mapping log probabilities of sample into percentual probabilities
            for kth, k in enumerate(data[i]['testSampsProbs'][j]):
                vizData[i][jth][k] = percProbs[kth]                                         # IGNORE minProb!
                if vizData[i][jth][k] < minZoom[kth]:
                    minZoom[kth] = vizData[i][jth][k]
            vizData[i][jth]['label'] = j
            vizData[i][jth]['size'] = len(data[i]['testSamps'][j])  # word count
    
    figure = {
        'data': [{ 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[chosenCats[0]], vizData[chosenCats[0]])],
            'b': [i for i in map(lambda x: x[chosenCats[1]], vizData[chosenCats[0]])],
            'c': [i for i in map(lambda x: x[chosenCats[2]], vizData[chosenCats[0]])],
            'text': [i for i in map(lambda x: x['label'] + ", words: " + str(x['size']), vizData[chosenCats[0]])],
            'marker': {
                'color': 'red',
            },
            'name': chosenCats[0],
            'customdata': [chosenCats[0] for i in range(0, len(vizData[chosenCats[0]]))]
        },
        { 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[chosenCats[0]], vizData[chosenCats[1]])],
            'b': [i for i in map(lambda x: x[chosenCats[1]], vizData[chosenCats[1]])],
            'c': [i for i in map(lambda x: x[chosenCats[2]], vizData[chosenCats[1]])],
            'text': [i for i in map(lambda x: x['label'] + ", words: " + str(x['size']), vizData[chosenCats[1]])],
            'marker': {
                'color': 'green',
            },
            'name': chosenCats[1],
            'customdata': [chosenCats[1] for i in range(0, len(vizData[chosenCats[1]]))]
        },
        
        { 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[chosenCats[0]], vizData[chosenCats[2]])],
            'b': [i for i in map(lambda x: x[chosenCats[1]], vizData[chosenCats[2]])],
            'c': [i for i in map(lambda x: x[chosenCats[2]], vizData[chosenCats[2]])],
            'text': [i for i in map(lambda x: x['label'] + ", words: " + str(x['size']), vizData[chosenCats[2]])],
            'marker': {
                'color': 'blue',
            },
            'name': chosenCats[2],
            'customdata': [chosenCats[2] for i in range(0, len(vizData[chosenCats[2]]))]
        },

        # probability separation lines
        {
            'a': [0, 33],
            'b': [50, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary'
        },
        {
            'a': [50, 33],
            'b': [0, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary'
        },
        {
            'a': [50, 33],
            'b': [50, 33],
            'c': [0, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary'
        }],

        'layout': {
            'autosize': False,
            'width': 750,
            'height': 750,
            'hoverdistance': 3,
            'ternary': {
                'sum': 100,
                'aaxis': makeAxis(chosenCats[0], minZoom[0]),
                'baxis': makeAxis(chosenCats[1], minZoom[1]),
                'caxis': makeAxis(chosenCats[2], minZoom[2])
            },
                #'annotations': [{
                #    'showarrow': False,
                #    'text': 'Simple Ternary Plot with Markers',
                #    'x': 0.5,
                #    'y': 1.3,
                #    'font': { 'size': 15 }
                #}]
        },
    }
    return figure



# sample data sender
@app.callback(
    dash.dependencies.Output('selected-sample', 'children'),
    [dash.dependencies.Input('graph-ternarySamples', 'clickData')],
    [dash.dependencies.State('intermediate-value', 'children')])
def getSelectedData(clickData, intermediate):
    #if intermediate == None:
    #    return
    data = json.loads(intermediate)
    # click -> {'points': [{'curveNumber': 2, 'pointNumber': 18, 'customdata': 'news-Baseball', 'a': 35.965673392081925, 'b': 29.978185948456115, 'c': 34.046140659461955, 'text': '104359'}]}
    # wData -> wData[class][0(probList)], last classes are sampleText(list of words), sampleName
    #                      # first value is prior, last is probability sum of logs
    wData = data[clickData['points'][0]['customdata']]['testSampsProbs'][re.search(r'\d+', clickData['points'][0]['text']).group()]            # probs (search for first number sample id)
    wData['sampleText'] = data[clickData['points'][0]['customdata']]['testSamps'][re.search(r'\d+', clickData['points'][0]['text']).group()]   # words
    wData['sampleName'] = clickData['points'][0]['text']

    return json.dumps(wData)

# rangeslider of words (no percentages)
@app.callback(
    dash.dependencies.Output('slider-wordRange', 'value'),
    [dash.dependencies.Input('selected-sample', 'children')])
def getWordSum(wData):
    data = json.loads(wData)
    return [0, len(data['sampleText'])]

# rangeslider of words (setting max value, because 100 is default)
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
    #if wData == None:
    #    return
    data = json.loads(wData)        # ['class1', 'class2', 'class3', 'sampleText', 'sampleName']

    # getting right data format for visualization
    chosenCats = []                 # discovering selected categories (no need another input for callback) and class with biggest probability for color setting
    minZoom = [33, 33, 33]          # min probability values for visualization with zoom
    wRawProbs = []                  # sum class probabilities for visualization
    wPercProbs = []                 # wPercProbs[0(prob list of words)][classProb]
    flag1 = wRange[0]
    flag2 = wRange[1]

    for i in data:
        chosenCats.append(i)            # chosenCats['class1', 'class2', 'class3']
    
    for i in range(1+flag1, 1+flag2):   # +1 for prior ignore
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

    for i in range(0, (flag2-flag1)):               # finding zoom word outliers
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

    for ith, i in enumerate(data['sampleText'][flag1:flag2]):    # word ordering in sample
        data['sampleText'][ith+flag1] = (str(flag1+ith+1) + ". " + i)

    figure = {
        'data': [{ 
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [i for i in map(lambda x: x[0], wPercProbs)],
            'b': [i for i in map(lambda x: x[1], wPercProbs)],
            'c': [i for i in map(lambda x: x[2], wPercProbs)],
            'text': [i for i in map(lambda x: x, data['sampleText'][flag1:flag2])],
            'marker': {
                'color': ['red', 'green', 'blue'][maxPiC[1]],       # int(tabSelection) % 2
            },
            'name': maxPiC[2],
        },
        {
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [priorPerc[0]],
            'b': [priorPerc[1]],
            'c': [priorPerc[2]],
            'text': 'priors',
            'marker': {
                'color': 'black',       # int(tabSelection) % 2
                'size': 15,
                'opacity': 0.5
            },
            'name': 'priors',
        },
        {
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [nPriorProb[0]],
            'b': [nPriorProb[1]],
            'c': [nPriorProb[2]],
            'text': 'probability without priors',
            'marker': {
                'color': 'orange',       # int(tabSelection) % 2
                'size': 15,
                'opacity': 0.5
            },
            'name': 'probability without priors',
        },
        
        {
            'type': 'scatterternary',
            'mode': 'markers',
            'a': [yPriorProb[0]],
            'b': [yPriorProb[1]],
            'c': [yPriorProb[2]],
            'text': 'probability with priors',
            'marker': {
                'color': 'green',       # int(tabSelection) % 2
                'size': 15,
                'opacity': 0.5
            },
            'name': 'probability with priors',
        },

        # probability separation lines
        {
            'a': [0, 33],
            'b': [50, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            #'hoverinfo': None
        },
        {
            'a': [50, 33],
            'b': [0, 33],
            'c': [50, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            #'hoverinfo': None
        },
        {
            'a': [50, 33],
            'b': [50, 33],
            'c': [0, 33],
            'fillcolor': "#bebada",
            'line': {"color": "black"},
            'mode': 'lines',
            'type': 'scatterternary',
            #'hoverinfo': None
        }],

        'layout': {
            'title': data['sampleName'],
            'autosize': False,
            'width': 750,
            'height': 750,
            'hoverdistance': 3,
            'ternary': {
                'sum': 100,
                'aaxis': makeAxis(chosenCats[0], minZoom[0]),
                'baxis': makeAxis(chosenCats[1], minZoom[1]),
                'caxis': makeAxis(chosenCats[2], minZoom[2])
            },
                #'annotations': [{
                #    'showarrow': False,
                #    'text': 'Simple Ternary Plot with Markers',
                #    'x': 0.5,
                #    'y': 1.3,
                #    'font': { 'size': 15 }
                #}]
        },
    }
    return figure

# displaying range percentage selection of words
@app.callback(
    dash.dependencies.Output('div-wordRange', 'children'),
    [dash.dependencies.Input('slider-wordRange', 'value')])
def displayRangePerc(wRange):
    wRange[0] += 1
    return 'Words selection {}'.format(wRange)

# display text according to range selection
@app.callback(
    dash.dependencies.Output('div-sampleText', 'children'),
    [dash.dependencies.Input('slider-wordRange', 'value'),
     dash.dependencies.Input('selected-sample', 'children')])
def displayText(wRange, wData):
    #if wData == None:
    #    return
    data = json.loads(wData)
    flag1 = wRange[0]
    flag2 = wRange[1]
    wString = ''
    for i in range(flag1, flag2):
        wString += data['sampleText'][i] + " "

    return wString

# display probability calculation process with selected words
@app.callback(
    dash.dependencies.Output('graph-process', 'figure'),
    [dash.dependencies.Input('slider-wordRange', 'value'),
     dash.dependencies.Input('selected-sample', 'children'),
     dash.dependencies.Input('slider-chunkSum', 'value')])
def displayProcess(wRange, wData, chunk):
    #if wData == None:
    #    return
    data = json.loads(wData)        # ['class1', 'class2', 'class3', 'sampleText', 'sampleName']
    # wData -> wData[class][0(probList)], last classes are sampleText(list of words), sampleName
    #                      # first value is prior, last is probability sum of logs
    
    chosenCats = []                 # discovering selected categories (no need another input for callback) and class with biggest probability for color setting
    wRawProbs = []                  # sum class probabilities for visualization
    wPercProbs = []                 # wPercProbs[0(prob list of words)][classProb]
    wSumProbs = []

    dotSums = []
    dotWords = []
    flag1 = wRange[0]
    flag2 = wRange[1]

    for i in data:
        chosenCats.append(i)            # chosenCats['class1', 'class2', 'class3']
    sumRawProbs = np.array([.0, .0, .0])
    for ith, i in enumerate(range(1+flag1, 1+flag2)):   # +1 for prior ignore
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
    colors = ['red', 'green', 'blue'] #####################################
    for cValues in dotSums:
        maxVal = 0
        for ith, value in enumerate(cValues):
            if value > maxVal:
                maxInd = ith
                maxVal = value
        maxPiC.append([maxVal, maxInd])
    
    figure={
        'data': [{
            'x': data['sampleText'][flag1:flag2],
            'y': [i for i in map(lambda x: x[0], wSumProbs)],
            'name': "Sum: " + chosenCats[0],
            'line': {
                'color': 'red',
                'width': 4,
            }
        },
        {
            'x': data['sampleText'][flag1:flag2],
            'y': [i for i in map(lambda x: x[1], wSumProbs)],
            'name': "Sum: " + chosenCats[1],
            'line': {
                'color': 'green',
                'width': 4,
            }
        },
        {
            'x': data['sampleText'][flag1:flag2],
            'y': [i for i in map(lambda x: x[2], wSumProbs)],
            'name': "Sum: " + chosenCats[2],
            'line': {
                'color': 'blue',
                'width': 4,
            },
        },
        {
            'x': data['sampleText'][flag1:flag2],
            'y': [i for i in map(lambda x: x[0], wPercProbs)],
            'name': "Word: " + chosenCats[0],
            'line': {
                'color': 'red',
                'width': 1,
                'dash': 'dash'          # dot / dashdot
            },
            'opacity': 0.5
        },
        {
            'x': data['sampleText'][flag1:flag2],
            'y': [i for i in map(lambda x: x[1], wPercProbs)],
            'name': "Word: " + chosenCats[1],
            'line': {
                'color': 'green',
                'width': 1,
                'dash': 'dash'
            },
            'opacity': 0.5
        },
        {
            'x': data['sampleText'][flag1:flag2],
            'y': [i for i in map(lambda x: x[2], wPercProbs)],
            'name': "Word: " + chosenCats[2],
            'line': {
                'color': 'blue',
                'width': 1,
                'dash': 'dash'
            },
            'opacity': 0.5
        },
        {
            'x': dotWords,
            'y': [i for i in map(lambda x: x[0], maxPiC)],
            'mode': 'markers',
            'marker': {
                'size': '20',
                'color': [i for i in map(lambda x: colors[x[1]], maxPiC)]
            },
            'hoverinfo': 'skip'
        },
        ],
        'layout': {
            'title': data['sampleName'],
            'xaxis': {
                'title': 'Words',
                'rangeselector': {
                    'buttons': 'step'
                },
                'rangeslider': {}
            },
            'yaxis': {
                'title': 'Probability'
            },
            'rangeslider': {},
        }
    }
    return figure


# displaying chunk of words amount selection for separating summarization
@app.callback(
    dash.dependencies.Output('div-sumRange', 'children'),
    [dash.dependencies.Input('slider-chunkSum', 'value')])
def displayRangePerc(wcRange):
    return '{} - word chunks'.format(wcRange)



# display graph for each class of most important word probability influences (not multiple words)
@app.callback(
    dash.dependencies.Output('div-wordImportances3', 'children'),
    [dash.dependencies.Input('slider-wordRange', 'value'),
     dash.dependencies.Input('selected-sample', 'children')])
def displayWordImp(wRange, wData):
    #wRange - [0, 10] - first 10 words 0-9
    data = json.loads(wData)        # ['class1', 'class2', 'class3', 'sampleText', 'sampleName']
    # wData -> wData[class][0(probList)], last classes are sampleText(list of words), sampleName
    #                      # first value is prior, last is probability sum of logs
    chosenCats = []
    wPercProbs = []
    maxDiff = {}            # maxDiff[0,1,2<classes>][diffVal]
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
        maxIVW[1] = data['sampleText'][i-1]#(str(i) + ". " + data['sampleText'][i-1])
        maxIVW[2] = 1
        if maxIVW[1] in map(lambda x: x[1], maxDiff[chosenCats[index]]):        # check if this word is already in maxDiff
            for ith, i in enumerate(maxDiff[chosenCats[index]]):                # if yes, we need to find index of it
                if maxIVW[1] == i[1]:
                    maxDiff[chosenCats[index]][ith][2] += 1                     # add another appearance
            continue
        if maxIVW[0] != 0:
            maxDiff[chosenCats[index]].append(maxIVW)                               # if no, we append new word with occurence of 1

    return html.Div([
        dcc.Graph(
            id='imp1',
            figure={
                'data': [{
                    'x': [i for i in map(lambda x: str(x[1] + "(" + str(x[2]) + ")"), sorted(maxDiff[chosenCats[0]], reverse=True))],
                    'y': [i for i in map(lambda x: x[0], sorted(maxDiff[chosenCats[0]], reverse=True))],
                    'name': chosenCats[0],
                    'marker': {'color': 'red' },
                    'type': 'bar',
                }],
                'layout': {
                    'xaxis': {
                        'rangeslider': {}
                    }
                }
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
                    'xaxis': {
                        'rangeslider': {}
                    }
                }
            }
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
                    'xaxis': {
                        'rangeslider': {}
                    }
                }
            }
        ),
    ])


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
        newPriors[i] = priors[i]                      # priors[className]                 {counted}
        newLikelihood[i] = likelihood[i]              # likelihood[className][word]       {counted}
        newContent[i] = content[i]                    # content[className][sampleName][words(list)]

    if radio == 'RTN':
        accuracy, skewInfo, contProb = cf.classifyTestSet(newLikelihood, newPriors, newContent, 1/(10 ** fixNum[0]))
    else:
        accuracy, skewInfo, contProb = cf.classifyTestSet(newLikelihood, newPriors, newContent, 1) # Laplace smoothing
    metricsReport = cf.calcReport(skewInfo)      # metricsReport -> [support/TN/FN/TP/FP] 
    metricsReportAVG = metricsReport['Avg/total']

    data = {}
    for i in chosenCategories:                            # data[className]
        data[i] = {'sampleAmount': priors[i],       # data[className]['sampleAmount']                       {counted}
                    'wordFreqs': likelihood[i],     # data[className]['wordFreqs']      ['<word>s']         {counted}
                    'testSamps': content[i],        # data[className]['testSamps']      [sampleName][words(list)]
                    'testSampsProbs': contProb[i],  # data[className]['testSampsProbs'] [sampleName][className][2d(row=cat, col=word), minP]
                    'report': metricsReport[i]}     # data[className]['report']         [prec,rec..]
    data['Avg/total'] = metricsReport['Avg/total']
    data['accuracy'] = accuracy

    return json.dumps(data)         # return dict as string


# displaying overall accuracy
@app.callback(
    dash.dependencies.Output('div-accuracy', 'children'),
    [dash.dependencies.Input('intermediate-value', 'children')])
def displayAccuracy(intermediate):
    data = json.loads(intermediate) # load dict from sent string
    return "Accuracy: " + str(data['accuracy'][0] / data['accuracy'][1]) + "   Samples: " + str(data['accuracy'][1]) + "   Correct predictions: " + str(data['accuracy'][0])

if __name__ == '__main__':
    app.run_server(debug=True)
    