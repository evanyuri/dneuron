#Dash Plotly Libraries
import dash
from dash import Dash, dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
# ML Libraries
import tensorflow as tf
from sklearn.linear_model import LinearRegression
# Data Structure
import pandas as pd
import numpy as np
import base64
import io
import os
from json import JSONEncoder
import json
import pickle
import codecs

app = Dash(__name__, assets_folder="assets", external_stylesheets = [dbc.themes.BOOTSTRAP, 'style.css'], suppress_callback_exceptions=True)
app.title = 'ML It'
app._favicon = ("assets/falvicon.ico")

app.layout = html.Div(style={},
    children=[ 
html.Div(children=[
html.H1(children='Deep Neuron', ),
html.H6(children='Make machine learning and insights easy and for the masses', style={'color':'rgb(240,200,255)'}),

],style={'color':'white','background-image':'linear-gradient(to right, rgb(100,20,150), rgb(150,50,200))', 
'padding':'20px', 'padding-left':'20px'},
),

html.A("Kruchowy.com", href='https://www.kruchowy.com', target="_blank", className='link-1'),

html.Div(className="virus"),
html.Div(style={'border-radius': '20px', 'padding':'30px'},
    children=[

    dbc.Row([
    dbc.Col([
        html.H5(children='1. Upload single header data file (.csv or Excel)'),
        
        dcc.Upload(id='upload-data',children=html.Div(['Drop or ',html.A('Select File')]),multiple=False,className='dropIn',),
        html.A("Or download example weather data to use", href='https://drive.google.com/file/d/1HN4-2rccp-vELPO1BLpJmH92fpH61X62/view?usp=sharing', target="_blank",style={'font-size': '12px','text-decoration': 'none', 'padding':'5px', 'margin-bottom':'10px'}),

    html.H5(children='2. Choose variables', style={'margin-top':'20px'}),
    dcc.Dropdown(id='variables', multi = True, placeholder='Choose Variables',options=[],value=[], style={'margin-bottom': '15px'}),
    html.H5(children='3. Choose output to predict'),
    dcc.Dropdown(id='outputs', multi = False, placeholder='Choose Output',options=[], style={'margin-bottom': '15px'}),
    html.Button('Build!',id ='train-button', className="button-col-3", style={'width': '100%',}),

    ], width=3),
    dbc.Col([
        html.Div(id='results'),
    ], width = 6), 
    dbc.Col([
        html.Div(id='slider-div-parent'),
    ], width = 3), 
    dcc.Store(id='intermediate-value'),
    dcc.Store(id='intermediate-model'),
    dcc.Store(id='intermediate-weights'),


    ]),]),])

@app.callback(
    Output(component_id='variables', component_property='options'),
    Output(component_id='variables', component_property='value'),
    Output(component_id='outputs', component_property='options'),
    Output(component_id='outputs', component_property='value'),
    Output('intermediate-value', 'data'),
    
    Input('upload-data', 'contents',),
    State('upload-data', 'filename'),
    Input('outputs', 'value'),
    State('variables', 'value'),

    prevent_initial_call=True,
)
def ingest_csv(contents, fname, output, vars):

        # try:
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
        else:
            print('Nothing Uploaded')
            return [], [], [], [], []
        if 'csv' in fname:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')),  skiprows=0)
        elif 'xls' in fname:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), skiprows=0,)

        df = df.select_dtypes(include=['float64', 'int64']).astype('float64')
 
        #set initial output value
        if not output:
            output = df.columns[0]
        # Remove output from variable options
        if vars == []:
            vars = df.columns.tolist()
        var_options = df.columns.tolist()
        if output in vars:
            var_options.remove(output)       
        return var_options, vars, df.columns.tolist(), output, df.to_json(date_format='iso', orient='split')
        # except Exception as e:
        #     print(e)
        #     return [], [], [], []

@app.callback(

    Output('results', 'children'),
    Output('intermediate-model', 'data'),
    Output('intermediate-weights', 'data'),

    Input('train-button', 'n_clicks'),
    State('variables', 'value'),
    State('outputs', 'value'),
    State('intermediate-value', 'data'),
)
def build_model(n_clicks, vars, output, df_JSON):
    print(output)
    changed_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if 'train-button' in changed_id:
        df = pd.read_json(df_JSON, orient='split')
        df = df[[output]+ vars]
        df = df.dropna()
        #print(df)
        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(output)
        test_labels = test_features.pop(output)

        normalizer = tf.keras.layers.Normalization(axis=-1, input_dim=len(vars))
        normalizer.adapt(np.array(train_features))

        #build model 
        def build_and_compile_model(norm):
            model = tf.keras.Sequential([
                norm,
                tf.keras.layers.Dense(len(vars), activation='relu', name='layer1'),
                #tf.keras.layers.Dense(len(vars), activation='relu', name='layer2'),
                tf.keras.layers.Dense(1, name='layerFinal')

            ])
            ##Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                        loss='mean_absolute_error',)
                        #metrics=['accuracy'])
            return model

        model = build_and_compile_model(normalizer)

        history = model.fit(
            train_features,
            train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        # print(history.history['loss'])
        # print(history.history['val_loss'])
        losses = pd.DataFrame(
        {'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],})
        fig_losses = go.Figure()
        fig_losses.add_trace(go.Scatter(x = losses.index, y = losses['val_loss'], name='val_loss',line=dict(color='rgb(200,200,200)')))
        fig_losses.add_trace(go.Scatter(x = losses.index, y = losses['loss'], name='loss', line=dict(color='purple')))
        fig_losses.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", hovermode="x", legend=dict(x=0.5,y=0.9))
        fig_losses.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False, title='Epoch')
        fig_losses.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False, title ='Error')

        print(test_features)
        test_predictions = model.predict(test_features).flatten()
        X = test_labels.values.reshape(-1,1)
        LR_model = LinearRegression()
        LR_model.fit(X, test_predictions)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = LR_model.predict(x_range.reshape(-1, 1))
        r2 = LR_model.score(X, test_predictions)

        fig_test = go.Figure()
        fig_test.add_annotation(xref="x domain",yref="y domain",x=0.8,y=0.95,text=f"R-squared: {r2:.2f}", showarrow=False,)
        fig_test.add_trace(go.Scatter(x = test_labels, y = test_predictions, marker=dict(color='rgba(30, 100, 200, 1)', size=5), mode='markers', marker_symbol='circle'))
        fig_test.add_trace(go.Scatter(name='line of best fit', x=x_range, y=y_range, mode='lines', line=dict(color='pink')))


        fig_test.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", hovermode="x", showlegend=False)
        fig_test.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False, title='Predictions')
        fig_test.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False, title ='True Value')

        fig_nodes = go.Figure()
        layers = model.layers
        #print(layers[1].name)
        print(layers[1])
        for k in range(1,len(layers)):
            layer_nodes = np.shape(model.get_layer(layers[k].name).weights[0])[0]
            for i in range(layer_nodes):
                for j in range(np.shape(model.get_layer(layers[k].name).weights[0][1])[0]):
                    value = model.get_layer(layers[k].name).weights[0][i][j].numpy()
                    width = (abs(value))*5
                    if value > 0: color = 'rgba(0,120,20,0.5)'
                    else: color = 'rgba(60,0,120,0.5)'
                    y1 = -layer_nodes/2 + i
                    y2 = -np.shape(model.get_layer(layers[k].name).weights[0][1])[0]/2 + j
                    fig_nodes.add_trace(go.Scatter(go.Scatter(x=[k,k+1]), y=[y1,y2], mode='markers+lines', line=dict(color=color, width=width), marker=dict(color='black', size=10)))

        fig_nodes.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", showlegend=False, hovermode=False,)
        fig_nodes.update_xaxes(range = [0.9,len(layers)+0.1], showticklabels=False, showline=False, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False)
        fig_nodes.update_yaxes(tickvals=np.arange(-len(model.get_weights()[0])/2,len(model.get_weights()[0]/2)), ticktext = vars, showline=False, showticklabels=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False)

        print(model.summary())
        div = [
            html.H3('Model Strength'),
            dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_losses, config={'displaylogo': False,  'displayModeBar': False}, style={'height':'20em'})),
            dbc.Col(dcc.Graph(figure=fig_test, config={'displaylogo': False,  'displayModeBar': False}, style={'height':'20em'})),
            ]),
            dcc.Graph(figure=fig_nodes, config={'displaylogo': False, 'displayModeBar': False}, style={'height':'50em'})]

        weights_list = model.get_weights()


        obj = weights_list
        obj_base64string = codecs.encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode('latin1')

        return div, model.to_json(), obj_base64string,
    else: return [], '', ''


@app.callback(

    Output('slider-div-parent', 'children'),
    Input('train-button', 'n_clicks'),
    State('variables', 'value'),
    State('outputs', 'value'),
    State('intermediate-value', 'data'),
    Input('intermediate-model', 'data'),

)
def build_sliders(n_clicks, vars, output, df_JSON, model_JSON):
    changed_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if 'train-button' in changed_id:
        if model_JSON != '':
            #model  = tf.keras.models.model_from_json(model_JSON)
            df = pd.read_json(df_JSON, orient='split')
            df = df[[output]+ vars]
            Min = df[vars].min().values.tolist()
            Mean = df[vars].mean().values.tolist()
            Max = df[vars].max().values.tolist()
            sliders = html.Div(
                        children =[dbc.Row([
                                    html.H6(vars[i]),
                                    dcc.Slider(
                                        Min[i],
                                        Max[i],
                                        included=False,
                                        value=Mean[i],
                                        id={'type':'cutsom-sliders', 'id': 'input %i' % i},
                                        tooltip={"placement": "top", "always_visible": True} )])
                                    for (i, var) in enumerate(vars)])

            return dbc.Col([
                html.H3('Make Predictions'),
                html.H4(f'Predicted {output}:'),
                html.Div(id='model-outcome',  style={'background-color':'pink', 'border-radius': '20px', 'padding-left':'5px'}),
                #html.H4(result)
                sliders])
    else: return []

@app.callback(

    Output({'type':'cutsom-sliders', 'id': ALL}, 'value'),
    Output('model-outcome', 'children'),

    Input({'type':'cutsom-sliders', 'id': ALL}, 'value'),
    State('variables', 'value'),
    Input('intermediate-model', 'data'),
    Input('intermediate-weights', 'data'),
    )


def update_sliders(slider_values, vars, model_JSON, weightsB64):
    #print(slider_values)
    model  = tf.keras.models.model_from_json(model_JSON)
    weights = pickle.loads(codecs.decode(weightsB64.encode('latin1'), "base64"))
    model.set_weights(weights)
    print(model.summary())

    #slider_values = [22.8,16.2,5.4,7.7,31.0,7.0,6,82,32,1024.1,0.0]
    # df = pd.DataFrame([slider_values])
    # df.columns =vars
    test_predictions = model.predict([slider_values]).flatten()

    #test_predictions = model.predict(df).flatten()
    return slider_values, html.H3("{:.2f}".format(test_predictions[0]))

if __name__ == '__main__':
    app.run_server(debug=True)
