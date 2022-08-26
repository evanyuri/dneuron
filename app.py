#Dash Plotly Libraries
from dash import dcc
import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# ML Libraries
import tensorflow as tf

# Data Structure
import pandas as pd
import numpy as np
import base64
import io
import os

#just for app building

# path = os.getcwd() + "/weather.csv"
# df = pd.read_csv(path)


app = Dash(__name__, assets_folder="assets", external_stylesheets = [dbc.themes.BOOTSTRAP, 'style.css'], suppress_callback_exceptions=True)


app.layout = html.Div(
    style={'border-radius': '20px', 'margin':'30px'},
    children=[

    html.H1(children='Deep Neural Network It'),
    dbc.Row([
    dbc.Col([
        html.H5(children='1. Upload single header data file'),
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drop or ',
            html.A('Select File')
            ]),
        style={
        'width': '100%',
        'padding-top': '35px',
        'height': '100px',
        'lineHeight': '20px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin-bottom': '30px',
        'border-radius': '20px',
        },
    # Allow multiple files to be uploaded
     multiple=False
            ),
    html.H5(children='2. Choose variables'),
    dcc.Dropdown(id='variables', multi = True, placeholder='Choose Variables',options=[],value=[], style={'margin-bottom': '15px'}),
    html.H5(children='3. Choose ouput to predict'),
    dcc.Dropdown(id='outputs', multi = False, placeholder='Choose Output',options=[], style={'margin-bottom': '15px'}),
    html.Button('Train!',id ='train-button', className="button-col-2", style={'width': '100%',}),

    ], width=4),
    dbc.Col([
        html.Div(id='sliders'),
    ], width = 3), 
    dbc.Col([
        html.Div(id='results'),
    ], width = 5), 
    dcc.Store(id='intermediate-value'),

    ]),])

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

    Output('sliders', 'children'),
    Input('train-button', 'n_clicks'),
    State('variables', 'value'),
    State('outputs', 'value'),
    State('intermediate-value', 'data'),
)
def update_output(n_clicks, vars, output, df_JSON):
    changed_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if 'train-button' in changed_id:
        df = pd.read_json(df_JSON, orient='split')
        df = df[[output]+ vars]
        Min = df.min().values.tolist()
        Mean = df.mean().values.tolist()
        Max = df.max().values.tolist()
        sliders = []
        for i in range(len(vars)):
            sliders = sliders + [dbc.Row([html.H6(vars[i]), dcc.Slider(Min[i], Max[i], included=False, value=Mean[i], id=f'slider{i}',tooltip={"placement": "top", "always_visible": True} )])]
        return sliders
    else: return []


@app.callback(

    Output('results', 'children'),
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
        #print('NA values in df: ', df.isna().sum())
        df = df[[output]+ vars]
        df = df.dropna()
        #print(df)

        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(output)
        test_labels = test_features.pop(output)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        #build model 
        def build_and_compile_model(norm):
            model = tf.keras.Sequential([
                norm,
                tf.keras.layers.Dense(8, activation='relu', name='layer1'),
                #tf.keras.layers.Dense(8, activation='relu', name='layer2'),
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
        fig_losses.add_trace(go.Scatter(x = losses.index, y = losses['val_loss'], name='val_loss'))
        fig_losses.add_trace(go.Scatter(x = losses.index, y = losses['loss'], name='loss'))
        fig_losses.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", hovermode="x", legend=dict(x=0.7,y=0.9))
        fig_losses.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False, title='Epoch')
        fig_losses.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False, title ='Error')


        test_predictions = model.predict(test_features).flatten()
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x = test_labels, y = test_predictions, marker=dict(color='rgba(30, 100, 200, 1)', size=5), mode='markers', marker_symbol='circle'))
        fig_test.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", hovermode="x", showlegend=False)
        fig_test.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False, title='Predictions')
        fig_test.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False, title ='True Value')


        #print(model.layers[0].weights.shape[0])
        fig_nodes = go.Figure()
        fig_nodes.add_trace(go.Scatter(go.Scatter(x=np.zeros(len(model.get_weights()[0])), y=np.arange(len(model.get_weights()[0])), mode='markers')))
        # fig_nodes.add_trace(go.Scatter(go.Scatter(x=np.zeros(len(model.get_weights()[1])) + 1, y=np.arange(len(model.get_weights()[1])), mode='markers')))
        #fig_nodes.add_trace(go.Scatter(go.Scatter(x=np.zeros(len(model.get_weights()[2])) + 2, y=np.arange(len(model.get_weights()[2])), mode='markers')))

        fig_nodes.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", showlegend=False)
        fig_nodes.update_xaxes(range = [-0.1,3], showticklabels=False, showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False)
        fig_nodes.update_yaxes(tickvals=np.arange(len(model.get_weights()[0])), ticktext = vars, showline=True, showticklabels=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False)



        div = [dcc.Graph(figure=fig_losses),dcc.Graph(figure=fig_test), dcc.Graph(figure=fig_nodes)]



        print('print shape:')
        # for layer in model.layers:
        #     print(layer.name, layer)
        print((model.get_layer("layer1").weights[0][1]))
        # print(model.get_weights()[1])
        # print(model.get_weights()[2])
        # print(model.get_weights()[3])


        #print(model.get_weights()[1])
        # print(model.layers[0].bias.numpy())
        # print(model.layers[0].bias_initializer)

        return div
    else: return [],

if __name__ == '__main__':
    app.run_server(debug=True)
