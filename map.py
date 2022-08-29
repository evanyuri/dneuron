#Dash Plotly Libraries
from dash import dcc
import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ML Libraries
import tensorflow as tf
from sklearn.linear_model import LinearRegression

# Data Structure
import pandas as pd
import numpy as np
import base64
import io
import os
#just for app building

path = os.getcwd() + "/weather.csv"
df = pd.read_csv(path)
df = df.select_dtypes(include=['float64', 'int64']).astype('float64')
output = 'MinTemp'
vars = df.columns.tolist()
vars.remove(output)


df = df.dropna()
print(df.columns)
#print(df)

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()
# print('test features:', test_features)

train_labels = train_features.pop(output)
test_labels = test_features.pop(output)

# print('test lables:', test_labels)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
#build model 
def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        tf.keras.layers.Dense(len(vars), activation='tanh', name='layer1'),
        tf.keras.layers.Dense(len(vars), activation='tanh', name='layer2'),
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

X = test_labels.values.reshape(-1,1)

LR_model = LinearRegression()
LR_model.fit(X, test_predictions)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = LR_model.predict(x_range.reshape(-1, 1))
r2 = LR_model.score(X, test_predictions)

fig_test = go.Figure()

fig_test.add_trace(go.Scatter(name='line of best fit', x=x_range, y=y_range, mode='lines'))
fig_test.add_trace(go.Scatter(x = test_labels, y = test_predictions, marker=dict(color='rgba(30, 100, 200, 1)', size=5), mode='markers', marker_symbol='circle'))
fig_test.add_annotation(xref="x domain",yref="y domain",x=0.7,y=0.9,text=f"R-squared: {r2:.2f}", showarrow=False,)




fig_test.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", hovermode="x", showlegend=False)
fig_test.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False, title='Predictions')
fig_test.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False, title ='True Value')
fig_test.show(config= {'displaylogo': False})

#print(model.layers[0].weights.shape[0])
fig_nodes = go.Figure()
# print('print shape:')
for layer in model.layers:
    print(layer.name)
# print(len(model.layers))
# print(np.shape(model.get_layer("layer1").weights[0])[0])
# print(np.shape(model.get_layer("layer1").weights[0][1]))
# print(model.get_layer("layer1").weights[0][1][1].numpy())

layers = model.layers
#print(layers[1].name)
# print(layers[1])
for k in range(1,len(layers)):
    layer_nodes = np.shape(model.get_layer(layers[k].name).weights[0])[0]
    for i in range(layer_nodes):
        for j in range(np.shape(model.get_layer(layers[k].name).weights[0][1])[0]):
            value = model.get_layer(layers[k].name).weights[0][i][j].numpy()
            width = (abs(value))*5
            if value > 0: color = 'rgba(0,80,140,0.5)'
            else: color = 'rgba(150,100,10,0.5)'
            y1 = -layer_nodes/2 + i
            y2 = -np.shape(model.get_layer(layers[k].name).weights[0][1])[0]/2 + j
            fig_nodes.add_trace(go.Scatter(go.Scatter(x=[k,k+1]), y=[y1,y2], mode='markers+lines', line=dict(color=color, width=width), marker=dict(color='black', size=10)))

fig_nodes.update_layout(margin=dict(l=1, r=10, t=30, b=70), plot_bgcolor = "white", paper_bgcolor="white", showlegend=False, hovermode=False,)
fig_nodes.update_xaxes(range = [0.9,len(layers)+0.1], showticklabels=False, showline=True, linewidth=1, linecolor='black', mirror=True, showgrid =False, zeroline = False)
fig_nodes.update_yaxes(tickvals=np.arange(-len(model.get_weights()[0])/2,len(model.get_weights()[0]/2)), ticktext = vars, showline=True, showticklabels=True, linewidth=1, linecolor='black', mirror=True, showgrid = False, zeroline = False)
# print(model.get_layer("layerFinal").weights)
# print(model.get_layer("layerFinal").weights[0])
# print(model.get_layer("layerFinal").weights[0][1])
print(model.get_weights()[0])
print(model.get_weights()[2])
print(model.get_weights()[3])


#print(model.get_weights()[1])
# print(model.layers[0].bias.numpy())
# print(model.layers[0].bias_initializer)

fig_nodes.show(config= {'displaylogo': False})