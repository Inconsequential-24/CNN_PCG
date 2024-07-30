import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from keras.models import load_model # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
import io
import base64
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

# Load your model
model = load_model('pcg_model.h5')

# Define your classes
classes = ['normal', 'aortic_stenosis', 'mitral_stenosis', 'mitral_valve_prolapse', 'pericardial_murmurs']

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(id='main-container', children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Footer(html.Div('Copyright @Juhi Dwivedi', style={'textAlign': 'center'}))
])

# Home Page Layout
home_page = html.Div([
    html.H1("Namastey, Welcome to Sushrut Samitah"),
    html.Div(id='home-container', children=[
        html.H2("Phonocardiogram (PCG) Analysis"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload WAV File'),
            multiple=False
        ),
        html.Div(id='output-data-upload'),
        html.Button('Predict Heart Condition', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output')
    ])
])

# Detailed Information Pages
info_pages = {
    'aortic_stenosis': html.Div([
        html.H2('Aortic Stenosis'),
        html.P('Aortic stenosis is a condition in which the aortic valve in the heart becomes narrowed.'),
        html.H3('Possible Treatments'),
        html.P('Treatment options include medication, valve repair, or valve replacement surgery.'),
        html.H3('References'),
        html.P('For more information, visit [American Heart Association](https://www.heart.org).')
    ]),
    'mitral_stenosis': html.Div([
        html.H2('Mitral Stenosis'),
        html.P('Mitral stenosis is a narrowing of the mitral valve in the heart.'),
        html.H3('Possible Treatments'),
        html.P('Treatment options include medication, balloon valvuloplasty, or mitral valve repair/replacement.'),
        html.H3('References'),
        html.P('For more information, visit [Mayo Clinic](https://www.mayoclinic.org).')
    ]),
    'mitral_valve_prolapse': html.Div([
        html.H2('Mitral Valve Prolapse'),
        html.P('Mitral valve prolapse occurs when the mitral valve in the heart does not close properly.'),
        html.H3('Possible Treatments'),
        html.P('Treatment options may include medication or surgery if the condition is severe.'),
        html.H3('References'),
        html.P('For more information, visit [Cleveland Clinic](https://my.clevelandclinic.org).')
    ]),
    'pericardial_murmurs': html.Div([
        html.H2('Pericardial Murmurs'),
        html.P('Pericardial murmurs are sounds produced by the movement of the heart’s outer lining.'),
        html.H3('Possible Treatments'),
        html.P('Treatment depends on the underlying cause, which may include medication or surgery.'),
        html.H3('References'),
        html.P('For more information, visit [Johns Hopkins Medicine](https://www.hopkinsmedicine.org).'),
        html.Button('Connect with a Doctor', id='connect-button'),
        html.Div(id='doctor-output')
    ])
}

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return home_page
    elif pathname == '/aortic_stenosis':
        return info_pages['aortic_stenosis']
    elif pathname == '/mitral_stenosis':
        return info_pages['mitral_stenosis']
    elif pathname == '/mitral_valve_prolapse':
        return info_pages['mitral_valve_prolapse']
    elif pathname == '/pericardial_murmurs':
        return info_pages['pericardial_murmurs']
    else:
        return html.Div([
            html.H2('Page not found')
        ])

def parse_wav(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    audio = io.BytesIO(decoded)
    _, signal = wavfile.read(audio)
    return signal

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents')
)
def update_output(uploaded_file):
    if uploaded_file is None:
        raise PreventUpdate

    # Read the uploaded WAV file
    signal = parse_wav(uploaded_file)

    # Preprocess the signal (normalization and padding)
    signal = np.array(signal, dtype=float)
    scaler = MinMaxScaler()
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    signal_padded = pad_sequences([signal], maxlen=1000, dtype='float32', padding='post', truncating='post')[0]
    signal_padded = signal_padded.reshape(1, 1000, 1)  # Reshape for CNN

    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Heart Sound Signal'))
    fig.update_layout(title='Heart Sound Signal', xaxis_title='Sample', yaxis_title='Amplitude')

    return html.Div([
        html.H4('File uploaded successfully. Ready for prediction.'),
        dcc.Graph(figure=fig)
    ])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('upload-data', 'contents')
)
def predict_heart_condition(n_clicks, uploaded_file):
    if n_clicks == 0 or uploaded_file is None:
        raise PreventUpdate

    # Read and preprocess the signal
    signal = parse_wav(uploaded_file)
    signal = np.array(signal, dtype=float)
    scaler = MinMaxScaler()
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    signal_padded = pad_sequences([signal], maxlen=1000, dtype='float32', padding='post', truncating='post')[0]
    signal_padded = signal_padded.reshape(1, 1000, 1)  # Reshape for CNN

    # Predict using the loaded model
    prediction = model.predict(signal_padded)
    predicted_class = classes[np.argmax(prediction)]

    # Define messages and know more link
    if predicted_class == 'normal':
        message = "Your diagnosis is normal, stay healthy!"
        know_more_button = ""
    else:
        message = html.Div([
            f"You have been diagnosed with {predicted_class.replace('_', ' ').title()}.",
            html.Br(),
            html.A("Know More", href=f"/{predicted_class.replace('_', '-')}", className="button")
        ])

    return message

@app.callback(
    Output('doctor-output', 'children'),
    Input('connect-button', 'n_clicks')
)
def connect_with_doctor(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    return html.Div([
        html.Img(src='/assets/dr_mike.png', style={'width': '300px', 'height': 'auto'}),
        html.P("This feature is under development. We are looking for collaborations.")
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
