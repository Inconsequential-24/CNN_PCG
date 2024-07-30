# page_contents.py
from dash import html

info_pages = {
    'aortic_stenosis': html.Div([
        html.H2('Aortic Stenosis'),
        html.P('Aortic stenosis is a condition in which the aortic valve in the heart becomes narrowed.'),
        html.H3('Possible Treatments'),
        html.P('Treatment options include medication, valve repair, or valve replacement surgery.'),
        html.H3('References'),
        html.P('For more information, visit [Johns Hopkins Medicine](https://www.hopkinsmedicine.org).')
    ]),
    'mitral_stenosis': html.Div([
        html.H2('Mitral Stenosis'),
        html.P('Mitral stenosis is a narrowing of the mitral valve in the heart.'),
        html.H3('Possible Treatments'),
        html.P('Treatment options include medication, balloon valvuloplasty, or mitral valve repair/replacement.'),
        html.H3('References'),
        html.P('For more information, visit [Johns Hopkins Medicine](https://www.hopkinsmedicine.org).')
    ]),
    'mitral_valve_prolapse': html.Div([
        html.H2('Mitral Valve Prolapse'),
        html.P('Mitral valve prolapse occurs when the mitral valve in the heart does not close properly.'),
        html.H3('Possible Treatments'),
        html.P('Treatment options may include medication or surgery if the condition is severe.'),
        html.H3('References'),
        html.P('For more information, visit [Johns Hopkins Medicine](https://www.hopkinsmedicine.org).')
        
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
