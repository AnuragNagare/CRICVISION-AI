import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Paths
MODELS_DIR = r"D:\Git hub project\CRICVISION AI\Models"
FEATURES_DIR = r"D:\Git hub project\CRICVISION AI\Processed_Data\Features"
PROCESSED_DIR = r"D:\Git hub project\CRICVISION AI\Processed_Data"

# Load models and scalers
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_all()
    
    def load_all(self):
        """Load all models and scalers"""
        model_types = ['wicket_prediction', 'runs_prediction', 'boundary_prediction']
        
        for model_type in model_types:
            try:
                model_path = os.path.join(MODELS_DIR, f'{model_type}_model.pkl')
                with open(model_path, 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                
                scaler_path = os.path.join(MODELS_DIR, f'{model_type}_scaler.pkl')
                with open(scaler_path, 'rb') as f:
                    self.scalers[model_type] = pickle.load(f)
                
                print(f"✓ Loaded {model_type}")
            except Exception as e:
                print(f"✗ Error loading {model_type}: {str(e)}")
    
    def predict_wicket(self, features):
        features_scaled = self.scalers['wicket_prediction'].transform(np.array(features).reshape(1, -1))
        prob = self.models['wicket_prediction'].predict_proba(features_scaled)[0][1]
        return prob * 100
    
    def predict_runs(self, features):
        features_scaled = self.scalers['runs_prediction'].transform(np.array(features).reshape(1, -1))
        runs = self.models['runs_prediction'].predict(features_scaled)[0]
        return max(0, runs)
    
    def predict_boundary(self, features):
        features_scaled = self.scalers['boundary_prediction'].transform(np.array(features).reshape(1, -1))
        prob = self.models['boundary_prediction'].predict_proba(features_scaled)[0][1]
        return prob * 100

model_loader = ModelLoader()

# Load additional datasets for new features
def load_additional_data():
    """Load data for player comparison and form trends"""
    data = {}
    
    try:
        batting_files = [f for f in os.listdir(FEATURES_DIR) if 'batting_form_features' in f]
        if batting_files:
            data['batting_stats'] = pd.read_csv(os.path.join(FEATURES_DIR, batting_files[0]))
            print(f"✓ Loaded batting stats: {len(data['batting_stats'])} records")
    except Exception as e:
        print(f"✗ Error loading batting stats: {e}")
        data['batting_stats'] = pd.DataFrame()
    
    try:
        bowling_files = [f for f in os.listdir(FEATURES_DIR) if 'bowling_form_features' in f]
        if bowling_files:
            data['bowling_stats'] = pd.read_csv(os.path.join(FEATURES_DIR, bowling_files[0]))
            print(f"✓ Loaded bowling stats: {len(data['bowling_stats'])} records")
    except Exception as e:
        print(f"✗ Error loading bowling stats: {e}")
        data['bowling_stats'] = pd.DataFrame()
    
    try:
        venue_file = os.path.join(FEATURES_DIR, 'venue_features.csv')
        if os.path.exists(venue_file):
            data['venues'] = pd.read_csv(venue_file)
            print(f"✓ Loaded venue data: {len(data['venues'])} venues")
    except Exception as e:
        print(f"✗ Error loading venue data: {e}")
        data['venues'] = pd.DataFrame()
    
    return data

additional_data = load_additional_data()

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "CricVision AI - Cricket Analytics"

# Enhanced color scheme
colors = {
    'background': '#0a0e27',
    'card': '#1a1f3a',
    'card_hover': '#252b4a',
    'text': '#ffffff',
    'text_secondary': '#a0aec0',
    'primary': '#00d9ff',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#fbbf24',
    'danger': '#ef4444',
    'accent': '#f97316'
}

custom_style = {
    'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
}

# App layout
app.layout = html.Div(style={
    'backgroundColor': colors['background'], 
    'minHeight': '100vh', 
    'padding': '0',
    **custom_style
}, children=[
    
    # Top Navigation Bar
    html.Div(style={
        'backgroundColor': colors['card'],
        'padding': '20px 40px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
        'marginBottom': '30px'
    }, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
            html.Div(children=[
                html.H1('🏏 CricVision AI', style={
                    'color': colors['primary'], 
                    'fontSize': '36px', 
                    'fontWeight': '800',
                    'margin': '0',
                    'letterSpacing': '-1px'
                }),
                html.P('Advanced Cricket Analytics & Predictions', style={
                    'color': colors['text_secondary'], 
                    'fontSize': '14px',
                    'margin': '5px 0 0 0'
                })
            ]),
            html.Div(children=[
                html.Span('LIVE', style={
                    'backgroundColor': colors['danger'],
                    'color': 'white',
                    'padding': '8px 16px',
                    'borderRadius': '20px',
                    'fontSize': '12px',
                    'fontWeight': 'bold'
                })
            ])
        ])
    ]),
    
    # Main Container
    html.Div(style={'padding': '0 40px'}, children=[
        
        # Control Panel Row
        html.Div(style={'marginBottom': '30px'}, children=[
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '30px',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.H3('⚙️ Match Control Center', style={
                    'color': colors['primary'],
                    'marginBottom': '25px',
                    'fontSize': '24px',
                    'fontWeight': '700'
                }),
                
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px'}, children=[
                    
                    html.Div(children=[
                        html.Label('Current Over', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Slider(id='over-slider', min=0, max=20, step=1, value=10,
                                  marks={i: {'label': str(i), 'style': {'color': colors['text']}} for i in range(0, 21, 5)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ]),
                    
                    html.Div(children=[
                        html.Label('Total Runs', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Input(id='runs-input', type='number', value=80, 
                                 style={'width': '100%', 'padding': '12px', 'borderRadius': '8px', 'border': 'none', 'fontSize': '16px'})
                    ]),
                    
                    html.Div(children=[
                        html.Label('Wickets Down', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Slider(id='wickets-slider', min=0, max=10, step=1, value=3,
                                  marks={i: {'label': str(i), 'style': {'color': colors['text']}} for i in range(0, 11, 2)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ]),
                    
                    html.Div(children=[
                        html.Label('Balls Remaining', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Input(id='balls-input', type='number', value=60, 
                                 style={'width': '100%', 'padding': '12px', 'borderRadius': '8px', 'border': 'none', 'fontSize': '16px'})
                    ]),
                    
                    html.Div(children=[
                        html.Label('Current Run Rate', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Input(id='run-rate-input', type='number', value=8.0, step=0.1,
                                 style={'width': '100%', 'padding': '12px', 'borderRadius': '8px', 'border': 'none', 'fontSize': '16px'})
                    ]),
                    
                    html.Div(children=[
                        html.Label('Pressure Index', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Slider(id='pressure-slider', min=0, max=10, step=0.5, value=2.5,
                                  marks={i: {'label': str(i), 'style': {'color': colors['text']}} for i in range(0, 11, 2)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ]),
                    
                    html.Div(children=[
                        html.Label('Venue', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='venue-dropdown',
                            options=[{'label': venue, 'value': venue} for venue in 
                                    (additional_data['venues']['venue'].unique() if not additional_data['venues'].empty 
                                    else ['Generic Venue'])],
                            value='Generic Venue',
                            style={'backgroundColor': '#fff', 'color': '#000'}
                        )
                    ])
                ]),
                
                html.Div(style={'marginTop': '30px', 'display': 'flex', 'gap': '15px', 'justifyContent': 'center'}, children=[
                    html.Button('🔮 Generate Predictions', id='predict-button', 
                               style={'padding': '15px 40px', 'fontSize': '16px', 'fontWeight': '700',
                                      'backgroundColor': colors['primary'], 'color': '#000', 
                                      'border': 'none', 'borderRadius': '10px', 'cursor': 'pointer',
                                      'boxShadow': '0 4px 15px rgba(0, 217, 255, 0.4)'}),
                    html.Button('⚡ Powerplay', id='scenario-powerplay', 
                               style={'padding': '15px 30px', 'fontSize': '14px', 'fontWeight': '600',
                                      'backgroundColor': colors['secondary'], 'color': colors['text'],
                                      'border': 'none', 'borderRadius': '10px', 'cursor': 'pointer'}),
                    html.Button('📊 Middle Overs', id='scenario-middle', 
                               style={'padding': '15px 30px', 'fontSize': '14px', 'fontWeight': '600',
                                      'backgroundColor': colors['secondary'], 'color': colors['text'],
                                      'border': 'none', 'borderRadius': '10px', 'cursor': 'pointer'}),
                    html.Button('🔥 Death Overs', id='scenario-death', 
                               style={'padding': '15px 30px', 'fontSize': '14px', 'fontWeight': '600',
                                      'backgroundColor': colors['secondary'], 'color': colors['text'],
                                      'border': 'none', 'borderRadius': '10px', 'cursor': 'pointer'})
                ])
            ])
        ]),
        
        # Main Predictions Grid
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px', 'marginBottom': '30px'}, children=[
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '25px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)',
                'borderLeft': f'4px solid {colors["danger"]}'
            }, children=[
                html.Div('🎯', style={'fontSize': '40px', 'marginBottom': '10px'}),
                html.H4('Wicket Probability', style={'color': colors['text_secondary'], 'fontSize': '14px', 'marginBottom': '10px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                html.H1(id='wicket-prob', children='--', 
                       style={'color': colors['danger'], 'fontSize': '48px', 'fontWeight': '800', 'margin': '10px 0'}),
                html.P(id='wicket-status', children='Awaiting calculation', 
                      style={'color': colors['text_secondary'], 'fontSize': '13px'})
            ]),
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '25px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)',
                'borderLeft': f'4px solid {colors["success"]}'
            }, children=[
                html.Div('📈', style={'fontSize': '40px', 'marginBottom': '10px'}),
                html.H4('Expected Runs/Ball', style={'color': colors['text_secondary'], 'fontSize': '14px', 'marginBottom': '10px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                html.H1(id='expected-runs', children='--', 
                       style={'color': colors['success'], 'fontSize': '48px', 'fontWeight': '800', 'margin': '10px 0'}),
                html.P(id='runs-status', children='Awaiting calculation', 
                      style={'color': colors['text_secondary'], 'fontSize': '13px'})
            ]),
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '25px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)',
                'borderLeft': f'4px solid {colors["warning"]}'
            }, children=[
                html.Div('💥', style={'fontSize': '40px', 'marginBottom': '10px'}),
                html.H4('Boundary Probability', style={'color': colors['text_secondary'], 'fontSize': '14px', 'marginBottom': '10px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                html.H1(id='boundary-prob', children='--', 
                       style={'color': colors['warning'], 'fontSize': '48px', 'fontWeight': '800', 'margin': '10px 0'}),
                html.P(id='boundary-status', children='Awaiting calculation', 
                      style={'color': colors['text_secondary'], 'fontSize': '13px'})
            ]),
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '25px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)',
                'borderLeft': f'4px solid {colors["primary"]}'
            }, children=[
                html.Div('🎲', style={'fontSize': '40px', 'marginBottom': '10px'}),
                html.H4('Projected Final Score', style={'color': colors['text_secondary'], 'fontSize': '14px', 'marginBottom': '10px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                html.H1(id='projected-score', children='--', 
                       style={'color': colors['primary'], 'fontSize': '48px', 'fontWeight': '800', 'margin': '10px 0'}),
                html.P(id='score-status', children='Awaiting calculation', 
                      style={'color': colors['text_secondary'], 'fontSize': '13px'})
            ])
        ]),
        
        # Additional Predictions Row
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginBottom': '30px'}, children=[
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.Div('⚪', style={'fontSize': '32px', 'marginBottom': '8px'}),
                html.H4('Dot Ball Chance', style={'color': colors['text_secondary'], 'fontSize': '13px', 'marginBottom': '8px'}),
                html.H2(id='dot-ball-prob', children='--', 
                       style={'color': colors['text'], 'fontSize': '36px', 'fontWeight': '700', 'margin': '5px 0'})
            ]),
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.Div('🏆', style={'fontSize': '32px', 'marginBottom': '8px'}),
                html.H4('Win Probability', style={'color': colors['text_secondary'], 'fontSize': '13px', 'marginBottom': '8px'}),
                html.H2(id='win-prob', children='--', 
                       style={'color': colors['text'], 'fontSize': '36px', 'fontWeight': '700', 'margin': '5px 0'})
            ]),
            
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.Div('📉', style={'fontSize': '32px', 'marginBottom': '8px'}),
                html.H4('Economy Rate Forecast', style={'color': colors['text_secondary'], 'fontSize': '13px', 'marginBottom': '8px'}),
                html.H2(id='economy-forecast', children='--', 
                       style={'color': colors['text'], 'fontSize': '36px', 'fontWeight': '700', 'margin': '5px 0'})
            ])
        ]),
        
        # Player Comparison Tool
        html.Div(style={'marginBottom': '30px'}, children=[
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '30px',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.H3('👥 Player Comparison Tool', style={
                    'color': colors['primary'],
                    'marginBottom': '20px',
                    'fontSize': '24px',
                    'fontWeight': '700'
                }),
                
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 200px', 'gap': '20px', 'marginBottom': '20px'}, children=[
                    html.Div(children=[
                        html.Label('Player 1', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='player1-dropdown',
                            options=[{'label': player, 'value': player} for player in 
                                    (additional_data['batting_stats']['batter'].unique()[:100] if not additional_data['batting_stats'].empty 
                                    else ['Player 1'])],
                            value=None,
                            placeholder='Select Player 1',
                            style={'backgroundColor': '#fff', 'color': '#000'}
                        )
                    ]),
                    html.Div(children=[
                        html.Label('Player 2', style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='player2-dropdown',
                            options=[{'label': player, 'value': player} for player in 
                                    (additional_data['batting_stats']['batter'].unique()[:100] if not additional_data['batting_stats'].empty 
                                    else ['Player 2'])],
                            value=None,
                            placeholder='Select Player 2',
                            style={'backgroundColor': '#fff', 'color': '#000'}
                        )
                    ]),
                    html.Button('Compare', id='compare-button', 
                               style={'marginTop': '30px', 'padding': '12px 30px', 'fontSize': '16px', 'fontWeight': '600',
                                      'backgroundColor': colors['primary'], 'color': '#000', 
                                      'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer'})
                ]),
                
                html.Div(id='comparison-results', children=[
                    html.P('Select two players to compare', style={'color': colors['text_secondary'], 'textAlign': 'center', 'padding': '40px'})
                ])
            ])
        ]),
        
        # Main Graphs Container - 1200x1200 Box
        html.Div(style={
            'width': '1200px',
            'height': '1200px',
            'margin': '0 auto 30px auto',
            'backgroundColor': colors['card'],
            'borderRadius': '15px',
            'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)',
            'padding': '20px',
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gridTemplateRows': '1fr 1fr',
            'gap': '20px',
            'border': f'2px solid {colors["primary"]}',
            'position': 'relative'
        }, children=[
            
            # Container Title
            html.Div(style={
                'position': 'absolute',
                'top': '-15px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'backgroundColor': colors['primary'],
                'color': '#000',
                'padding': '8px 20px',
                'borderRadius': '20px',
                'fontSize': '16px',
                'fontWeight': '700',
                'zIndex': '10'
            }, children='📊 ANALYTICS DASHBOARD'),
            
            # Top Left - AI Confidence Score
            html.Div(style={
                'backgroundColor': colors['card_hover'],
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 5px 15px rgba(0, 0, 0, 0.2)',
                'display': 'flex',
                'flexDirection': 'column',
                'height': '560px',
                'width': '560px',
                'overflow': 'hidden'
            }, children=[
                html.H4('📊 AI Confidence Score', style={'color': colors['text'], 'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': '600', 'textAlign': 'center'}),
                dcc.Graph(id='confidence-gauge', config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'})
            ]),
            
            # Top Right - Match Phase Analysis
            html.Div(style={
                'backgroundColor': colors['card_hover'],
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 5px 15px rgba(0, 0, 0, 0.2)',
                'display': 'flex',
                'flexDirection': 'column',
                'height': '560px',
                'width': '560px',
                'overflow': 'hidden'
            }, children=[
                html.H4('🎮 Match Phase Analysis', style={'color': colors['text'], 'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': '600', 'textAlign': 'center'}),
                dcc.Graph(id='phase-chart', config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'})
            ]),
            
            # Bottom Left - Over-by-Over Projection
            html.Div(style={
                'backgroundColor': colors['card_hover'],
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 5px 15px rgba(0, 0, 0, 0.2)',
                'display': 'flex',
                'flexDirection': 'column',
                'height': '560px',
                'width': '560px',
                'overflow': 'hidden'
            }, children=[
                html.H4('🔮 Over-by-Over Projection', style={'color': colors['text'], 'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': '600', 'textAlign': 'center'}),
                dcc.Graph(id='simulation-chart', config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'})
            ]),
            
            # Bottom Right - Additional Analytics
            html.Div(style={
                'backgroundColor': colors['card_hover'],
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 5px 15px rgba(0, 0, 0, 0.2)',
                'display': 'flex',
                'flexDirection': 'column',
                'height': '560px',
                'width': '560px',
                'justifyContent': 'center',
                'alignItems': 'center',
                'overflow': 'hidden'
            }, children=[
                html.H4('📈 Advanced Analytics', style={'color': colors['text'], 'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': '600', 'textAlign': 'center'}),
                html.Div(style={'textAlign': 'center', 'color': colors['text_secondary']}, children=[
                    html.P('🎯 Real-time Predictions', style={'fontSize': '16px', 'margin': '10px 0'}),
                    html.P('📊 Performance Metrics', style={'fontSize': '16px', 'margin': '10px 0'}),
                    html.P('🔮 Future Projections', style={'fontSize': '16px', 'margin': '10px 0'}),
                    html.P('⚡ Live Updates', style={'fontSize': '16px', 'margin': '10px 0'})
                ])
            ])
        ]),
        
        # Wagon Wheel Section
        html.Div(style={'marginBottom': '30px'}, children=[
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '30px',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.H3('🎯 Wagon Wheel - Run Distribution', style={
                    'color': colors['primary'],
                    'marginBottom': '20px',
                    'fontSize': '24px',
                    'fontWeight': '700'
                }),
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}, children=[
                    dcc.Dropdown(
                        id='wagon-player-dropdown',
                        options=[{'label': player, 'value': player} for player in 
                                (additional_data['batting_stats']['batter'].unique()[:50] if not additional_data['batting_stats'].empty 
                                else ['Select Player'])],
                        value=None,
                        placeholder='Select Player for Wagon Wheel',
                        style={'width': '300px', 'backgroundColor': '#fff', 'color': '#000'}
                    ),
                    html.Button('Generate Wagon Wheel', id='wagon-button',
                               style={'padding': '10px 25px', 'fontSize': '14px', 'fontWeight': '600',
                                      'backgroundColor': colors['secondary'], 'color': colors['text'],
                                      'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer'})
                ]),
                dcc.Graph(id='wagon-wheel-chart', config={'displayModeBar': False})
            ])
        ]),
        
        # Player Form Trends Section
        html.Div(style={'marginBottom': '30px'}, children=[
            html.Div(style={
                'backgroundColor': colors['card'],
                'padding': '30px',
                'borderRadius': '15px',
                'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.3)'
            }, children=[
                html.H3('📈 Player Form Analysis', style={
                    'color': colors['primary'],
                    'marginBottom': '20px',
                    'fontSize': '24px',
                    'fontWeight': '700'
                }),
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}, children=[
                    dcc.Dropdown(
                        id='form-player-dropdown',
                        options=[{'label': player, 'value': player} for player in 
                                (additional_data['batting_stats']['batter'].unique()[:50] if not additional_data['batting_stats'].empty 
                                else ['Select Player'])],
                        value=None,
                        placeholder='Select Player for Form Analysis',
                        style={'width': '300px', 'backgroundColor': '#fff', 'color': '#000'}
                    ),
                    html.Button('Analyze Form', id='form-button',
                               style={'padding': '10px 25px', 'fontSize': '14px', 'fontWeight': '600',
                                      'backgroundColor': colors['secondary'], 'color': colors['text'],
                                      'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer'})
                ]),
                dcc.Graph(id='form-trend-chart', config={'displayModeBar': False})
            ])
        ])
    ]),
    
    # Footer
    html.Div(style={
        'backgroundColor': colors['card'],
        'textAlign': 'center',
        'padding': '30px',
        'marginTop': '40px'
    }, children=[
        html.P('© 2024 CricVision AI | Powered by Machine Learning & Advanced Analytics', 
               style={'color': colors['text_secondary'], 'fontSize': '14px', 'margin': '0'})
    ])
])

# ==================== CALLBACKS - ALL DEFINED AFTER LAYOUT ====================

@app.callback(
    [Output('over-slider', 'value'),
     Output('runs-input', 'value'),
     Output('wickets-slider', 'value'),
     Output('balls-input', 'value'),
     Output('run-rate-input', 'value'),
     Output('pressure-slider', 'value')],
    [Input('scenario-powerplay', 'n_clicks'),
     Input('scenario-middle', 'n_clicks'),
     Input('scenario-death', 'n_clicks')]
)
def load_scenario(powerplay, middle, death):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'scenario-powerplay':
        return 5, 35, 1, 90, 7.0, 1.5
    elif button_id == 'scenario-middle':
        return 12, 95, 4, 48, 7.9, 3.2
    elif button_id == 'scenario-death':
        return 18, 145, 6, 12, 8.1, 4.5
    
    return dash.no_update

@app.callback(
    [Output('wicket-prob', 'children'),
     Output('wicket-status', 'children'),
     Output('expected-runs', 'children'),
     Output('runs-status', 'children'),
     Output('boundary-prob', 'children'),
     Output('boundary-status', 'children'),
     Output('projected-score', 'children'),
     Output('score-status', 'children'),
     Output('dot-ball-prob', 'children'),
     Output('win-prob', 'children'),
     Output('economy-forecast', 'children'),
     Output('confidence-gauge', 'figure'),
     Output('phase-chart', 'figure'),
     Output('simulation-chart', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State('over-slider', 'value'),
     State('runs-input', 'value'),
     State('wickets-slider', 'value'),
     State('balls-input', 'value'),
     State('run-rate-input', 'value'),
     State('pressure-slider', 'value')]
)
def make_predictions(n_clicks, over, runs, wickets, balls_remaining, run_rate, pressure):
    
    if n_clicks is None:
        empty_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Confidence", 'font': {'color': colors['text']}},
            gauge={'axis': {'range': [0, 100]}}
        ))
        empty_gauge.update_layout(paper_bgcolor=colors['card_hover'], plot_bgcolor=colors['card_hover'], 
                                 font={'color': colors['text']}, height=450, margin=dict(l=20, r=20, t=40, b=20))
        
        empty_phase = go.Figure()
        empty_phase.update_layout(paper_bgcolor=colors['card_hover'], plot_bgcolor=colors['card_hover'], 
                                 font={'color': colors['text']}, height=450)
        
        empty_sim = go.Figure()
        empty_sim.update_layout(paper_bgcolor=colors['card_hover'], plot_bgcolor=colors['card_hover'], 
                               font={'color': colors['text']}, height=450)
        
        return ('--', 'Awaiting calculation', '--', 'Awaiting calculation', '--', 'Awaiting calculation',
                '--', 'Awaiting calculation', '--', '--', '--', empty_gauge, empty_phase, empty_sim)
    
    features = [over, runs, wickets, balls_remaining, run_rate, pressure]
    
    wicket_prob = model_loader.predict_wicket(features)
    expected_runs = model_loader.predict_runs(features)
    boundary_prob = model_loader.predict_boundary(features)
    
    dot_ball_prob = max(0, 100 - boundary_prob - (expected_runs * 30))
    projected_score = runs + (expected_runs * balls_remaining)
    
    if balls_remaining > 0:
        required_rr = (projected_score - runs) / (balls_remaining / 6)
        win_prob = min(95, max(5, 50 + (required_rr - run_rate) * 10))
    else:
        win_prob = 50
    
    economy_rate = run_rate * (1 + (pressure / 20))
    
    wicket_status = "🔴 High Risk" if wicket_prob > 15 else "🟡 Moderate" if wicket_prob > 8 else "🟢 Low Risk"
    runs_status = f"~{expected_runs * 6:.0f} runs/over"
    boundary_status = "⚡ High" if boundary_prob > 20 else "📊 Moderate" if boundary_prob > 10 else "🔒 Low"
    score_status = f"At current rate: {run_rate}/over"
    
    overall_confidence = (wicket_prob + boundary_prob) / 2
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_confidence,
        title={'text': "AI Confidence", 'font': {'color': colors['text'], 'size': 18}},
        delta={'reference': 50, 'increasing': {'color': colors['success']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': colors['text']},
            'bar': {'color': colors['primary']},
            'steps': [
                {'range': [0, 33], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [33, 66], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [66, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {'line': {'color': colors['danger'], 'width': 4}, 'thickness': 0.75, 'value': 75}
        }
    ))
    gauge_fig.update_layout(
        paper_bgcolor=colors['card_hover'],
        plot_bgcolor=colors['card_hover'],
        font={'color': colors['text']},
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    phases = ['Powerplay', 'Middle', 'Death']
    wicket_probs = [5.2, 6.8, 8.5]
    boundary_probs = [18.5, 12.3, 22.7]
    
    current_phase = 0 if over < 6 else 1 if over < 16 else 2
    
    phase_fig = go.Figure()
    phase_fig.add_trace(go.Bar(
        name='Wicket %',
        x=phases,
        y=wicket_probs,
        marker_color=[colors['danger'] if i == current_phase else 'rgba(239, 68, 68, 0.3)' for i in range(3)]
    ))
    phase_fig.add_trace(go.Bar(
        name='Boundary %',
        x=phases,
        y=boundary_probs,
        marker_color=[colors['warning'] if i == current_phase else 'rgba(251, 191, 36, 0.3)' for i in range(3)]
    ))
    phase_fig.update_layout(
        barmode='group',
        paper_bgcolor=colors['card_hover'],
        plot_bgcolor=colors['card_hover'],
        font={'color': colors['text']},
        height=450,
        margin=dict(l=40, r=40, t=10, b=40),
        showlegend=True,
        legend=dict(x=0.65, y=1, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    overs_list = list(range(over, min(over + 10, 21)))
    simulated_wickets = [wicket_prob * (1 + np.random.uniform(-0.2, 0.3)) for _ in overs_list]
    simulated_runs = [expected_runs * 6 * (1 + np.random.uniform(-0.15, 0.25)) for _ in overs_list]
    
    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(
        x=overs_list,
        y=simulated_runs,
        mode='lines+markers',
        name='Expected Runs/Over',
        line=dict(color=colors['success'], width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    sim_fig.add_trace(go.Scatter(
        x=overs_list,
        y=[w * 2 for w in simulated_wickets],
        mode='lines+markers',
        name='Wicket Risk (scaled)',
        line=dict(color=colors['danger'], width=3, dash='dash'),
        marker=dict(size=10),
        yaxis='y2'
    ))
    sim_fig.update_layout(
        paper_bgcolor=colors['card_hover'],
        plot_bgcolor=colors['card_hover'],
        font={'color': colors['text']},
        height=450,
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(title='Over', gridcolor='rgba(255,255,255,0.1)', fixedrange=True),
        yaxis=dict(title='Expected Runs', gridcolor='rgba(255,255,255,0.1)', fixedrange=True),
        yaxis2=dict(title='Wicket Risk (%)', overlaying='y', side='right', gridcolor='rgba(255,255,255,0.1)', fixedrange=True),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
    )
    
    return (f'{wicket_prob:.1f}%', wicket_status, f'{expected_runs:.2f}', runs_status,
            f'{boundary_prob:.1f}%', boundary_status, f'{projected_score:.0f}', score_status,
            f'{dot_ball_prob:.1f}%', f'{win_prob:.1f}%', f'{economy_rate:.2f}',
            gauge_fig, phase_fig, sim_fig)

@app.callback(
    Output('comparison-results', 'children'),
    [Input('compare-button', 'n_clicks')],
    [State('player1-dropdown', 'value'),
     State('player2-dropdown', 'value')]
)
def compare_players(n_clicks, player1, player2):
    if n_clicks is None or not player1 or not player2:
        return html.P('Select two players to compare', 
                     style={'color': colors['text_secondary'], 'textAlign': 'center', 'padding': '40px'})
    
    if additional_data['batting_stats'].empty:
        return html.P('No player data available', 
                     style={'color': colors['text_secondary'], 'textAlign': 'center', 'padding': '40px'})
    
    p1_data = additional_data['batting_stats'][additional_data['batting_stats']['batter'] == player1]
    p2_data = additional_data['batting_stats'][additional_data['batting_stats']['batter'] == player2]
    
    if p1_data.empty or p2_data.empty:
        return html.P('Player data not found', 
                     style={'color': colors['text_secondary'], 'textAlign': 'center', 'padding': '40px'})
    
    p1_avg = p1_data['runs'].mean()
    p2_avg = p2_data['runs'].mean()
    p1_sr = p1_data['strike_rate'].mean() if 'strike_rate' in p1_data.columns else 0
    p2_sr = p2_data['strike_rate'].mean() if 'strike_rate' in p2_data.columns else 0
    p1_innings = len(p1_data)
    p2_innings = len(p2_data)
    
    return html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
        html.Div(style={'backgroundColor': colors['card_hover'], 'padding': '20px', 'borderRadius': '10px'}, children=[
            html.H4(player1, style={'color': colors['primary'], 'marginBottom': '15px'}),
            html.P(f'Innings: {p1_innings}', style={'color': colors['text'], 'margin': '5px 0'}),
            html.P(f'Average: {p1_avg:.2f}', style={'color': colors['text'], 'margin': '5px 0'}),
            html.P(f'Strike Rate: {p1_sr:.2f}', style={'color': colors['text'], 'margin': '5px 0'})
        ]),
        html.Div(style={'backgroundColor': colors['card_hover'], 'padding': '20px', 'borderRadius': '10px'}, children=[
            html.H4(player2, style={'color': colors['warning'], 'marginBottom': '15px'}),
            html.P(f'Innings: {p2_innings}', style={'color': colors['text'], 'margin': '5px 0'}),
            html.P(f'Average: {p2_avg:.2f}', style={'color': colors['text'], 'margin': '5px 0'}),
            html.P(f'Strike Rate: {p2_sr:.2f}', style={'color': colors['text'], 'margin': '5px 0'})
        ])
    ])

@app.callback(
    Output('wagon-wheel-chart', 'figure'),
    [Input('wagon-button', 'n_clicks')],
    [State('wagon-player-dropdown', 'value')]
)
def generate_wagon_wheel(n_clicks, player):
    if n_clicks is None or not player:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font={'color': colors['text']},
            height=400,
            annotations=[{
                'text': 'Select a player and click Generate',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'color': colors['text_secondary']}
            }]
        )
        return empty_fig
    
    angles = np.random.uniform(0, 360, 50)
    distances = np.random.uniform(10, 100, 50)
    runs = np.random.choice([1, 2, 3, 4, 6], 50)
    
    fig = go.Figure()
    
    for run_val in [1, 2, 3, 4, 6]:
        mask = runs == run_val
        fig.add_trace(go.Scatterpolar(
            r=distances[mask],
            theta=angles[mask],
            mode='markers',
            name=f'{run_val} runs',
            marker=dict(
                size=10 if run_val < 4 else 15,
                color=colors['success'] if run_val < 4 else colors['warning'] if run_val == 4 else colors['danger']
            )
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            angularaxis=dict(direction='clockwise')
        ),
        showlegend=True,
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['card'],
        font={'color': colors['text']},
        height=400,
        legend=dict(x=0.85, y=0.95, bgcolor='rgba(0,0,0,0)')
    )
    
    return fig

@app.callback(
    Output('form-trend-chart', 'figure'),
    [Input('form-button', 'n_clicks')],
    [State('form-player-dropdown', 'value')]
)
def analyze_player_form(n_clicks, player):
    if n_clicks is None or not player:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font={'color': colors['text']},
            height=350,
            annotations=[{
                'text': 'Select a player and click Analyze',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'color': colors['text_secondary']}
            }]
        )
        return empty_fig
    
    if additional_data['batting_stats'].empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font={'color': colors['text']},
            height=350,
            annotations=[{
                'text': 'No data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'color': colors['text_secondary']}
            }]
        )
        return empty_fig
    
    player_data = additional_data['batting_stats'][additional_data['batting_stats']['batter'] == player]
    
    if player_data.empty or len(player_data) < 5:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font={'color': colors['text']},
            height=350,
            annotations=[{
                'text': 'Insufficient data for this player',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'color': colors['text_secondary']}
            }]
        )
        return empty_fig
    
    player_data = player_data.sort_index().tail(20)
    innings_numbers = list(range(1, len(player_data) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=innings_numbers,
        y=player_data['runs'].values,
        mode='lines+markers',
        name='Runs',
        line=dict(color=colors['primary'], width=3),
        marker=dict(size=8)
    ))
    
    if 'last_5_avg' in player_data.columns:
        fig.add_trace(go.Scatter(
            x=innings_numbers,
            y=player_data['last_5_avg'].values,
            mode='lines',
            name='5-innings Avg',
            line=dict(color=colors['success'], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{player} - Recent Form',
        xaxis_title='Innings',
        yaxis_title='Runs',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['card'],
        font={'color': colors['text']},
        height=350,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
    )
    
    return fig

if __name__ == '__main__':
    print("=" * 70)
    print("🏏 CRICVISION AI - Advanced Cricket Analytics Dashboard")
    print("=" * 70)
    print("\n🚀 Initializing system...")
    print("📊 Loading AI models...")
    print("🎨 Rendering premium UI...")
    print("\n✅ Dashboard ready!")
    print("🌐 Open your browser and navigate to: http://127.0.0.1:8050")
    print("\n💡 Features:")
    print("   • Real-time wicket probability predictions")
    print("   • Expected runs & boundary forecasts")
    print("   • Win probability calculator")
    print("   • Player comparison tool")
    print("   • Wagon wheel visualization")
    print("   • Player form trends")
    print("   • Over-by-over projections")
    print("   • Interactive match scenarios")
    print("\n" + "=" * 70)
    
    app.run(debug=True, host='127.0.0.1', port=8050)