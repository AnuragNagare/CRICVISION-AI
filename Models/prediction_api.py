
import pickle
import numpy as np

class CricketPredictionAPI:
    """API for making cricket predictions"""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models and scalers"""
        model_types = ['wicket_prediction', 'runs_prediction', 'boundary_prediction']
        
        for model_type in model_types:
            try:
                # Load model
                model_path = f"{self.models_dir}/{model_type}_model.pkl"
                with open(model_path, 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                
                # Load scaler
                scaler_path = f"{self.models_dir}/{model_type}_scaler.pkl"
                with open(scaler_path, 'rb') as f:
                    self.scalers[model_type] = pickle.load(f)
                
                print(f"✓ Loaded {model_type}")
            except Exception as e:
                print(f"✗ Error loading {model_type}: {str(e)}")
    
    def predict_wicket_probability(self, over, cumulative_runs, cumulative_wickets, 
                                   balls_remaining, current_run_rate, pressure_index):
        """Predict probability of wicket on next ball"""
        features = np.array([[over, cumulative_runs, cumulative_wickets, 
                            balls_remaining, current_run_rate, pressure_index]])
        
        features_scaled = self.scalers['wicket_prediction'].transform(features)
        probability = self.models['wicket_prediction'].predict_proba(features_scaled)[0][1]
        
        return probability
    
    def predict_runs(self, over, cumulative_runs, cumulative_wickets, 
                    balls_remaining, current_run_rate, pressure_index):
        """Predict runs likely to be scored"""
        features = np.array([[over, cumulative_runs, cumulative_wickets, 
                            balls_remaining, current_run_rate, pressure_index]])
        
        features_scaled = self.scalers['runs_prediction'].transform(features)
        predicted_runs = self.models['runs_prediction'].predict(features_scaled)[0]
        
        return max(0, predicted_runs)  # Ensure non-negative
    
    def predict_boundary_probability(self, over, cumulative_runs, cumulative_wickets, 
                                     balls_remaining, current_run_rate, pressure_index):
        """Predict probability of boundary (four or six)"""
        features = np.array([[over, cumulative_runs, cumulative_wickets, 
                            balls_remaining, current_run_rate, pressure_index]])
        
        features_scaled = self.scalers['boundary_prediction'].transform(features)
        probability = self.models['boundary_prediction'].predict_proba(features_scaled)[0][1]
        
        return probability
    
    def get_match_insights(self, over, cumulative_runs, cumulative_wickets, 
                          balls_remaining, current_run_rate, pressure_index):
        """Get comprehensive match insights"""
        wicket_prob = self.predict_wicket_probability(
            over, cumulative_runs, cumulative_wickets, 
            balls_remaining, current_run_rate, pressure_index
        )
        
        expected_runs = self.predict_runs(
            over, cumulative_runs, cumulative_wickets, 
            balls_remaining, current_run_rate, pressure_index
        )
        
        boundary_prob = self.predict_boundary_probability(
            over, cumulative_runs, cumulative_wickets, 
            balls_remaining, current_run_rate, pressure_index
        )
        
        return {
            'wicket_probability': round(wicket_prob * 100, 2),
            'expected_runs': round(expected_runs, 2),
            'boundary_probability': round(boundary_prob * 100, 2),
            'pressure_level': 'High' if pressure_index > 5 else 'Medium' if pressure_index > 2 else 'Low'
        }

# Example usage:
# api = CricketPredictionAPI(models_dir="path/to/models")
# insights = api.get_match_insights(over=15, cumulative_runs=120, cumulative_wickets=3,
#                                   balls_remaining=30, current_run_rate=8.0, pressure_index=2.5)
# print(insights)
