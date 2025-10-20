 🏏 CricVision AI - Advanced Cricket Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-3.2.0-brightgreen.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.1.4-yellow.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24.3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

An intelligent cricket analytics platform that leverages machine learning to provide real-time match predictions, comprehensive player insights, and interactive visualizations for cricket enthusiasts, analysts, and fantasy league players.



 🌟 Features

 🎯 Real-Time Predictions
- Wicket Probability: AI-powered prediction of wicket fall likelihood
- Expected Runs: Accurate runs per ball forecasting
- Boundary Probability: Prediction of boundary-hitting chances
- Win Probability: Dynamic win percentage calculator
- Dot Ball Chance: Analysis of defensive play probability
- Economy Rate Forecast: Bowling economy predictions

 📊 Interactive Visualizations
- AI Confidence Gauge: Real-time model confidence scoring
- Match Phase Analysis: Powerplay, Middle, and Death overs breakdown
- Over-by-Over Projections: Future performance simulations
- Wagon Wheel: Player run distribution visualization
- Player Form Trends: Recent performance analytics

 👥 Player Analytics
- Head-to-Head Comparison: Compare two players' statistics
- Form Analysis: Track player performance over last 20 innings
- Strike Rate & Average: Comprehensive batting metrics
- Innings History: Detailed player performance records

 🎮 Match Scenarios
- Quick Load Scenarios: Powerplay, Middle Overs, Death Overs
- Custom Match States: Adjustable overs, runs, wickets, and pressure
- Venue Selection: Location-based analysis
- Real-Time Updates: Dynamic prediction refreshing

 🛠️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core programming language |
| Dash | 3.2.0 | Web application framework |
| Plotly | Latest | Interactive data visualizations |
| Pandas | 2.1.4 | Data manipulation and analysis |
| NumPy | 1.24.3 | Numerical computations |
| Scikit-learn | Latest | Machine learning models |
| Pickle | Built-in | Model serialization |



 🚀 Getting Started

 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Edge)


 Configuration

Update the file paths in `dashboard.py` to match your directory structure:


MODELS_DIR = r"path/to/your/Models"
FEATURES_DIR = r"path/to/your/Processed_Data/Features"
PROCESSED_DIR = r"path/to/your/Processed_Data"


 Running the Application


python Dashboard.py


The dashboard will be available at: http://127.0.0.1:8050

 📦 Requirements.txt


dash==3.2.0
pandas==2.1.4
numpy==1.24.3
plotly>=5.18.0
scikit-learn>=1.3.0


 💡 Usage Guide

 1. Match Control Center
- Adjust Current Over (0-20)
- Set Total Runs scored
- Configure Wickets Down (0-10)
- Enter Balls Remaining
- Set Current Run Rate
- Adjust Pressure Index (0-10)
- Select Venue

 2. Generate Predictions
Click "🔮 Generate Predictions" to get:
- Wicket probability percentage
- Expected runs per ball
- Boundary likelihood
- Projected final score
- Win probability
- Economy rate forecast

 3. Quick Scenarios
Use preset buttons for instant analysis:
- ⚡ Powerplay (Overs 1-6)
- 📊 Middle Overs (Overs 7-15)
- 🔥 Death Overs (Overs 16-20)

 4. Player Analysis
- Compare two players head-to-head
- Generate wagon wheel visualizations
- Analyze recent form trends
- View strike rates and averages

 🎨 Features in Detail

 AI Prediction Models
- Wicket Prediction Model: Binary classification using historical ball-by-ball data
- Runs Prediction Model: Regression model for runs per delivery forecasting
- Boundary Prediction Model: Classification model for 4s and 6s probability

 Data Processing
- Feature engineering with rolling averages
- Player form tracking (last 5, 10, 20 innings)
- Venue-specific statistics
- Match phase categorization
- Pressure index calculation

 Visualization Components
- Gauge Charts: AI confidence scoring
- Bar Charts: Phase-wise analysis
- Line Charts: Over-by-over projections
- Polar Charts: Wagon wheel run distribution
- Time Series: Player form trends





 📊 Model Performance

| Model | Accuracy/R² | Features Used |
|-------|-------------|---------------|
| Wicket Prediction | 78%+ | Over, runs, wickets, pressure, run_rate |
| Runs Prediction | R² 0.82+ | Match state, phase, venue, form |
| Boundary Prediction | 75%+ | Batsman form, bowler stats, pressure |



 🔮 Future Enhancements

- [ ] Real-time match data integration via APIs
- [ ] Historical match replay feature
- [ ] Team composition optimizer
- [ ] Pitch condition analysis
- [ ] Weather impact predictions
- [ ] Player fatigue modeling
- [ ] Tournament simulation mode
- [ ] Mobile responsive design
- [ ] Export reports to PDF
- [ ] Integration with fantasy league platforms
