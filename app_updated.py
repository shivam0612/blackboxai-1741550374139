from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime, timedelta
import logging
import json
from src.polygon_api import PolygonAPI
from src.data_preprocessing import DataPreprocessor
from src.model_trainer import StockPredictor
import plotly
import plotly.graph_objs as go
import pandas as pd

app = Flask(__name__, static_folder='static')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.api = PolygonAPI()
        self.preprocessor = DataPreprocessor()
    
    def get_stock_data_and_prediction(self, symbol: str, expiry_date: str) -> dict:
        """Get historical data and prediction for a stock."""
        try:
            # Get historical data
            df = self.api.get_stock_data(
                symbol=symbol,
                from_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                to_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Process data for prediction
            df_processed = self.preprocessor.add_technical_indicators(df)
            
            # Prepare data for training if model doesn't exist
            predictor = StockPredictor(symbol)
            try:
                predictor.load_model()
            except FileNotFoundError:
                logger.info(f"No existing model found for {symbol}. Training new model...")
                # Prepare training data
                X, y = self.preprocessor.prepare_sequences(df_processed)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model
                predictor.train(X_train, y_train, validation_data=(X_test, y_test))
                predictor.save_model(self.preprocessor.price_scaler)
                logger.info(f"Model training completed for {symbol}")
            
            # Prepare sequence for prediction
            sequence = self.preprocessor.prepare_realtime_data(df_processed)
            
            prediction_scaled = predictor.predict(sequence)
            prediction = self.preprocessor.inverse_transform_prices(prediction_scaled)
            
            # Calculate prediction metrics
            current_price = df['close'].iloc[-1]
            predicted_price = float(prediction[0][0])
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Create plotly figure
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Add prediction point
            fig.add_trace(go.Scatter(
                x=[df.index[-1] + timedelta(days=1)],
                y=[predicted_price],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=10)
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Convert plot to JSON
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Get company information
            company_info = self.api.get_company_info(symbol)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'plot_data': plot_json,
                'company_info': company_info
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            raise

prediction_service = PredictionService()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', year=datetime.now().year)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        symbol = request.form['symbol'].upper()
        expiry_date = request.form['expiry-date']  # Get the selected expiry date
        result = prediction_service.get_stock_data_and_prediction(symbol, expiry_date)
        return render_template(
            'prediction.html',
            result=result,
            error=None,
            year=datetime.now().year
        )
    except Exception as e:
        logger.error(f"Error in prediction route: {str(e)}")
        return render_template(
            'prediction.html',
            result=None,
            error=str(e),
            year=datetime.now().year
        )

@app.route('/api/predict/<symbol>')
def api_predict(symbol):
    """API endpoint for predictions."""
    try:
        result = prediction_service.get_stock_data_and_prediction(symbol.upper())
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
