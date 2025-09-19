# Detection and Classification of Medicinal Plants Using AI

## Description

This project leverages artificial intelligence to accurately identify and classify medicinal plants through image analysis. By analyzing features like leaf shape, color, and texture, AI models can efficiently recognize plant species and provide detailed information about their medicinal properties. The system also includes a crop recommendation feature based on environmental parameters.

The project combines:
- Plant detection and classification using Roboflow's Inference API
- Detailed leaf analysis with computer vision techniques
- Crop recommendation using machine learning models
- Web interfaces for user interaction
- Data collection and analysis for medicinal plant information

## Features

- **Plant Detection**: Upload leaf images to identify medicinal plants using AI
- **Detailed Analysis**: Get comprehensive leaf metrics (dimensions, color, shape, texture)
- **Medicinal Information**: Access detailed healthcare benefits, traditional uses, and precautions
- **Crop Recommendation**: Predict the best crops to cultivate based on soil and environmental parameters
- **Web Interface**: User-friendly web application for easy interaction
- **API Endpoints**: RESTful APIs for integration with other systems
- **Analytics**: Track usage statistics and plant identification trends
- **Feedback System**: Report accuracy issues for model improvement

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Detection-and-Classification-of-Medicinal-Plants-Using-AI-main
   ```

2. Install dependencies:
   ```bash
   pip install -r project_final_year/requirement.txt
   ```

3. Set up environment variables:
   - Obtain a Roboflow API key and update it in `main.py`
   - Ensure all data files are in place

4. Run the application:
   ```bash
   cd project_final_year
   python main.py
   ```

   For crop recommendation:
   ```bash
   cd project_final_year/plant_recommendation
   python app.py
   ```

## Usage

### Web Interface

1. Start the FastAPI server
2. Open browser to `http://localhost:8000`
3. Upload leaf images for analysis
4. View detailed plant information and recommendations

### API Usage

#### Plant Detection
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "files=@leaf_image.jpg"
```

#### Leaf Analysis
```bash
curl -X POST "http://localhost:8000/analyze_leaf" \
     -F "file=@leaf_image.jpg"
```

#### Crop Recommendation
```bash
curl -X POST "http://localhost:8000/predict_crop" \
     -d "Nitrogen=50&Phosphorus=40&Potassium=40&Temperature=25&Humidity=60&pH=6&Rainfall=100"
```

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Detect plants from uploaded images
- `POST /analyze_leaf`: Analyze leaf characteristics
- `POST /predict_crop`: Get crop recommendations
- `GET /admin/usage_stats`: View usage statistics
- `GET /admin/leaf_analysis`: View leaf analysis data
- `POST /report_accuracy_feedback`: Report accuracy issues

## Model Details

### Plant Detection Model
- Uses Roboflow Inference API with custom trained model
- Supports multiple medicinal plant species
- Provides confidence scores and detailed predictions

### Crop Recommendation Model
- Random Forest classifier trained on crop dataset
- Features: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- Supports 22 different crop types
- Accuracy: ~95% (based on training data)

### Leaf Analysis
- Computer vision techniques using OpenCV
- Analyzes color, shape, texture, and dimensions
- Provides quantitative metrics for plant identification

## Data Sources

- Medicinal plant information: CSV files with healthcare benefits and traditional uses
- Crop data: Dataset with environmental parameters and crop recommendations
- Plant images: User-uploaded images for analysis
- External APIs: Roboflow for AI inference, IP geolocation for analytics

## Project Structure

```
Detection-and-Classification-of-Medicinal-Plants-Using-AI-main/
├── README.md
├── project_final_year/
│   ├── main.py                    # FastAPI application
│   ├── app.py                     # Flask app for crop recommendation
│   ├── sample.py                  # Data scraping script
│   ├── requirement.txt            # Python dependencies
│   ├── medicinal_plants.csv       # Basic plant information
│   ├── medicinal_plants_healthcare.csv  # Healthcare data
│   ├── plant_recommendation/
│   │   ├── app.py                 # Flask app
│   │   ├── index.html             # Web template
│   │   ├── model.pkl              # Trained ML model
│   │   ├── standscaler.pkl        # Standard scaler
│   │   ├── minmaxscaler.pkl       # Min-max scaler
│   │   └── Crop_recommendation.csv # Training data
│   ├── static/                    # Static web assets
│   └── uploads/                   # Uploaded images
```

## Technologies Used

- **Backend**: FastAPI, Flask
- **AI/ML**: Roboflow Inference, Scikit-learn, OpenCV
- **Data Processing**: Pandas, NumPy
- **Web**: HTML, CSS, JavaScript, Jinja2
- **Image Processing**: Pillow, OpenCV
- **APIs**: Requests, BeautifulSoup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Working Steps

1. **Data Collection**: Gather medicinal plant data and environmental parameters
2. **Model Training**: Train ML models for crop recommendation
3. **API Integration**: Set up Roboflow for plant detection
4. **Web Development**: Create user interfaces for interaction
5. **Testing**: Validate model accuracy and system performance
6. **Deployment**: Deploy the application for production use

## Requirements

- Python 3.8+
- Roboflow API key
- Internet connection for API calls
- Web browser for interface

For detailed package requirements, see `project_final_year/requirement.txt`.
