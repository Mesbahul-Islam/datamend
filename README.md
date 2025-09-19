# Data Quality Engine - Simplified

A streamlined data quality management system focused on core functionality.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Data Profiling**: Comprehensive analysis of data quality metrics
- **Anomaly Detection**: Statistical detection of outliers and anomalies
- **AI Recommendations**: LLM-powered suggestions for data quality improvements
- **Interactive Dashboard**: Streamlit-based web interface

## Project Structure

```
hackathon_2/
├── app_simple.py              # Main Streamlit application
├── src/
│   ├── data_quality/
│   │   ├── engine.py          # Core data quality engine
│   │   └── anomaly_detector.py # Statistical anomaly detection
│   ├── connectors/
│   │   └── data_connectors.py # Data source connectors (CSV, Excel)
│   └── llm/
│       └── analyzer.py        # LLM-based analysis and recommendations
├── data/
│   └── sample_credit_data.csv # Sample dataset
├── test_data.csv              # Test dataset
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv data_quality_env
   source data_quality_env/bin/activate  # On Windows: data_quality_env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app_simple.py
   ```

3. **Open your browser** to `http://localhost:8501`

## Usage

1. **Upload Data**: Use the sidebar to upload CSV or Excel files
2. **Configure Engine**: Adjust chunk size, workers, and anomaly threshold
3. **Run Analysis**: Click "Run Data Quality Analysis" 
4. **Review Results**: Explore quality metrics, column profiles, and anomalies
5. **Get AI Insights**: View LLM-generated recommendations

## Core Components

### Data Quality Engine
- Multithreaded processing for large datasets
- Configurable chunk sizes for memory efficiency
- Comprehensive quality scoring algorithm

### Anomaly Detection
- Z-score based detection
- Interquartile Range (IQR) method
- Modified Z-score for robust detection

### Data Connectors
- CSV file support with encoding detection
- Excel file support (multiple sheets)
- Automatic data type inference

### LLM Analyzer
- AI-powered data quality insights
- Contextual recommendations
- Priority-based suggestions

## Requirements

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Plotly
- OpenAI (optional, for LLM features)

## Sample Data

The project includes sample datasets:
- `data/sample_credit_data.csv` - Credit risk dataset
- `test_data.csv` - Small test dataset for validation

## Performance

- Handles datasets up to millions of rows
- Configurable parallel processing
- Memory-efficient chunked processing
- Real-time progress tracking

---

**Simplified and focused on core data quality functionality.**

## 📄 License

This project is designed for hackathon demonstration purposes. Please review and adapt for production use.

## 🆘 Support

For questions, issues, or feature requests:
1. Check the [Quick Start Guide](QUICKSTART.md)
2. Review the test cases for usage examples
3. Examine the sample data generation for data structure requirements

## 🔮 Future Enhancements

- **Additional Connectors**: MongoDB, Snowflake, BigQuery support
- **Advanced Analytics**: Time series data quality analysis
- **Machine Learning**: Predictive data quality modeling
- **API Integration**: RESTful API for programmatic access
- **Scheduling**: Automated quality monitoring and alerts
- **Data Lineage**: Track data transformation history
