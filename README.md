# DataMend - Data Quality Management Platform

A comprehensive data quality management system that provides advanced data profiling, anomaly detection, and AI-powered insights across multiple data sources including files, databases, and big data platforms.

## Overview

DataMend is designed to help organizations ensure data quality across their entire data infrastructure. It provides automated data profiling, statistical anomaly detection, and AI-driven recommendations to improve data quality and reliability.

## Key Features

### Data Source Connectivity
- **File Support**: CSV and Excel files with automatic encoding detection
- **Database Integration**: Snowflake and Oracle Cloud databases
- **Big Data Platform**: HDFS (Hadoop Distributed File System) support
- **Flexible Authentication**: Environment-based configuration and manual setup

### Data Quality Analysis
- **Comprehensive Profiling**: Statistical analysis using ydata-profiling
- **Anomaly Detection**: Multiple algorithms including Z-score, IQR, and modified Z-score
- **Quality Scoring**: Automated quality assessment with configurable thresholds
- **Data Lineage**: Visualization of data dependencies for database sources

### AI-Powered Insights
- **LLM Integration**: Support for Google Gemini and OpenAI models
- **Contextual Recommendations**: AI-generated suggestions for data quality improvements
- **Priority-based Analysis**: Intelligent ranking of quality issues
- **Interactive Interface**: Streamlit-based web dashboard

### Enterprise Features
- **Scalable Processing**: Handles large datasets with chunked processing
- **Real-time Analysis**: Live data quality monitoring
- **Configurable Thresholds**: Customizable anomaly detection sensitivity
- **Session Management**: Persistent analysis sessions

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mesbahul-Islam/datamend.git
   cd datamend
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables** (Optional):
   Copy and configure the `.env` file with your database credentials and API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your specific configuration
   ```

5. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

6. **Access the Dashboard**:
   Open your browser to `http://localhost:8501`

## Configuration

### Environment Variables

#### LLM Configuration
```bash
# Choose your LLM provider
LLM_PROVIDER=gemini  # or openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.0-flash  # or gpt-3.5-turbo
```

#### Snowflake Configuration
```bash
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USERNAME=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=PUBLIC
```

#### Oracle Cloud Configuration
```bash
ORACLE_CONNECTION_STRING=(description=(retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=your_host.oraclecloud.com))(connect_data=(service_name=your_service.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))
ORACLE_USERNAME=your_username
ORACLE_PASSWORD=your_password
```

#### HDFS Configuration
```bash
HDFS_NAMENODE_URL=http://namenode:9870
HDFS_USERNAME=hdfs
HDFS_TIMEOUT=30
```

## Usage Guide

### 1. Data Source Connection

#### File Upload
- Select "CSV Files" or "Excel Files" from the dropdown
- Use the file uploader to browse and select your data files
- Files are automatically processed and validated

#### Database Connection
- Choose "Snowflake Database" or "Oracle Cloud Database"
- Configure connection settings manually or use environment variables
- Test connection and browse available tables
- Select tables and configure data loading parameters

#### HDFS Integration
- Select "HDFS" from the data source dropdown
- Configure NameNode URL and authentication
- Browse HDFS directories and select data files
- Supports CSV, Parquet, JSON, and Excel files in HDFS

### 2. Data Analysis Workflow

#### Data Overview
- Basic dataset information and statistics
- Data type analysis and missing value assessment
- Quality metrics including completeness, consistency, and validity scores

#### Data Profiling
- **Quick Summary**: Essential data quality metrics
- **Detailed Report**: Comprehensive ydata-profiling analysis
- **Statistical Outliers**: Automated anomaly detection
- **Analytics Quality**: Overall quality assessment with scoring

#### AI Recommendations
- Configure LLM provider (Google Gemini or OpenAI)
- Generate contextual data quality insights
- Receive prioritized recommendations for improvement
- Export analysis results

#### Data Lineage (Database Sources)
- Visualize data dependencies and relationships
- Understand data flow and transformations
- AI-powered lineage analysis and insights

### 3. Advanced Configuration

#### Anomaly Detection
- Adjust threshold sensitivity (1.0 to 5.0)
- Higher values mean stricter anomaly detection
- Configure detection algorithms

#### Performance Tuning
- Set appropriate data loading limits
- Configure timeout settings for large datasets
- Optimize memory usage with chunked processing

## Architecture

### Project Structure
```
datamend/
├── main.py                           # Main application entry point
├── requirements.txt                  # Python dependencies
├── .env                             # Environment configuration
├── data/                            # Sample datasets
│   └── sample_credit_data.csv
├── src/
│   ├── ui/                          # User interface modules
│   │   ├── data_source.py           # Data source management
│   │   ├── data_profiling.py        # Profiling interface
│   │   ├── anomaly_detection.py     # Anomaly detection UI
│   │   ├── data_lineage.py          # Lineage visualization
│   │   └── data_sources/            # Modular data source handlers
│   │       ├── csv_handler.py
│   │       ├── excel_handler.py
│   │       ├── snowflake_handler.py
│   │       ├── oracle_handler.py
│   │       └── hdfs_handler.py
│   ├── connectors/                  # Data connectivity layer
│   │   └── data_connectors.py       # Unified connector framework
│   ├── data_quality/                # Quality analysis engine
│   │   ├── engine.py               # Core quality processing
│   │   └── anomaly_detector.py     # Statistical analysis
│   ├── llm/                        # AI integration
│   │   └── analyzer.py             # LLM-powered insights
│   ├── metadata/                   # Data metadata extraction
│   └── utils/                      # Utility functions
└── tests/                          # Test suite
```

### Technology Stack
- **Frontend**: Streamlit for web interface
- **Data Processing**: Pandas, NumPy for data manipulation
- **Profiling**: ydata-profiling for comprehensive analysis
- **Visualization**: Plotly, Matplotlib for charts and graphs
- **Database Connectivity**: Native drivers for Snowflake, Oracle, HDFS
- **AI Integration**: Google Gemini and OpenAI API support
- **Session Management**: Streamlit session state

## Dependencies

### Core Requirements
- streamlit==1.49.1
- pandas>=1.5.0
- numpy>=1.24.0
- ydata-profiling==4.16.0
- plotly==6.3.0
- matplotlib>=3.6.0

### Database Connectors
- snowflake-connector-python==3.17.3
- oracledb>=2.0.0
- hdfs3>=0.3.1

### AI Integration
- google-genai==1.38.0
- openai>=1.0.0

### Additional Libraries
- python-dotenv==1.1.1
- openpyxl>=3.0.0
- xlrd>=2.0.0
- networkx>=3.0.0
- SQLAlchemy==2.0.43

## Performance Considerations

### Large Dataset Handling
- Configurable row limits for initial analysis
- Memory-efficient chunked processing
- Optimized data loading strategies

### Database Connections
- Connection pooling for improved performance
- Timeout configuration for network reliability
- Automatic connection management

### HDFS Integration
- Supports multiple file formats (CSV, Parquet, JSON, Excel)
- Directory browsing with depth limitations
- Efficient file transfer mechanisms

## Troubleshooting

### Common Issues

#### Connection Problems
- Verify network connectivity and firewall settings
- Check credential configuration in environment variables
- Ensure database services are running and accessible

#### Performance Issues
- Reduce data loading limits for large datasets
- Increase timeout settings for slow connections
- Monitor memory usage during processing

#### HDFS Connectivity
- Verify NameNode URL and port configuration
- Check HDFS service availability
- Ensure proper authentication credentials

### Error Resolution
- Review application logs for detailed error messages
- Verify environment variable configuration
- Check data format compatibility

## License

This project is designed for demonstration and educational purposes. Please review and adapt the code for production use according to your organization's requirements and security policies.

## Support

For technical issues, feature requests, or contributions:
1. Review the troubleshooting guide above
2. Check the sample data and configuration examples
3. Examine the test cases for usage patterns
4. Refer to the inline documentation and code comments
