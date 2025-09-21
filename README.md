# DataMend - Data Quality Management Platform

A comprehensive, cross-platform data quality management system that provides advanced data profiling, anomaly detection, and AI-powered insights across multiple data sources including files, databases, and big data platforms.

## Overview

DataMend is designed to help organizations ensure data quality across their entire data infrastructure, regardless of their operating system or deployment environment. Built with Python and web technologies, it provides automated data profiling, statistical anomaly detection, and AI-driven recommendations to improve data quality and reliability across Windows, macOS, Linux, and containerized environments.

## Platform Independence

DataMend is built with cross-platform compatibility as a core principle:

### Operating System Support
- **Windows**: Full native support with PowerShell and Command Prompt
- **macOS**: Native support with Terminal and shell environments
- **Linux**: Complete compatibility with all major distributions (Ubuntu, CentOS, RHEL, etc.)
- **Unix-like Systems**: Support for FreeBSD, OpenBSD, and other Unix variants

### Deployment Options
- **Local Development**: Run directly on any OS with Python 3.10+
- **Docker Containers**: Platform-agnostic containerized deployment
- **Cloud Platforms**: Deploy on AWS, Azure, GCP, Oracle Cloud, or any cloud provider
- **On-Premises**: Enterprise deployment in private data centers
- **Hybrid Environments**: Mix of cloud and on-premises infrastructure

### Browser Compatibility
- **Universal Access**: Works with any modern web browser
- **Cross-Device**: Responsive design for desktop, tablet, and mobile devices
- **No Platform-Specific Dependencies**: Pure web-based interface accessible from any device

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
- **Python 3.10 or higher** (Available on all platforms)
- **pip package manager** (Included with Python)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### Platform-Specific Setup

#### Windows
```cmd
# Using Command Prompt or PowerShell
git clone https://github.com/Mesbahul-Islam/datamend.git
cd datamend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```

#### macOS
```bash
# Using Terminal
git clone https://github.com/Mesbahul-Islam/datamend.git
cd datamend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

#### Linux/Unix
```bash
# Using Terminal/Shell
git clone https://github.com/Mesbahul-Islam/datamend.git
cd datamend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

### Universal Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mesbahul-Islam/datamend.git
   cd datamend
   ```

2. **Create Virtual Environment**:
   - **Windows**: `python -m venv .venv && .venv\Scripts\activate`
   - **macOS/Linux**: `python3 -m venv .venv && source .venv/bin/activate`

3. **Install Dependencies** (Universal):
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables** (Optional):
   Copy and configure the `.env` file with your database credentials and API keys:
   - **Windows**: `copy .env.example .env`
   - **macOS/Linux**: `cp .env.example .env`
   ```bash
   # Edit .env with your specific configuration using your preferred editor
   ```

5. **Run the Application** (All Platforms):
   ```bash
   streamlit run main.py
   ```

6. **Access the Dashboard**:
   Open your browser to `http://localhost:8501` (works on all platforms)

### Docker Deployment (Platform Independent)

For the most consistent cross-platform experience, use Docker:

#### Quick Start with Docker
```bash
# Development environment (all platforms)
docker-compose -f docker-compose.dev.yml up --build

# Production environment (all platforms)
docker-compose -f docker-compose.prod.yml up -d
```

#### Platform-Specific Docker Installation
- **Windows**: Install Docker Desktop for Windows
- **macOS**: Install Docker Desktop for Mac
- **Linux**: Install Docker Engine (varies by distribution)

See `DOCKER.md` for comprehensive Docker deployment instructions.

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

### Technology Stack (Cross-Platform)
- **Frontend**: Streamlit (web-based, platform-independent interface)
- **Backend**: Python (runs on Windows, macOS, Linux, Unix)
- **Data Processing**: Pandas, NumPy (cross-platform data manipulation)
- **Profiling**: ydata-profiling (universal data analysis)
- **Visualization**: Plotly, Matplotlib (browser-based charts, works everywhere)
- **Database Connectivity**: Native drivers with cross-platform support
  - Snowflake: Works on all platforms
  - Oracle: Cross-platform oracledb driver
  - HDFS: Platform-independent Python clients
- **AI Integration**: Web-based APIs (Google Gemini, OpenAI) - platform agnostic
- **Session Management**: Browser-based (works on any device/OS)
- **Containerization**: Docker support for consistent deployment across environments

## Dependencies (Cross-Platform Compatible)

All dependencies are designed to work across Windows, macOS, Linux, and Unix systems:

### Core Requirements (Universal)
- streamlit==1.49.1 (web framework - platform independent)
- pandas>=1.5.0 (cross-platform data processing)
- numpy>=1.24.0 (universal numerical computing)
- ydata-profiling==4.16.0 (works on all platforms)
- plotly==6.3.0 (browser-based visualization)
- matplotlib>=3.6.0 (cross-platform plotting)

### Database Connectors (Multi-Platform)
- snowflake-connector-python==3.17.3 (Windows, macOS, Linux support)
- oracledb>=2.0.0 (cross-platform Oracle connectivity)
- hdfs3>=0.3.1 (platform-independent HDFS client)

### AI Integration (API-Based, Platform Agnostic)
- google-genai==1.38.0 (web API - works everywhere)
- openai>=1.0.0 (web API - universal access)

### Additional Libraries (Cross-Platform)
- python-dotenv==1.1.1 (environment configuration)
- openpyxl>=3.0.0 (Excel file support - all platforms)
- xlrd>=2.0.0 (legacy Excel support)
- networkx>=3.0.0 (graph processing)
- SQLAlchemy==2.0.43 (database abstraction layer)

## Performance Considerations (Platform Optimized)

### Large Dataset Handling (Universal)
- Configurable row limits for initial analysis
- Memory-efficient chunked processing (optimized for each platform)
- Cross-platform data loading strategies

### Database Connections (Multi-Platform)
- Connection pooling for improved performance across all systems
- Platform-specific timeout configuration for network reliability
- Automatic connection management (Windows, macOS, Linux compatible)

### HDFS Integration (Cross-Platform)
- Supports multiple file formats (CSV, Parquet, JSON, Excel) on all platforms
- Directory browsing with depth limitations
- Efficient file transfer mechanisms optimized for different OS environments

### Resource Optimization
- **Windows**: Optimized for Windows file system and memory management
- **macOS**: Tuned for macOS performance characteristics
- **Linux**: Optimized for Linux container and server environments
- **Docker**: Consistent performance across all host platforms

## Troubleshooting (Platform-Specific Guidance)

### Common Issues

#### Installation Problems
**Windows:**
- Use PowerShell as Administrator if permission issues occur
- Ensure Python is added to PATH environment variable
- Consider using Windows Subsystem for Linux (WSL) for Unix-like experience

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for Python if system version is outdated
- Check Python installation with `python3 --version`

**Linux:**
- Install build essentials: `sudo apt-get install build-essential` (Ubuntu/Debian)
- Ensure Python development headers: `sudo apt-get install python3-dev`
- For RHEL/CentOS: `sudo yum install python3-devel gcc`

#### Connection Problems (Universal)
- Verify network connectivity and firewall settings (platform-specific firewall tools)
- Check credential configuration in environment variables
- Ensure database services are running and accessible from your platform

#### Performance Issues (Platform-Optimized Solutions)
- **Windows**: Monitor Task Manager for memory usage, adjust virtual memory settings
- **macOS**: Use Activity Monitor, consider increasing Docker memory allocation
- **Linux**: Monitor with `htop`, adjust container limits in production environments
- **All Platforms**: Reduce data loading limits for large datasets

#### Docker-Specific Issues
- **Windows**: Enable WSL 2 backend for better performance
- **macOS**: Allocate sufficient memory in Docker Desktop preferences
- **Linux**: Ensure Docker daemon is running and user has proper permissions

### Platform-Specific Error Resolution
- **Windows**: Check Event Viewer for system-level errors
- **macOS**: Review Console app for application logs
- **Linux**: Check system logs with `journalctl` or `/var/log/`
- **All Platforms**: Review application logs in Streamlit interface

### Environment Variable Configuration
- **Windows**: Use System Properties → Environment Variables or PowerShell `$env:` commands
- **macOS/Linux**: Use shell profile files (`.bashrc`, `.zshrc`) or export commands
- **Docker**: Configure through docker-compose environment sections

## License

This project is designed for demonstration and educational purposes. Please review and adapt the code for production use according to your organization's requirements and security policies.

## Support

For technical issues, feature requests, or contributions:
1. Review the troubleshooting guide above
2. Check the sample data and configuration examples
3. Examine the test cases for usage patterns
4. Refer to the inline documentation and code comments
