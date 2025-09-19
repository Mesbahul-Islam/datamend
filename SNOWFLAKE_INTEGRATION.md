# Snowflake Integration Guide

## Overview
The DataMend application now supports connecting to Snowflake databases as a data source alongside CSV and Excel file uploads. This allows you to:

- Connect directly to your Snowflake account
- Browse available tables in your schema
- Load data using table selection or custom SQL queries
- Run the same data profiling and anomaly detection on Snowflake data as with uploaded files

## Prerequisites

### Required Python Packages
The following packages are automatically installed:
- `snowflake-connector-python` - Official Snowflake Python connector
- `snowflake-sqlalchemy` - SQLAlchemy support for Snowflake

### Snowflake Account Requirements
You need:
- A Snowflake account with valid credentials
- Access to at least one database and warehouse
- Appropriate permissions to read data from tables

## How to Use

### 1. Access Snowflake Connection
1. Start the application: `streamlit run main.py`
2. Navigate to the **Data Source** tab
3. Select **"Snowflake Database"** from the dropdown

### 2. Connection Setup
Fill in the required connection details:

- **Account Identifier**: Your Snowflake account identifier (e.g., `abc12345.us-east-1`)
- **Username**: Your Snowflake username
- **Password**: Your Snowflake password
- **Warehouse**: The warehouse to use (e.g., `COMPUTE_WH`)
- **Database**: The database name to connect to
- **Schema**: The schema name (defaults to `PUBLIC`)

### 3. Connect to Snowflake
Click **"Connect to Snowflake"** to establish the connection. The application will:
- Test the connection with your credentials
- Display available tables in the specified schema
- Show connection status and any error messages

### 4. Load Data

#### Option A: Select from Tables
1. Choose **"Select from Tables"** method
2. Select a table from the dropdown list
3. Optionally view table information (rows, columns, data types)
4. Choose whether to limit the number of rows loaded
5. Click **"Load Data from Snowflake"**

#### Option B: Custom SQL Query
1. Choose **"Custom SQL Query"** method
2. Write your SQL query in the text area
3. Optionally add a row limit for performance
4. Click **"Load Data from Snowflake"**

### 5. Data Analysis
Once data is loaded from Snowflake:
- It appears in the **Loaded Datasets** section
- You can switch between Snowflake data and uploaded files
- All analysis features work the same way:
  - **Data Profiling** tab: Generate comprehensive ydata-profiling reports
  - **Anomaly Detection** tab: Detect outliers using ydata-profiling
  - **AI Recommendations** tab: Get insights and recommendations

## Example Workflow

### Connecting to Snowflake
```
Account: mycompany.us-east-1
Username: analyst_user
Password: [your_password]
Warehouse: ANALYTICS_WH
Database: SALES_DB
Schema: REPORTS
```

### Example SQL Queries
```sql
-- Load recent sales data
SELECT * FROM sales_transactions 
WHERE transaction_date >= '2024-01-01'
LIMIT 10000

-- Load aggregated monthly data
SELECT 
    DATE_TRUNC('month', transaction_date) as month,
    SUM(amount) as total_sales,
    COUNT(*) as transaction_count
FROM sales_transactions 
WHERE transaction_date >= '2023-01-01'
GROUP BY month
ORDER BY month
```

## Features

### Data Integration
- **Seamless Integration**: Snowflake data appears alongside uploaded files
- **Dataset Switching**: Easy switching between different data sources
- **Metadata Preservation**: SQL queries and source information are preserved

### Performance Considerations
- **Row Limits**: Default 10,000 row limit for performance
- **Custom Queries**: Write optimized queries for large datasets
- **Memory Management**: Large datasets are handled efficiently

### Error Handling
- **Connection Validation**: Immediate feedback on connection issues
- **Query Validation**: SQL syntax and permission error messages
- **Helpful Tips**: Context-aware suggestions for common issues

## Troubleshooting

### Common Connection Issues
1. **Invalid Account Identifier**: Check the format (usually includes region)
2. **Authentication Failures**: Verify username and password
3. **Warehouse Access**: Ensure you have access to the specified warehouse
4. **Network Issues**: Check firewall settings and network connectivity

### Performance Tips
1. **Use LIMIT clauses** for large tables
2. **Filter data** with WHERE clauses when possible
3. **Select specific columns** instead of using SELECT *
4. **Monitor memory usage** in the dataset metrics

### SQL Query Guidelines
- Include `LIMIT` for exploratory analysis
- Use proper table and column names (case-sensitive)
- Consider using qualified names: `schema.table_name`
- Test queries in Snowflake console first for complex queries

## Security Notes

- **Credentials**: Passwords are not stored; re-enter for each session
- **Connection Scope**: Connections are session-specific
- **Data Privacy**: Data is processed locally in your environment
- **Network Security**: Uses secure TLS connections to Snowflake

## Next Steps

After loading Snowflake data, you can:
1. **Generate Profiles**: Use ydata-profiling for comprehensive data analysis
2. **Detect Anomalies**: Apply statistical anomaly detection
3. **Get Recommendations**: Use AI-powered insights
4. **Compare Datasets**: Analyze multiple data sources side by side

For support or questions about Snowflake integration, refer to the main documentation or contact your administrator.
