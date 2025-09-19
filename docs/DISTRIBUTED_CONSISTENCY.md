# Distributed Data Platform Consistency Checker

## Overview

This system provides comprehensive cross-platform data consistency checking across distributed data platforms including Snowflake, Oracle, Hadoop, PostgreSQL, MySQL, and other databases.

## Features

### ✅ Supported Platforms
- **Snowflake** - Cloud data warehouse
- **Oracle** - Enterprise database
- **Hadoop/Hive** - Big data platform
- **PostgreSQL** - Open source database
- **MySQL** - Popular database
- **Any JDBC-compatible database** (extensible)

### ✅ Consistency Checks
- **Schema Consistency** - Compare table structures, column names, data types
- **Row Count Consistency** - Verify record counts match across platforms
- **Data Consistency** - Sample-based comparison of actual data values
- **Data Type Compatibility** - Check for compatible data types across platforms

### ✅ Advanced Features
- **Parallel Processing** - Multi-threaded execution for faster checks
- **Configurable Sampling** - Adjustable sample sizes for data comparison
- **Tolerance Configuration** - Set acceptable variances for numeric and date fields
- **Detailed Reporting** - Comprehensive inconsistency analysis and recommendations
- **Export Capabilities** - Generate reports for compliance and audit

## Quick Start

### 1. Installation

Install additional dependencies for database connectivity:

```bash
pip install -r requirements_distributed.txt
```

### 2. Basic Usage

1. **Configure Platforms** - Add your data platform connections
2. **Map Tables** - Define which tables should be consistent across platforms
3. **Run Checks** - Execute consistency validation
4. **Review Results** - Analyze inconsistencies and take action

### 3. Platform-Specific Setup

#### Snowflake
```python
connection_config = ConnectionConfig(
    platform_type="snowflake",
    connection_string="your-account.snowflakecomputing.com",
    username="your_username",
    password="your_password",
    database_name="YOUR_DATABASE",
    schema_name="YOUR_SCHEMA",
    warehouse="YOUR_WAREHOUSE",
    role="YOUR_ROLE"
)
```

#### Oracle
```python
connection_config = ConnectionConfig(
    platform_type="oracle",
    connection_string="oracle-host:1521/ORCL",
    username="your_username",
    password="your_password",
    database_name="YOUR_DB",
    schema_name="YOUR_SCHEMA"
)
```

#### Hadoop/Hive
```python
connection_config = ConnectionConfig(
    platform_type="hadoop",
    connection_string="hadoop-cluster.company.com",
    username="your_username",
    password="your_password",
    database_name="your_warehouse"
)
```

#### PostgreSQL
```python
connection_config = ConnectionConfig(
    platform_type="postgresql",
    connection_string="postgres-host",
    username="your_username",
    password="your_password",
    database_name="your_database",
    schema_name="public"
)
```

## Configuration

### Table Mapping Example

```python
table_mapping = TableMapping(
    logical_name="customer_data",
    platforms={
        "snowflake_prod": {
            "database": "PROD_DB",
            "schema": "CUSTOMERS",
            "table_name": "CUSTOMER_MASTER"
        },
        "oracle_prod": {
            "database": "PROD_DB", 
            "schema": "CRM",
            "table_name": "CUSTOMERS"
        }
    },
    primary_keys=["customer_id"],
    comparison_columns=["customer_id", "first_name", "last_name", "email", "created_date"],
    business_keys=["customer_id", "email"],
    tolerance_config={
        "created_date": {"tolerance_hours": 24},
        "amount_fields": {"tolerance_percentage": 0.1}
    }
)
```

### Consistency Check Configuration

```python
config = ConsistencyCheckConfig(
    sample_size=10000,               # Number of rows to sample for data comparison
    enable_data_comparison=True,     # Compare actual data values
    enable_schema_comparison=True,   # Compare table schemas
    enable_count_comparison=True,    # Compare row counts
    numeric_tolerance=0.001,         # Tolerance for numeric comparisons (0.1%)
    date_tolerance_hours=24,         # Tolerance for date comparisons (24 hours)
    parallel_execution=True,         # Enable parallel processing
    max_workers=4                    # Number of parallel workers
)
```

## Programmatic Usage

### Basic Example

```python
from data_quality.cross_platform_checker import CrossPlatformConsistencyChecker, ConsistencyCheckConfig
from connectors.database_connectors import ConnectionConfig

# Create checker
checker = CrossPlatformConsistencyChecker(ConsistencyCheckConfig())

# Register platforms
snowflake_config = ConnectionConfig(
    platform_type="snowflake",
    connection_string="account.snowflakecomputing.com",
    username="user",
    password="pass",
    database_name="DB",
    warehouse="WH"
)

oracle_config = ConnectionConfig(
    platform_type="oracle", 
    connection_string="oracle-host:1521/ORCL",
    username="user",
    password="pass",
    database_name="DB"
)

checker.register_platform("snowflake_prod", snowflake_config)
checker.register_platform("oracle_prod", oracle_config)

# Add table mapping
mapping = TableMapping(
    logical_name="customers",
    platforms={
        "snowflake_prod": {"database": "DB", "schema": "SCHEMA", "table_name": "CUSTOMERS"},
        "oracle_prod": {"database": "DB", "schema": "SCHEMA", "table_name": "CUSTOMERS"}
    },
    primary_keys=["id"],
    comparison_columns=["id", "name", "email"],
    business_keys=["id"]
)

checker.add_table_mapping(mapping)

# Run consistency checks
results = checker.check_all_tables_consistency()

# Analyze results
for table_name, result in results.items():
    print(f"Table: {table_name}")
    print(f"Consistent: {result.is_consistent}")
    print(f"Score: {result.consistency_score}/100")
    
    if result.inconsistencies:
        print("Issues found:")
        for issue in result.inconsistencies:
            print(f"- {issue['severity']}: {issue['description']}")

# Cleanup
checker.cleanup()
```

## Inconsistency Types and Severity

### Schema Inconsistencies
- **Critical**: Missing required columns, incompatible data types
- **High**: Data type mismatches that could cause data loss
- **Medium**: Extra columns, nullable differences
- **Low**: Minor naming or formatting differences

### Data Inconsistencies  
- **Critical**: Row count differences > 10%
- **High**: Row count differences 5-10%, significant value differences
- **Medium**: Row count differences 1-5%, moderate value differences  
- **Low**: Minor value differences within tolerance

### Count Inconsistencies
- **Critical**: > 10% difference in row counts
- **High**: 5-10% difference in row counts
- **Medium**: 1-5% difference in row counts

## Best Practices

### 1. Platform Configuration
- Use dedicated read-only accounts for consistency checking
- Configure appropriate timeouts for large datasets
- Test connections before running large-scale checks

### 2. Table Mapping Strategy
- Start with critical business tables
- Define clear business keys for record matching
- Set appropriate tolerance levels for your data

### 3. Performance Optimization
- Use appropriate sample sizes (larger for critical tables)
- Enable parallel execution for multiple tables
- Schedule checks during low-traffic periods

### 4. Monitoring and Alerting
- Set up regular consistency checks
- Monitor consistency scores over time
- Alert on critical inconsistencies

## Troubleshooting

### Common Issues

#### Connection Problems
```
Error: Failed to connect to platform
```
**Solution**: Check network connectivity, credentials, and firewall settings

#### Schema Mismatches
```
Error: Table not found
```
**Solution**: Verify table names, schema names, and case sensitivity

#### Performance Issues
```
Query timeout
```
**Solution**: Reduce sample size, optimize network, or increase timeout settings

### Platform-Specific Notes

#### Snowflake
- Ensure warehouse is running and appropriately sized
- Consider using ACCOUNTADMIN role for metadata access
- Use proper case for object names (usually uppercase)

#### Oracle
- Install Oracle client libraries
- Configure TNS names or use full connection strings
- Handle character encoding properly

#### Hadoop/Hive
- Ensure Hive Metastore is accessible
- Configure Kerberos authentication if required
- Handle partitioned tables appropriately

## Security Considerations

### Data Access
- Use principle of least privilege for database accounts
- Consider data masking for sensitive columns
- Audit access to consistency checking results

### Credential Management
- Store credentials securely (not in code)
- Use environment variables or secure vaults
- Rotate credentials regularly

### Network Security
- Use VPN or private networks when possible
- Enable SSL/TLS for database connections
- Monitor network traffic for anomalies

## Extending the System

### Adding New Platforms

1. Create a new connector class inheriting from `BaseDatabaseConnector`
2. Implement required methods: `connect()`, `execute_query()`, `get_table_metadata()`, etc.
3. Register the connector in `DatabaseConnectorFactory`
4. Add platform-specific configuration options

### Custom Consistency Rules

1. Extend `InconsistencyDetails` for new types
2. Add custom comparison logic in `CrossPlatformConsistencyChecker`
3. Implement custom tolerance configurations
4. Update reporting to handle new rule types

## API Reference

### Core Classes

- `CrossPlatformConsistencyChecker` - Main consistency checking engine
- `ConnectionConfig` - Database connection configuration
- `TableMapping` - Cross-platform table mapping definition
- `ConsistencyCheckConfig` - Configuration for consistency checks
- `DataConsistencyResult` - Results of consistency check for a table

### Key Methods

- `register_platform()` - Add a data platform
- `add_table_mapping()` - Define table mapping
- `check_all_tables_consistency()` - Run all consistency checks
- `check_table_consistency()` - Check specific table
- `generate_consistency_report()` - Create detailed report

## Support and Contributing

For issues, feature requests, or contributions, please refer to the project documentation or contact the development team.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
