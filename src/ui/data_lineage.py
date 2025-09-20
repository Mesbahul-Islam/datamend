"""
Data Lineage Module
Handles Snowflake data lineage tracking and visualization
"""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import snowflake.connector
from typing import Dict, List, Tuple, Optional
import logging
from src.ui.data_profiling import display_ai_recommendations_section

logger = logging.getLogger(__name__)


def data_lineage_tab():
    """Data lineage analysis tab for Snowflake data"""
    st.header("Data Lineage")
    
    # Check if current data is from Snowflake
    if not st.session_state.get('data_source') == 'snowflake':
        st.info("Data lineage is only available for Snowflake data sources.")
        st.write("Load data from Snowflake to view lineage information.")
        return
    
    # Check if we have Snowflake connection info
    snowflake_config = st.session_state.get('snowflake_config')
    if not snowflake_config:
        st.warning("Snowflake connection information not found.")
        st.write("Please reconnect to Snowflake to enable lineage tracking.")
        return
    
    # Get table information from session state
    current_table = st.session_state.get('current_table', '')
    current_schema = st.session_state.get('current_schema', 'PUBLIC')
    current_database = st.session_state.get('current_database', '')
    
    if not current_table:
        st.warning("Current table information not available.")
        return
    
    st.info(f"Analyzing lineage for: {current_database}.{current_schema}.{current_table}")
    
    # Lineage controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Target Table:** `{current_table}`")
        st.write(f"**Schema:** `{current_schema}`")
        st.write(f"**Database:** `{current_database}`")
    
    with col2:
        if st.button("Fetch Lineage", type="primary"):
            fetch_and_display_lineage(snowflake_config, current_database, current_schema, current_table)
    
    # Show results if available
    lineage_data = st.session_state.get('lineage_data')
    if lineage_data is not None and not lineage_data.empty:
        display_lineage_results()


def fetch_and_display_lineage(config: Dict, database: str, schema: str, table: str):
    """Fetch lineage data from Snowflake and display visualization"""
    try:
        with st.spinner("Fetching data lineage from Snowflake..."):
            lineage_data = get_snowflake_lineage(config, database, schema, table)
            
            if lineage_data.empty:
                st.warning("No lineage data found for this table.")
                st.session_state.lineage_data = None
                return
            
            st.session_state.lineage_data = lineage_data
            st.success(f"Found {len(lineage_data)} lineage relationships")
            
    except Exception as e:
        st.error(f"Error fetching lineage data: {str(e)}")
        logger.error(f"Lineage fetch error: {e}")


def get_snowflake_lineage(config: Dict, database: str, schema: str, table: str) -> pd.DataFrame:
    """Query Snowflake for actual object dependencies and transformation history"""
    try:
        # Connect to Snowflake with ACCOUNT_USAGE schema for lineage queries
        conn = snowflake.connector.connect(
            user=config['username'],
            password=config['password'],
            account=config['account'],
            warehouse=config.get('warehouse', 'COMPUTE_WH'),
            database='SNOWFLAKE',  # Must use SNOWFLAKE database for ACCOUNT_USAGE
            schema='ACCOUNT_USAGE'
        )
        
        cur = conn.cursor()
        
        # Try to get actual object dependencies from OBJECT_DEPENDENCIES view
        dependency_results = []
        dependency_columns = []
        
        try:
            dependencies_query = """
            SELECT 
                od.REFERENCED_DATABASE as SOURCE_DATABASE,
                od.REFERENCED_SCHEMA as SOURCE_SCHEMA, 
                od.REFERENCED_OBJECT_NAME as SOURCE_TABLE,
                od.REFERENCED_OBJECT_DOMAIN as SOURCE_TYPE,
                od.REFERENCING_DATABASE as TARGET_DATABASE,
                od.REFERENCING_SCHEMA as TARGET_SCHEMA,
                od.REFERENCING_OBJECT_NAME as TARGET_TABLE,
                od.REFERENCING_OBJECT_DOMAIN as TARGET_TYPE,
                'DEPENDENCY' as RELATIONSHIP_TYPE,
                CURRENT_TIMESTAMP() as DEPENDENCY_CREATED,
                'OBJECT_DEPENDENCIES' as SOURCE_VIEW
            FROM SNOWFLAKE.ACCOUNT_USAGE.OBJECT_DEPENDENCIES od
            WHERE (
                (UPPER(od.REFERENCED_DATABASE) = UPPER(%s) 
                 AND UPPER(od.REFERENCED_SCHEMA) = UPPER(%s) 
                 AND UPPER(od.REFERENCED_OBJECT_NAME) = UPPER(%s))
                OR 
                (UPPER(od.REFERENCING_DATABASE) = UPPER(%s) 
                 AND UPPER(od.REFERENCING_SCHEMA) = UPPER(%s) 
                 AND UPPER(od.REFERENCING_OBJECT_NAME) = UPPER(%s))
            )
            AND od.REFERENCING_OBJECT_DOMAIN = 'TABLE'
            ORDER BY od.REFERENCING_OBJECT_NAME DESC
            """
            
            cur.execute(dependencies_query, (database, schema, table, database, schema, table))
            dependency_results = cur.fetchall()
            dependency_columns = [desc[0] for desc in cur.description]
            
        except Exception as dep_error:
            logger.warning(f"OBJECT_DEPENDENCIES query failed (may not be available): {dep_error}")
            dependency_results = []
            dependency_columns = []
        
        # Get actual access history for transformations (focus on table modifications)
        access_results = []
        access_columns = []
        
        try:
            # Focus on queries that actually modify or create the target table
            access_history_query = """
            SELECT 
                qh.QUERY_ID,
                qh.START_TIME as QUERY_START_TIME,
                qh.USER_NAME,
                qh.QUERY_TEXT,
                qh.QUERY_TYPE,
                qh.EXECUTION_STATUS,
                qh.TOTAL_ELAPSED_TIME,
                qh.WAREHOUSE_NAME,
                qh.DATABASE_NAME,
                qh.SCHEMA_NAME,
                'QUERY_HISTORY' as SOURCE_VIEW,
                '' as DIRECT_OBJECTS_ACCESSED,
                '' as BASE_OBJECTS_ACCESSED,
                '' as OBJECTS_MODIFIED
            FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
            WHERE qh.START_TIME >= DATEADD('days', -90, CURRENT_TIMESTAMP())
            AND qh.EXECUTION_STATUS = 'SUCCESS'
            AND (
                -- Queries that create or modify our target table
                (qh.QUERY_TYPE IN ('INSERT', 'UPDATE', 'MERGE', 'CREATE_TABLE_AS_SELECT', 'CREATE_TABLE')
                 AND (UPPER(qh.QUERY_TEXT) LIKE UPPER('%%INTO%%' || %s || '%%')
                      OR UPPER(qh.QUERY_TEXT) LIKE UPPER('%%UPDATE%%' || %s || '%%')
                      OR UPPER(qh.QUERY_TEXT) LIKE UPPER('%%TABLE%%' || %s || '%%')
                      OR UPPER(qh.QUERY_TEXT) LIKE UPPER('%%' || %s || '.' || %s || '.' || %s || '%%')))
                OR
                -- CTAS queries that reference other tables to create our target
                (qh.QUERY_TYPE = 'CREATE_TABLE_AS_SELECT'
                 AND UPPER(qh.QUERY_TEXT) LIKE UPPER('%%CREATE%%TABLE%%' || %s || '%%AS%%SELECT%%'))
                OR
                -- Select queries that our table might be based on (recent large selects)
                (qh.QUERY_TYPE = 'SELECT' 
                 AND qh.TOTAL_ELAPSED_TIME > 10000
                 AND UPPER(qh.QUERY_TEXT) LIKE UPPER('%%FROM%%' || %s || '%%'))
            )
            ORDER BY qh.START_TIME DESC
            LIMIT 50
            """
            
            # Pass parameters for all placeholders
            cur.execute(access_history_query, (
                table, table, table,  # for INTO, UPDATE, TABLE patterns
                database, schema, table,  # for full table name
                table,  # for CTAS
                table   # for SELECT FROM
            ))
            access_results = cur.fetchall()
            access_columns = [desc[0] for desc in cur.description]
            
        except Exception as access_error:
            logger.warning(f"Access history query failed: {access_error}")
            
            # Simplified fallback query
            try:
                fallback_query = """
                SELECT 
                    qh.QUERY_ID,
                    qh.START_TIME as QUERY_START_TIME,
                    qh.USER_NAME,
                    qh.QUERY_TEXT,
                    qh.QUERY_TYPE,
                    qh.EXECUTION_STATUS,
                    qh.TOTAL_ELAPSED_TIME,
                    qh.WAREHOUSE_NAME,
                    qh.DATABASE_NAME,
                    qh.SCHEMA_NAME,
                    'QUERY_HISTORY_SIMPLE' as SOURCE_VIEW
                FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
                WHERE qh.START_TIME >= DATEADD('days', -30, CURRENT_TIMESTAMP())
                AND qh.EXECUTION_STATUS = 'SUCCESS'
                AND UPPER(qh.QUERY_TEXT) LIKE UPPER('%%' || %s || '%%')
                AND qh.QUERY_TYPE IN ('INSERT', 'UPDATE', 'CREATE_TABLE_AS_SELECT', 'SELECT', 'MERGE')
                ORDER BY qh.START_TIME DESC
                LIMIT 30
                """
                
                cur.execute(fallback_query, (table,))
                access_results = cur.fetchall()
                access_columns = [desc[0] for desc in cur.description]
                
            except Exception as fallback_error:
                logger.warning(f"Fallback query also failed: {fallback_error}")
                access_results = []
                access_columns = []
        
        # Try to get DDL history for table transformations (may not be available in all accounts)
        ddl_results = []
        ddl_columns = []
        
        try:
            ddl_history_query = """
            SELECT 
                ddl.QUERY_ID,
                ddl.QUERY_TEXT,
                ddl.DATABASE_NAME,
                ddl.SCHEMA_NAME,
                ddl.OBJECT_NAME,
                ddl.OBJECT_TYPE,
                ddl.DDL_ACTION,
                ddl.EXECUTED_BY,
                ddl.EXECUTION_TIME,
                'DDL_HISTORY' as SOURCE_VIEW
            FROM SNOWFLAKE.ACCOUNT_USAGE.DDL_HISTORY ddl
            WHERE UPPER(ddl.DATABASE_NAME) = UPPER(%s)
            AND UPPER(ddl.SCHEMA_NAME) = UPPER(%s)  
            AND UPPER(ddl.OBJECT_NAME) = UPPER(%s)
            AND ddl.EXECUTION_TIME >= DATEADD('days', -90, CURRENT_TIMESTAMP())
            AND ddl.DDL_ACTION IN ('CREATE', 'ALTER', 'DROP')
            AND ddl.OBJECT_TYPE = 'TABLE'
            ORDER BY ddl.EXECUTION_TIME DESC
            LIMIT 50
            """
            
            cur.execute(ddl_history_query, (database, schema, table))
            ddl_results = cur.fetchall()
            ddl_columns = [desc[0] for desc in cur.description]
            
        except Exception as ddl_error:
            logger.warning(f"DDL_HISTORY query failed (may not be available): {ddl_error}")
            
            # Fallback: Try to get DDL information from QUERY_HISTORY
            try:
                fallback_ddl_query = """
                SELECT 
                    qh.QUERY_ID,
                    qh.QUERY_TEXT,
                    %s as DATABASE_NAME,
                    %s as SCHEMA_NAME,
                    %s as OBJECT_NAME,
                    'TABLE' as OBJECT_TYPE,
                    CASE 
                        WHEN UPPER(qh.QUERY_TEXT) LIKE '%%CREATE%%TABLE%%' THEN 'CREATE'
                        WHEN UPPER(qh.QUERY_TEXT) LIKE '%%ALTER%%TABLE%%' THEN 'ALTER'
                        WHEN UPPER(qh.QUERY_TEXT) LIKE '%%DROP%%TABLE%%' THEN 'DROP'
                        ELSE 'UNKNOWN'
                    END as DDL_ACTION,
                    qh.USER_NAME as EXECUTED_BY,
                    qh.START_TIME as EXECUTION_TIME,
                    'QUERY_HISTORY_DDL' as SOURCE_VIEW
                FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
                WHERE qh.START_TIME >= DATEADD('days', -90, CURRENT_TIMESTAMP())
                AND (
                    UPPER(qh.QUERY_TEXT) LIKE UPPER('%%CREATE%%TABLE%%' || %s || '%%')
                    OR UPPER(qh.QUERY_TEXT) LIKE UPPER('%%ALTER%%TABLE%%' || %s || '%%')
                    OR UPPER(qh.QUERY_TEXT) LIKE UPPER('%%DROP%%TABLE%%' || %s || '%%')
                )
                AND qh.QUERY_TYPE IN ('CREATE_TABLE', 'ALTER', 'DROP')
                AND qh.EXECUTION_STATUS = 'SUCCESS'
                ORDER BY qh.START_TIME DESC
                LIMIT 20
                """
                
                cur.execute(fallback_ddl_query, (database, schema, table, table, table, table))
                ddl_results = cur.fetchall()
                ddl_columns = [desc[0] for desc in cur.description]
                
            except Exception as fallback_error:
                logger.warning(f"Fallback DDL query also failed: {fallback_error}")
                ddl_results = []
                ddl_columns = []
        
        cur.close()
        conn.close()
        
        # Combine all results into a comprehensive lineage DataFrame
        all_results = []
        
        # Add dependency relationships
        if dependency_results and dependency_columns:
            for row in dependency_results:
                all_results.append(dict(zip(dependency_columns, row)))
        
        # Add access history with transformations
        if access_results and access_columns:
            for row in access_results:
                row_dict = dict(zip(access_columns, row))
                # Parse the JSON arrays to extract table names (if available)
                if row_dict.get('DIRECT_OBJECTS_ACCESSED'):
                    row_dict['PARSED_DIRECT_OBJECTS'] = str(row_dict['DIRECT_OBJECTS_ACCESSED'])
                if row_dict.get('BASE_OBJECTS_ACCESSED'):
                    row_dict['PARSED_BASE_OBJECTS'] = str(row_dict['BASE_OBJECTS_ACCESSED'])
                if row_dict.get('OBJECTS_MODIFIED'):
                    row_dict['PARSED_MODIFIED_OBJECTS'] = str(row_dict['OBJECTS_MODIFIED'])
                all_results.append(row_dict)
        
        # Add DDL history
        if ddl_results and ddl_columns:
            for row in ddl_results:
                all_results.append(dict(zip(ddl_columns, row)))
        
        if not all_results:
            logger.info(f"No lineage data found for {database}.{schema}.{table}")
            # Return an empty DataFrame with expected columns for consistency
            return pd.DataFrame(columns=['SOURCE_VIEW', 'QUERY_TYPE', 'QUERY_START_TIME', 'USER_NAME', 'QUERY_TEXT'])
        
        result_df = pd.DataFrame(all_results)
        logger.info(f"Found {len(result_df)} lineage records for {database}.{schema}.{table}")
        return result_df
        
    except Exception as e:
        logger.error(f"Snowflake lineage query error: {e}")
        raise


def display_lineage_results():
    """Display lineage data with visualization"""
    lineage_data = st.session_state.lineage_data
    
    if lineage_data is None or lineage_data.empty:
        st.warning("No lineage data to display.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Lineage Graph", "Query Analysis", "Timeline"])
    
    with tab1:
        display_lineage_graph(lineage_data)
    
    with tab2:
        display_lineage_table(lineage_data)
    
    with tab3:
        display_lineage_timeline(lineage_data)
    
    # AI Recommendations section after lineage analysis
    st.markdown("---")
    display_ai_recommendations_section("lineage")


def display_lineage_graph(lineage_data: pd.DataFrame):
    """Create and display an interactive lineage graph based on actual SQL transformations"""
    st.subheader("Data Transformation Lineage Graph")
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Get the current table as the center node
    current_table = st.session_state.get('current_table', '')
    current_database = st.session_state.get('current_database', '')
    current_schema = st.session_state.get('current_schema', '')
    center_node = f"{current_database}.{current_schema}.{current_table}"
    
    G.add_node(center_node, node_type='target', color='red', size=80)
    
    # Process transformation queries to extract actual source tables
    query_data = lineage_data[lineage_data['SOURCE_VIEW'].str.contains('QUERY_HISTORY')]
    
    transformation_count = 0
    source_tables = set()
    
    for _, row in query_data.iterrows():
        query_type = row.get('QUERY_TYPE', '')
        query_text = row.get('QUERY_TEXT', '').upper()
        timestamp = row.get('QUERY_START_TIME', '')
        user = row.get('USER_NAME', 'Unknown')
        
        # Parse SQL to find actual source tables
        if query_type in ['CREATE_TABLE_AS_SELECT', 'INSERT', 'MERGE']:
            # Look for FROM clauses to find source tables
            if 'FROM' in query_text:
                # Extract table references after FROM
                from_parts = query_text.split('FROM')[1:]
                for part in from_parts:
                    # Split on common SQL keywords to isolate table names
                    table_part = part.split('WHERE')[0].split('GROUP')[0].split('ORDER')[0].split('JOIN')[0]
                    
                    # Look for patterns like DATABASE.SCHEMA.TABLE or just TABLE
                    words = table_part.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                    for word in words[:3]:  # Only check first few words after FROM
                        word = word.strip()
                        if word and not word in ['SELECT', 'DISTINCT', 'TOP', 'ALL']:
                            # Clean up the word
                            clean_word = word.replace('"', '').replace("'", '').replace(';', '')
                            if '.' in clean_word:
                                # Full table reference
                                parts = clean_word.split('.')
                                if len(parts) >= 2:
                                    source_table = clean_word
                                    if source_table != center_node and len(source_table) > 3:
                                        source_tables.add(source_table)
                                        break
                            elif len(clean_word) > 2 and not clean_word.isdigit():
                                # Just table name, assume same schema/database
                                source_table = f"{current_database}.{current_schema}.{clean_word}"
                                if source_table != center_node:
                                    source_tables.add(source_table)
                                    break
            
            # Add transformation node
            transformation_count += 1
            transform_node = f"{query_type}_{transformation_count}"
            G.add_node(transform_node, 
                      node_type='transformation', 
                      color='yellow', 
                      size=60,
                      query_type=query_type,
                      timestamp=str(timestamp),
                      user=user)
            
            # Connect transformation to target
            G.add_edge(transform_node, center_node, 
                      relationship=query_type.lower(),
                      timestamp=str(timestamp))
        
        elif query_type == 'UPDATE':
            # For updates, the table is both source and target
            update_node = f"UPDATE_{transformation_count + 1}"
            transformation_count += 1
            G.add_node(update_node, 
                      node_type='update', 
                      color='orange', 
                      size=60,
                      query_type=query_type,
                      timestamp=str(timestamp),
                      user=user)
            
            # Self-referencing edge for updates
            G.add_edge(center_node, update_node, relationship='self_update')
            G.add_edge(update_node, center_node, relationship='updated')
    
    # Add source tables as nodes
    for source_table in source_tables:
        G.add_node(source_table, node_type='source', color='lightblue', size=70)
        
        # Connect sources to transformations if we can match them
        for node in G.nodes():
            if G.nodes[node].get('node_type') == 'transformation':
                # Connect source to transformation
                G.add_edge(source_table, node, relationship='feeds_into')
    
    # Process dependencies if available
    dependencies = lineage_data[lineage_data['SOURCE_VIEW'] == 'OBJECT_DEPENDENCIES']
    for _, row in dependencies.iterrows():
        source = f"{row['SOURCE_DATABASE']}.{row['SOURCE_SCHEMA']}.{row['SOURCE_TABLE']}"
        target = f"{row['TARGET_DATABASE']}.{row['TARGET_SCHEMA']}.{row['TARGET_TABLE']}"
        
        if source not in G.nodes():
            G.add_node(source, node_type='dependency_source', color='lightgreen', size=60)
        if target not in G.nodes():
            G.add_node(target, node_type='dependency_target', color='lightcoral', size=60)
        
        G.add_edge(source, target, relationship='dependency')
    
    if len(G.nodes()) <= 1:
        st.info(f"No transformation lineage found for {current_table}. This table may not have been created or modified through tracked SQL operations.")
        st.write("**Possible reasons:**")
        st.write("- Table was created outside of tracked time period")
        st.write("- Table was loaded via bulk operations")
        st.write("- Insufficient query history permissions")
        return
    
    # Create layout with the target table in center
    try:
        pos = nx.spring_layout(G, k=2, iterations=100)
        # Force center node to be in the center
        if center_node in pos:
            pos[center_node] = (0, 0)
            # Recalculate layout with fixed center
            fixed_nodes = [center_node]
            pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes, k=2, iterations=50)
    except:
        pos = nx.random_layout(G)
    
    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    edge_info = []
    edge_colors = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        relationship = edge[2].get('relationship', 'unknown')
        timestamp = edge[2].get('timestamp', '')
        
        # Color edges by relationship type
        if 'transformation' in relationship or 'create' in relationship:
            edge_colors.extend(['#ff7f0e', '#ff7f0e', None])  # Orange for transformations
        elif 'dependency' in relationship:
            edge_colors.extend(['#2ca02c', '#2ca02c', None])  # Green for dependencies
        elif 'update' in relationship:
            edge_colors.extend(['#d62728', '#d62728', None])  # Red for updates
        else:
            edge_colors.extend(['#1f77b4', '#1f77b4', None])  # Blue for others
        
        edge_info.append(f"{edge[0].split('.')[-1]} → {edge[1].split('.')[-1]} ({relationship}) {timestamp}")
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [node.split('.')[-1] if '.' in node else node for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', 'lightgray') for node in G.nodes()]
    node_sizes = [G.nodes[node].get('size', 50) for node in G.nodes()]
    
    # Create detailed hover text
    node_hover = []
    for node in G.nodes():
        node_data = G.nodes[node]
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Type: {node_data.get('node_type', 'unknown')}<br>"
        if node_data.get('query_type'):
            hover_text += f"Operation: {node_data.get('query_type')}<br>"
        if node_data.get('timestamp'):
            hover_text += f"Time: {node_data.get('timestamp')}<br>"
        if node_data.get('user'):
            hover_text += f"User: {node_data.get('user')}"
        node_hover.append(hover_text)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges with different colors
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.8)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_hover,
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='darkblue')
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Data Transformation Lineage for {current_table}",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40,l=20,r=20,t=60),
        annotations=[
            dict(
                text="Red: Target table | Blue: Source tables | Yellow: Transformations | Green: Dependencies",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=20)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True, key="lineage_graph")
    
    # Show transformation summary
    if transformation_count > 0:
        st.subheader("Transformation Summary")
        st.write(f"**Found {transformation_count} transformation operations**")
        st.write(f"**Source tables identified: {len(source_tables)}**")
        
        if source_tables:
            st.write("**Source tables:**")
            for source in sorted(source_tables):
                st.write(f"- {source}")
    else:
        st.info("No transformation operations found in query history.")


def display_lineage_table(lineage_data: pd.DataFrame):
    """Display lineage data in organized sections"""
    st.subheader("Lineage Analysis")
    
    # Separate different types of lineage data
    dependencies = lineage_data[lineage_data['SOURCE_VIEW'] == 'OBJECT_DEPENDENCIES']
    access_history = lineage_data[lineage_data['SOURCE_VIEW'].str.contains('QUERY_HISTORY', na=False)]
    ddl_history = lineage_data[lineage_data['SOURCE_VIEW'].str.contains('DDL_HISTORY', na=False)]
    
    # Create tabs for different views
    dep_tab, access_tab, ddl_tab = st.tabs(["Dependencies", "Access History", "DDL History"])
    
    with dep_tab:
        st.subheader("Object Dependencies")
        if not dependencies.empty:
            # Display dependency relationships
            dep_display = dependencies[[
                'SOURCE_DATABASE', 'SOURCE_SCHEMA', 'SOURCE_TABLE',
                'TARGET_DATABASE', 'TARGET_SCHEMA', 'TARGET_TABLE',
                'RELATIONSHIP_TYPE', 'DEPENDENCY_CREATED'
            ]].copy()
            
            dep_display.columns = [
                'Source DB', 'Source Schema', 'Source Table',
                'Target DB', 'Target Schema', 'Target Table',
                'Relationship', 'Created On'
            ]
            
            st.dataframe(dep_display, use_container_width=True, hide_index=True)
            
            # Summary
            st.write(f"**Total Dependencies:** {len(dependencies)}")
            unique_sources = len(dependencies[['SOURCE_DATABASE', 'SOURCE_SCHEMA', 'SOURCE_TABLE']].drop_duplicates())
            unique_targets = len(dependencies[['TARGET_DATABASE', 'TARGET_SCHEMA', 'TARGET_TABLE']].drop_duplicates())
            st.write(f"**Unique Source Objects:** {unique_sources}")
            st.write(f"**Unique Target Objects:** {unique_targets}")
        else:
            st.info("No object dependencies found.")
    
    with access_tab:
        st.subheader("Query History & Transformations")
        if not access_history.empty:
            # Check if QUERY_TYPE column exists before filtering
            if 'QUERY_TYPE' in access_history.columns:
                # Show transformation queries
                transformation_queries = access_history[
                    access_history['QUERY_TYPE'].isin(['INSERT', 'UPDATE', 'MERGE', 'CREATE_TABLE_AS_SELECT', 'CREATE_TABLE'])
                ]
                
                if not transformation_queries.empty:
                    st.write("**🔄 Transformation Operations:**")
                    for idx, row in transformation_queries.iterrows():
                        query_type = row.get('QUERY_TYPE', 'Unknown')
                        timestamp = row.get('QUERY_START_TIME', 'Unknown time')
                        user = row.get('USER_NAME', 'Unknown user')
                        execution_time = row.get('TOTAL_ELAPSED_TIME', 'Unknown')
                        warehouse = row.get('WAREHOUSE_NAME', 'Unknown')
                        
                        # Color code by query type
                        if query_type in ['INSERT', 'CREATE_TABLE_AS_SELECT']:
                            badge_color = "✅"  # Creation/Insert
                        elif query_type in ['UPDATE', 'MERGE']:
                            badge_color = "🔄"  # Modification
                        elif query_type == 'CREATE_TABLE':
                            badge_color = "🆕"  # New table
                        else:
                            badge_color = "⚡"
                        
                        with st.expander(f"{badge_color} {query_type} - {timestamp} by {user}"):
                            # Show query details
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Execution Time:** {execution_time} ms")
                            with col2:
                                st.write(f"**Warehouse:** {warehouse}")
                            with col3:
                                st.write(f"**Status:** {row.get('EXECUTION_STATUS', 'Unknown')}")
                            
                            # Show SQL query
                            query_text = row.get('QUERY_TEXT', 'No query text available')
                            if len(query_text) > 1000:
                                st.code(query_text[:1000] + "...", language='sql')
                                with st.expander("View full query"):
                                    st.code(query_text, language='sql')
                            else:
                                st.code(query_text, language='sql')
                
                # Show read-only queries separately
                read_queries = access_history[access_history['QUERY_TYPE'] == 'SELECT']
                
                if not read_queries.empty:
                    st.write("**📊 Read Operations:**")
                    for idx, row in read_queries.head(10).iterrows():  # Limit to first 10 SELECT queries
                        timestamp = row.get('QUERY_START_TIME', 'Unknown time')
                        user = row.get('USER_NAME', 'Unknown user')
                        execution_time = row.get('TOTAL_ELAPSED_TIME', 'Unknown')
                        
                        with st.expander(f"📊 SELECT - {timestamp} by {user}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Execution Time:** {execution_time} ms")
                            with col2:
                                st.write(f"**Warehouse:** {row.get('WAREHOUSE_NAME', 'Unknown')}")
                            
                            query_text = row.get('QUERY_TEXT', 'No query text available')
                            if len(query_text) > 500:
                                st.code(query_text[:500] + "...", language='sql')
                            else:
                                st.code(query_text, language='sql')
                
                # Summary statistics
                st.subheader("Query Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", len(access_history))
                
                with col2:
                    transformations = len(transformation_queries)
                    st.metric("Transformations", transformations)
                
                with col3:
                    reads = len(read_queries)
                    st.metric("Read Operations", reads)
                
                with col4:
                    if 'USER_NAME' in access_history.columns:
                        unique_users = access_history['USER_NAME'].nunique()
                        st.metric("Unique Users", unique_users)
                    else:
                        st.metric("Unique Users", "N/A")
                        
                # Timeline of activity
                if 'QUERY_START_TIME' in access_history.columns:
                    st.subheader("Activity Timeline")
                    timeline_data = access_history.copy()
                    timeline_data['QUERY_START_TIME'] = pd.to_datetime(timeline_data['QUERY_START_TIME'])
                    timeline_data['Date'] = timeline_data['QUERY_START_TIME'].dt.date
                    
                    daily_activity = timeline_data.groupby(['Date', 'QUERY_TYPE']).size().reset_index(name='Count')
                    
                    if len(daily_activity) > 1:
                        fig = px.bar(
                            daily_activity,
                            x='Date',
                            y='Count',
                            color='QUERY_TYPE',
                            title='Query Activity Over Time',
                            color_discrete_map={
                                'SELECT': '#1f77b4',
                                'INSERT': '#ff7f0e',
                                'UPDATE': '#2ca02c',
                                'CREATE_TABLE_AS_SELECT': '#d62728',
                                'MERGE': '#9467bd'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True, key="query_history_chart_1")
            
            else:
                # Fallback: show all queries without type filtering
                st.warning("Query type information not available in the data.")
                st.write("**Debug - Available columns:**", list(access_history.columns))
                st.write("**All Query History:**")
                for idx, row in access_history.head(20).iterrows():
                    timestamp = row.get('QUERY_START_TIME', 'Unknown time')
                    user = row.get('USER_NAME', 'Unknown user')
                    
                    with st.expander(f"Query {idx + 1} - {timestamp} by {user}"):
                        query_text = row.get('QUERY_TEXT', 'No query text available')
                        st.code(query_text, language='sql')
                        
                        # Show other available metadata
                        for col in access_history.columns:
                            if col not in ['QUERY_TEXT'] and pd.notna(row.get(col)):
                                st.write(f"**{col}:** {row.get(col)}")
                
                # Basic summary without query type filtering
                st.subheader("Basic Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", len(access_history))
                with col2:
                    if 'USER_NAME' in access_history.columns:
                        unique_users = access_history['USER_NAME'].nunique()
                        st.metric("Unique Users", unique_users)
                    else:
                        st.metric("Unique Users", "N/A")
                
        else:
            st.info("No query history found for this table.")
    
    with ddl_tab:
                for idx, row in transformation_queries.iterrows():
                    query_type = row.get('QUERY_TYPE', 'Unknown')
                    timestamp = row.get('QUERY_START_TIME', 'Unknown time')
                    user = row.get('USER_NAME', 'Unknown user')
                    execution_time = row.get('TOTAL_ELAPSED_TIME', 'Unknown')
                    warehouse = row.get('WAREHOUSE_NAME', 'Unknown')
                    
                    # Color code by query type
                    if query_type in ['INSERT', 'CREATE_TABLE_AS_SELECT']:
                        badge_color = "✅"  # Creation/Insert
                    elif query_type in ['UPDATE', 'MERGE']:
                        badge_color = "�"  # Modification
                    elif query_type == 'CREATE_TABLE':
                        badge_color = "🆕"  # New table
                    else:
                        badge_color = "⚡"
                    
                    with st.expander(f"{badge_color} {query_type} - {timestamp} by {user}"):
                        # Show query details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Execution Time:** {execution_time} ms")
                        with col2:
                            st.write(f"**Warehouse:** {warehouse}")
                        with col3:
                            st.write(f"**Status:** {row.get('EXECUTION_STATUS', 'Unknown')}")
                        
                        # Show SQL query
                        query_text = row.get('QUERY_TEXT', 'No query text available')
                        if len(query_text) > 1000:
                            st.code(query_text[:1000] + "...", language='sql')
                            with st.expander("View full query"):
                                st.code(query_text, language='sql')
                        else:
                            st.code(query_text, language='sql')
            
            # Show read-only queries separately
                read_queries = access_history[access_history['QUERY_TYPE'] == 'SELECT']
            
                if not read_queries.empty:
                    st.write("**📊 Read Operations:**")
                    for idx, row in read_queries.head(10).iterrows():  # Limit to first 10 SELECT queries
                        timestamp = row.get('QUERY_START_TIME', 'Unknown time')
                        user = row.get('USER_NAME', 'Unknown user')
                        execution_time = row.get('TOTAL_ELAPSED_TIME', 'Unknown')
                        
                        with st.expander(f"📊 SELECT - {timestamp} by {user}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Execution Time:** {execution_time} ms")
                            with col2:
                                st.write(f"**Warehouse:** {row.get('WAREHOUSE_NAME', 'Unknown')}")
                            
                            query_text = row.get('QUERY_TEXT', 'No query text available')
                            if len(query_text) > 500:
                                st.code(query_text[:500] + "...", language='sql')
                            else:
                                st.code(query_text, language='sql')
                
                # Summary statistics
                st.subheader("Query Summary")
                col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                    st.metric("Total Queries", len(access_history))
                
                with col2:
                    transformations = len(transformation_queries)
                    st.metric("Transformations", transformations)
                
                with col3:
                    reads = len(read_queries)
                    st.metric("Read Operations", reads)
                
                with col4:
                    unique_users = access_history['USER_NAME'].nunique()
                    st.metric("Unique Users", unique_users)
                    
                # Timeline of activity
                if 'QUERY_START_TIME' in access_history.columns:
                    st.subheader("Activity Timeline")
                    timeline_data = access_history.copy()
                    timeline_data['QUERY_START_TIME'] = pd.to_datetime(timeline_data['QUERY_START_TIME'])
                    timeline_data['Date'] = timeline_data['QUERY_START_TIME'].dt.date
                    
                    daily_activity = timeline_data.groupby(['Date', 'QUERY_TYPE']).size().reset_index(name='Count')
                    
                    if len(daily_activity) > 1:
                        fig = px.bar(
                            daily_activity,
                            x='Date',
                            y='Count',
                            color='QUERY_TYPE',
                            title='Query Activity Over Time',
                            color_discrete_map={
                                'SELECT': '#1f77b4',
                                'INSERT': '#ff7f0e',
                                'UPDATE': '#2ca02c',
                                'CREATE_TABLE_AS_SELECT': '#d62728',
                                'MERGE': '#9467bd'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True, key="query_history_chart_2")
                    
                else:
                    st.info("No query history found for this table.")
    
    with ddl_tab:
        st.subheader("DDL History (Table Structure Changes)")
        if not ddl_history.empty:
            for idx, row in ddl_history.iterrows():
                action = row.get('DDL_ACTION', 'Unknown')
                timestamp = row.get('EXECUTION_TIME', 'Unknown time')
                user = row.get('EXECUTED_BY', 'Unknown user')
                object_type = row.get('OBJECT_TYPE', 'Unknown')
                
                # Color code by action
                if action == 'CREATE':
                    badge_color = "✅"
                elif action == 'ALTER':
                    badge_color = "🔧"
                elif action == 'DROP':
                    badge_color = "❌"
                else:
                    badge_color = "⚡"
                
                with st.expander(f"{badge_color} {action} {object_type} - {timestamp} by {user}"):
                    st.code(row.get('QUERY_TEXT', 'No DDL text available'), language='sql')
            
            # Summary
            st.subheader("DDL Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total DDL Operations", len(ddl_history))
            with col2:
                creates = len(ddl_history[ddl_history['DDL_ACTION'] == 'CREATE'])
                st.metric("Creates", creates)
            with col3:
                alters = len(ddl_history[ddl_history['DDL_ACTION'] == 'ALTER'])
                st.metric("Alters", alters)
        else:
            st.info("No DDL history found.")


def display_lineage_timeline(lineage_data: pd.DataFrame):
    """Display transformation timeline showing how the table changed over time"""
    st.subheader("Transformation Timeline")
    
    # Combine all timestamped events
    timeline_events = []
    
    # Add dependency creation events
    dependencies = lineage_data[lineage_data['SOURCE_VIEW'] == 'OBJECT_DEPENDENCIES']
    for _, row in dependencies.iterrows():
        if pd.notna(row.get('DEPENDENCY_CREATED')):
            timeline_events.append({
                'timestamp': row['DEPENDENCY_CREATED'],
                'event_type': 'Dependency Created',
                'description': f"Dependency: {row['SOURCE_TABLE']} → {row['TARGET_TABLE']}",
                'category': 'dependency'
            })
    
    # Add access/transformation events
    access_history = lineage_data[lineage_data['SOURCE_VIEW'].str.contains('QUERY_HISTORY', na=False)]
    for _, row in access_history.iterrows():
        if pd.notna(row.get('QUERY_START_TIME')):
            query_type = row.get('QUERY_TYPE', 'Unknown')
            user = row.get('USER_NAME', 'Unknown')
            
            # Determine if this was a transformation
            is_transformation = query_type in ['INSERT', 'UPDATE', 'MERGE', 'CREATE_TABLE_AS_SELECT', 'DELETE']
            
            timeline_events.append({
                'timestamp': row['QUERY_START_TIME'],
                'event_type': f"{query_type} Query",
                'description': f"{query_type} by {user}",
                'category': 'transformation' if is_transformation else 'access',
                'execution_time': row.get('TOTAL_ELAPSED_TIME', 0)
            })
    
    # Add DDL events
    ddl_history = lineage_data[lineage_data['SOURCE_VIEW'].str.contains('DDL_HISTORY', na=False)]
    for _, row in ddl_history.iterrows():
        if pd.notna(row.get('EXECUTION_TIME')):
            timeline_events.append({
                'timestamp': row['EXECUTION_TIME'],
                'event_type': f"DDL {row.get('DDL_ACTION', 'Unknown')}",
                'description': f"{row.get('DDL_ACTION', 'Unknown')} {row.get('OBJECT_TYPE', 'Object')} by {row.get('EXECUTED_BY', 'Unknown')}",
                'category': 'ddl'
            })
    
    if not timeline_events:
        st.warning("No timestamped events found for timeline visualization.")
        return
    
    # Convert to DataFrame and sort
    timeline_df = pd.DataFrame(timeline_events)
    timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
    timeline_df = timeline_df.sort_values('timestamp')
    
    # Create timeline visualization
    if len(timeline_df) > 1:
        # Group by date for overview chart
        timeline_df['date'] = timeline_df['timestamp'].dt.date
        daily_counts = timeline_df.groupby(['date', 'category']).size().reset_index(name='count')
        
        # Create stacked bar chart
        fig = px.bar(
            daily_counts,
            x='date',
            y='count',
            color='category',
            title='Table Activity Over Time',
            color_discrete_map={
                'transformation': '#ff7f0e',  # Orange for transformations
                'access': '#1f77b4',         # Blue for access
                'dependency': '#2ca02c',     # Green for dependencies
                'ddl': '#d62728'             # Red for DDL
            }
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Events",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, key="timeline_chart")
    
    # Detailed timeline
    st.subheader("Detailed Event Timeline")
    
    # Show events in reverse chronological order
    for idx, event in timeline_df.head(20).iterrows():
        timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        event_type = event['event_type']
        description = event['description']
        category = event['category']
        
        # Choose icon based on category
        if category == 'transformation':
            icon = "🔄"
        elif category == 'access':
            icon = "📊"
        elif category == 'dependency':
            icon = "🔗"
        elif category == 'ddl':
            icon = "🛠️"
        else:
            icon = "⚡"
        
        # Show event with context
        with st.container():
            col1, col2 = st.columns([1, 6])
            with col1:
                st.write(f"**{timestamp}**")
            with col2:
                st.write(f"{icon} **{event_type}**: {description}")
                if event.get('execution_time'):
                    st.write(f"   ⏱️ Execution time: {event['execution_time']} ms")
    
    # Summary statistics
    st.subheader("Timeline Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        transformations = len(timeline_df[timeline_df['category'] == 'transformation'])
        st.metric("Transformations", transformations)
    
    with col2:
        access_events = len(timeline_df[timeline_df['category'] == 'access'])
        st.metric("Access Events", access_events)
    
    with col3:
        ddl_events = len(timeline_df[timeline_df['category'] == 'ddl'])
        st.metric("DDL Changes", ddl_events)
    
    with col4:
        date_range = timeline_df['timestamp'].max() - timeline_df['timestamp'].min()
        st.metric("Activity Period", f"{date_range.days} days")
