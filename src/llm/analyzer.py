"""
LLM Integration Module

This module provides AI-driven data quality recommendations using Large Language Models.
It analyzes data quality profiling results and generates actionable recommendations.
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMRecommendation:
    """Data class for LLM recommendations"""
    category: str
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    action_items: List[str]
    affected_columns: List[str]
    estimated_impact: str


@dataclass
class LLMConfig:
    def __init__(self, provider: str, model: str, api_key: str):
        # Normalize provider name
        self.provider = provider.lower().replace(" ", "").replace("-", "")
        self.model = model
        self.api_key = api_key
        self.api_url = self._get_api_url()

    def _get_api_url(self):
        if self.provider in ["gemini", "googlegemini", "google"]:
            return None  # Gemini uses SDK, not direct API calls
        elif self.provider in ["openai", "gpt"]:
            return "https://api.openai.com/v1/chat/completions"
        else:
            return "https://api.openai.com/v1/chat/completions"  # Default to OpenAI

class DataQualityLLMAnalyzer:
    """
    LLM-powered analyzer for generating data quality recommendations.
    """
    
    def __init__(self, config: LLMConfig = None):
        """
        Initialize the LLM analyzer.
        
        Args:
            config: LLM configuration. If None, will load from environment variables.
        """
        if config is None:
            self.config = self._load_config_from_env()
        else:
            self.config = config
        
        logger.info(f"LLM Analyzer initialized with model: {self.config.model}")
    
    def _load_config_from_env(self) -> LLMConfig:
        """Load LLM configuration from environment variables"""
        # Try multiple environment variable names for API key
        api_key = os.getenv('LLM_API_KEY', '')
        provider = os.getenv('LLM_PROVIDER', '').lower().replace(" ", "").replace("-", "")
        
        # If no LLM_API_KEY, try standard environment variables and auto-detect provider
        if not api_key:
            # Prioritize Gemini/Google AI keys first (since we default to Gemini)
            if os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY'):
                api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
                provider = 'gemini'
            elif os.getenv('OPENAI_API_KEY'):
                api_key = os.getenv('OPENAI_API_KEY')
                provider = 'openai'
        
        # If still no provider specified, default to gemini
        if not provider:
            provider = 'gemini'
        
        # Set default models based on provider
        if provider in ["gemini", "googlegemini", "google"]:
            default_model = 'gemini-2.0-flash'
        else:
            default_model = 'gpt-3.5-turbo'
            
        # Use environment model or default based on detected provider
        model = os.getenv('LLM_MODEL', default_model)
        
        # If LLM_MODEL is not set and we auto-detected provider, use the correct default
        if not os.getenv('LLM_MODEL'):
            model = default_model
        
        if not api_key:
            logger.warning("No API key found in environment variables (tried LLM_API_KEY, GOOGLE_AI_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY)")
        else:
            logger.info(f"API key loaded from environment for provider: {provider}")
        
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key
        )
    
    
    def _build_analysis_prompt(self, quality_report, context: str) -> str:
        """
        Build a comprehensive prompt for the LLM based on data quality findings.
        
        Args:
            quality_report: ydata-profiling ProfileReport object
            context: Domain context
            
        Returns:
            Formatted prompt string
        """
        # Check if quality_report is a ProfileReport object from ydata-profiling
        try:
            # Extract key information from the ProfileReport object
            if hasattr(quality_report, 'get_description'):
                # This is a ydata-profiling ProfileReport object
                description = quality_report.get_description()
                
                # Extract dataset overview
                table_info = description.table if hasattr(description, 'table') else {}
                total_rows = table_info.get('n', 0)
                total_columns = table_info.get('p', 0)
                
                # Extract variable information
                variables_info = description.variables if hasattr(description, 'variables') else {}
                
                # Build summary of data quality issues
                column_issues = []
                missing_data_columns = []
                high_cardinality_columns = []
                
                for col_name, col_info in variables_info.items():
                    # Check for missing data
                    missing_count = col_info.get('n_missing', 0)
                    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
                    
                    if missing_pct > 10:  # More than 10% missing
                        missing_data_columns.append(f"{col_name} ({missing_pct:.1f}% missing)")
                    
                    # Check for high cardinality
                    distinct_count = col_info.get('n_distinct', 0)
                    if distinct_count > total_rows * 0.9:  # High cardinality
                        high_cardinality_columns.append(f"{col_name} ({distinct_count} unique values)")
                    
                    # Check for data type issues
                    data_type = col_info.get('type', 'unknown')
                    if data_type == 'Unsupported':
                        column_issues.append(f"Unsupported data type in column: {col_name}")
                        
            elif hasattr(quality_report, 'description_set'):
                # Alternative access method
                description_set = quality_report.description_set
                
                # Extract dataset overview  
                dataset_info = description_set.get('dataset', {})
                total_rows = dataset_info.get('n', 0)
                total_columns = dataset_info.get('p', 0)
                
                # Extract variable information
                variables_info = description_set.get('variables', {})
                
                # Build summary of data quality issues
                column_issues = []
                missing_data_columns = []
                high_cardinality_columns = []
                
                for col_name, col_info in variables_info.items():
                    # Check for missing data
                    missing_count = col_info.get('n_missing', 0)
                    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
                    
                    if missing_pct > 10:  # More than 10% missing
                        missing_data_columns.append(f"{col_name} ({missing_pct:.1f}% missing)")
                    
                    # Check for high cardinality
                    distinct_count = col_info.get('n_distinct', 0)
                    if distinct_count > total_rows * 0.9:  # High cardinality
                        high_cardinality_columns.append(f"{col_name} ({distinct_count} unique values)")
                    
                    # Check for data type issues
                    data_type = col_info.get('type', 'unknown')
                    if data_type == 'Unsupported':
                        column_issues.append(f"Unsupported data type in column: {col_name}")
                
            else:
                # Fallback for dictionary format (legacy)
                total_rows = quality_report.get('total_rows', 0)
                total_columns = quality_report.get('total_columns', 0)
                column_issues = quality_report.get('critical_issues', [])
                missing_data_columns = []
                high_cardinality_columns = []
        
        except Exception as e:
            # Fallback if we can't extract from ProfileReport
            logger.warning(f"Could not extract from ProfileReport: {str(e)}")
            total_rows = 0
            total_columns = 0
            column_issues = [f"Error extracting data quality information: {str(e)}"]
            missing_data_columns = []
            high_cardinality_columns = []
        
        # Build the prompt with extracted information
        prompt = f"""You are a data quality expert analyzing a {context} dataset. Please provide specific, actionable recommendations to improve data quality.

Dataset Overview:
- Total Rows: {total_rows:,}
- Total Columns: {total_columns}

Detailed Column Analysis:"""
        
        # Add detailed column information
        if hasattr(quality_report, 'get_description'):
            try:
                description = quality_report.get_description()
                variables_info = description.variables if hasattr(description, 'variables') else {}
                
                prompt += "\n"
                for col_name, col_info in variables_info.items():
                    data_type = col_info.get('type', 'unknown')
                    missing_count = col_info.get('n_missing', 0)
                    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
                    distinct_count = col_info.get('n_distinct', 0)
                    distinct_pct = (distinct_count / total_rows * 100) if total_rows > 0 else 0
                    
                    prompt += f"\nColumn: {col_name}\n"
                    prompt += f"  - Data Type: {data_type}\n"
                    prompt += f"  - Missing Values: {missing_count} ({missing_pct:.1f}%)\n"
                    prompt += f"  - Unique Values: {distinct_count} ({distinct_pct:.1f}%)\n"
                    
                    # Add data type specific statistics
                    if data_type in ['Numeric', 'Integer', 'Float']:
                        if 'min' in col_info:
                            prompt += f"  - Range: {col_info.get('min', 'N/A')} to {col_info.get('max', 'N/A')}\n"
                        if 'mean' in col_info:
                            prompt += f"  - Mean: {col_info.get('mean', 'N/A'):.2f}\n"
                        if 'std' in col_info:
                            prompt += f"  - Std Dev: {col_info.get('std', 'N/A'):.2f}\n"
                    elif data_type in ['Categorical', 'Text', 'String']:
                        if 'mode' in col_info:
                            prompt += f"  - Most Frequent: {col_info.get('mode', 'N/A')}\n"
                        if 'word_counts_mean' in col_info:
                            prompt += f"  - Avg Length: {col_info.get('word_counts_mean', 'N/A')}\n"
                    elif data_type == 'DateTime':
                        if 'min' in col_info:
                            prompt += f"  - Date Range: {col_info.get('min', 'N/A')} to {col_info.get('max', 'N/A')}\n"
                    
                    # Add warnings for data quality issues
                    if missing_pct > 10:
                        prompt += f"  ⚠️ HIGH MISSING RATE ({missing_pct:.1f}%)\n"
                    if distinct_pct > 90:
                        prompt += f"  ⚠️ HIGH CARDINALITY ({distinct_pct:.1f}% unique)\n"
                    if data_type == 'Unsupported':
                        prompt += f"  ❌ UNSUPPORTED DATA TYPE\n"
                        
            except Exception as e:
                prompt += f"\n⚠️ Could not extract detailed column information: {str(e)}\n"
        
        prompt += "\nSummary of Key Issues:"
        
        # Add missing data issues
        if missing_data_columns:
            prompt += "\n\nMissing Data Issues:\n"
            for issue in missing_data_columns:
                prompt += f"- {issue}\n"
        
        # Add high cardinality issues
        if high_cardinality_columns:
            prompt += "\nHigh Cardinality Columns:\n"
            for issue in high_cardinality_columns:
                prompt += f"- {issue}\n"
        
        # Add other column issues
        if column_issues:
            prompt += "\nOther Column Issues:\n"
            for issue in column_issues:
                prompt += f"- {issue}\n"
        
        if not missing_data_columns and not high_cardinality_columns and not column_issues:
            prompt += "\n- No major data quality issues detected\n"
        
        prompt += f"""
                    Based on this analysis, please provide data quality recommendations in the following JSON format:
                    {{
                    "summary": "Brief overall assessment of data quality",
                    "recommendations": [
                        {{
                        "type": "data_validation|data_cleaning|data_standardization|data_monitoring",
                        "priority": "high|medium|low",
                        "title": "Short descriptive title",
                        "description": "Detailed description of the issue and why it matters for {context}",
                        "suggested_actions": ["Specific action 1", "Specific action 2", "..."],
                        "affected_columns": ["column1", "column2", "..."],
                        "estimated_impact": "Expected improvement description"
                        }}
                    ]
                    }}

                    Focus on:
                    1. Issues that could impact {context} analysis accuracy
                    2. Data quality controls that prevent downstream problems
                    3. Automated monitoring and validation rules
                    4. Data standardization for consistency
                    5. Practical, implementable solutions

                    Provide 3-7 recommendations, prioritizing the most impactful ones.
                """
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the appropriate LLM API based on the configured provider.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Raw response from the LLM
        """
        # Normalize provider name
        provider = self.config.provider.lower().replace(" ", "").replace("-", "")
        
        if provider in ["gemini", "googlegemini", "google"]:
            return self._call_gemini_api(prompt)
        elif provider in ["openai", "gpt"]:
            return self._call_openai_api(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}. Supported providers: 'openai', 'gemini'")
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Google Gemini API using google-genai library"""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai library not available. Install with: pip install google-genai")
        
        try:
            # Create the client
            client = genai.Client(api_key=self.config.api_key)
            
            # Add system instruction for data quality analysis
            system_instruction = "You are an expert data quality analyst with deep knowledge of data profiling, data cleaning, and data validation best practices. Provide specific, actionable recommendations in JSON format."
            
            # Combine system instruction with user prompt
            full_prompt = f"{system_instruction}\n\n{prompt}"
            
            # Generate content using the models API
            response = client.models.generate_content(
                model=self.config.model,
                contents=[{
                    'role': 'user',
                    'parts': [{'text': full_prompt}]
                }],
                config={
                    'temperature': 0.1,  # Low temperature for consistent, factual responses
                    'max_output_tokens': 2048,
                    'candidate_count': 1
                }
            )
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
                else:
                    raise ValueError("No content in Gemini response")
            else:
                raise ValueError("No valid candidates in Gemini response")
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API using requests"""
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.config.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert data quality analyst with deep knowledge of data profiling, data cleaning, and data validation best practices. Provide specific, actionable recommendations in JSON format.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.1,  # Low temperature for consistent responses
            'max_tokens': 2048
        }

        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=data,
                timeout=60
            )

            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' in response_data and response_data['choices']:
                return response_data['choices'][0]['message']['content']
            else:
                raise ValueError("No valid response from OpenAI API")
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured recommendations.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary with recommendations in the expected UI format
        """
        try:
            # Extract JSON from response (sometimes LLM adds extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Convert to the format expected by the UI
            result = {
                'summary': data.get('summary', 'AI-generated data quality recommendations'),
                'recommendations': []
            }
            
            for rec_data in data.get('recommendations', []):
                # Map field names to match UI expectations
                recommendation = {
                    'type': rec_data.get('type', rec_data.get('category', 'data_validation')),
                    'priority': rec_data.get('priority', 'medium'),
                    'title': rec_data.get('title', 'Data Quality Improvement'),
                    'description': rec_data.get('description', ''),
                    'suggested_actions': rec_data.get('suggested_actions', rec_data.get('action_items', [])),
                    'affected_columns': rec_data.get('affected_columns', []),
                    'estimated_impact': rec_data.get('estimated_impact', '')
                }
                result['recommendations'].append(recommendation)
            
            logger.info(f"Successfully parsed {len(result['recommendations'])} recommendations")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Response content: {response[:500]}...")
            
            # Fallback: create a simple recommendation based on raw response
            return {
                'summary': 'Unable to parse structured recommendations, but analysis was performed.',
                'recommendations': [{
                    'type': 'general',
                    'priority': 'medium',
                    'title': 'AI Analysis Result',
                    'description': f'Raw AI response: {response[:200]}...',
                    'suggested_actions': ['Review the raw response for insights'],
                    'affected_columns': [],
                    'estimated_impact': 'Varies'
                }]
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Return empty list if parsing fails
            return []
    
    def _generate_mock_recommendations(self, quality_report: Dict[str, Any]) -> List[LLMRecommendation]:
        """
        Generate mock recommendations for testing when no API key is available.
        
        Args:
            quality_report: Data quality report
            
        Returns:
            List of mock recommendations
        """
        recommendations = []
        
        # Check for high null rates
        column_profiles = quality_report.get('column_profiles', {})
        high_null_columns = [
            name for name, profile in column_profiles.items()
            if profile.get('null_percentage', 0) > 20
        ]
        
        if high_null_columns:
            recommendations.append(LLMRecommendation(
                category='data_validation',
                priority='high',
                title='Address High Missing Value Rates',
                description='Several columns have significant missing values that could impact analysis accuracy.',
                action_items=[
                    'Investigate root causes of missing values',
                    'Implement data validation at source',
                    'Consider imputation strategies for critical fields'
                ],
                affected_columns=high_null_columns,
                estimated_impact='Improved data completeness and analysis reliability'
            ))
        
        # Check for outliers
        outlier_columns = [
            name for name, profile in column_profiles.items()
            if any('OUTLIER' in issue for issue in profile.get('data_quality_issues', []))
        ]
        
        if outlier_columns:
            recommendations.append(LLMRecommendation(
                category='data_cleaning',
                priority='medium',
                title='Review and Handle Outliers',
                description='Statistical outliers detected that may indicate data quality issues or legitimate extreme values.',
                action_items=[
                    'Review outlier values for legitimacy',
                    'Implement automated outlier detection',
                    'Create business rules for outlier handling'
                ],
                affected_columns=outlier_columns,
                estimated_impact='More accurate statistical analysis and modeling'
            ))
        
        # General recommendation for low overall score
        overall_score = quality_report.get('overall_quality_score', 100)
        if overall_score < 80:
            recommendations.append(LLMRecommendation(
                category='data_monitoring',
                priority='high',
                title='Implement Comprehensive Data Quality Monitoring',
                description='Overall data quality score indicates need for systematic monitoring and controls.',
                action_items=[
                    'Set up automated data quality checks',
                    'Create data quality dashboards',
                    'Establish data quality SLAs',
                    'Implement data validation pipelines'
                ],
                affected_columns=[],
                estimated_impact='Proactive data quality management and issue prevention'
            ))
        
        return recommendations
    
    def _generate_fallback_recommendations(self, quality_report: Dict[str, Any]) -> List[LLMRecommendation]:
        """
        Generate rule-based recommendations as fallback when LLM fails.
        
        Args:
            quality_report: Data quality report
            
        Returns:
            List of rule-based recommendations
        """
        logger.info("Generating fallback recommendations using rule-based approach")
        return self._generate_mock_recommendations(quality_report)
    
    def summarize_recommendations(self, recommendations: List[LLMRecommendation]) -> Dict[str, Any]:
        """
        Create a summary of recommendations by category and priority.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Summary dictionary
        """
        if not recommendations:
            return {}
        
        summary = {
            'total_recommendations': len(recommendations),
            'by_priority': {
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'by_category': {},
            'high_priority_actions': []
        }
        
        for rec in recommendations:
            # Count by priority
            summary['by_priority'][rec.priority] += 1
            
            # Count by category
            category = rec.category
            if category not in summary['by_category']:
                summary['by_category'][category] = 0
            summary['by_category'][category] += 1
            
            # Collect high priority actions
            if rec.priority == 'high':
                summary['high_priority_actions'].append(rec.title)
        
        return summary
