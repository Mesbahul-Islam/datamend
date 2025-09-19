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
    """Configuration for LLM API"""
    api_key: str
    api_url: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.3


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
        api_key = os.getenv('LLM_API_KEY', '')
        api_url = os.getenv('LLM_API_URL', 'https://api.openai.com/v1/chat/completions')
        model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        
        if not api_key:
            logger.warning("LLM_API_KEY not found in environment variables")
        
        return LLMConfig(
            api_key=api_key,
            api_url=api_url,
            model=model
        )
    
    def generate_recommendations(self, quality_report: Dict[str, Any], 
                               context: str = "credit risk") -> List[LLMRecommendation]:
        """
        Generate data quality recommendations based on profiling results.
        
        Args:
            quality_report: Data quality report from the engine
            context: Domain context for better recommendations
            
        Returns:
            List of LLM-generated recommendations
        """
        if not self.config.api_key:
            logger.warning("No API key configured, returning mock recommendations")
            return self._generate_mock_recommendations(quality_report)
        
        try:
            # Prepare the prompt for LLM
            prompt = self._build_analysis_prompt(quality_report, context)
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            # Parse response into recommendations
            recommendations = self._parse_llm_response(response)
            
            logger.info(f"Generated {len(recommendations)} recommendations from LLM")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating LLM recommendations: {str(e)}")
            # Fallback to rule-based recommendations
            return self._generate_fallback_recommendations(quality_report)
    
    def _build_analysis_prompt(self, quality_report: Dict[str, Any], context: str) -> str:
        """
        Build a comprehensive prompt for the LLM based on data quality findings.
        
        Args:
            quality_report: Data quality report
            context: Domain context
            
        Returns:
            Formatted prompt string
        """
        # Extract key information from the report
        total_rows = quality_report.get('total_rows', 0)
        total_columns = quality_report.get('total_columns', 0)
        overall_score = quality_report.get('overall_quality_score', 0)
        critical_issues = quality_report.get('critical_issues', [])
        
        # Summarize column issues
        column_issues = []
        column_profiles = quality_report.get('column_profiles', {})
        
        for col_name, profile in column_profiles.items():
            issues = profile.get('data_quality_issues', [])
            if issues:
                column_issues.append(f"- {col_name}: {', '.join(issues)}")
        
        prompt = f"""
You are a data quality expert analyzing a {context} dataset. Please provide specific, actionable recommendations to improve data quality.

Dataset Overview:
- Total Rows: {total_rows:,}
- Total Columns: {total_columns}
- Overall Quality Score: {overall_score:.1f}/100

Critical Issues Identified:
{chr(10).join(f"- {issue}" for issue in critical_issues) if critical_issues else "- None"}

Column-Specific Issues:
{chr(10).join(column_issues) if column_issues else "- No specific column issues detected"}

Based on this analysis, please provide data quality recommendations in the following JSON format:
{{
  "recommendations": [
    {{
      "category": "data_validation|data_cleaning|data_standardization|data_monitoring",
      "priority": "high|medium|low",
      "title": "Short descriptive title",
      "description": "Detailed description of the issue and why it matters for {context}",
      "action_items": ["Specific action 1", "Specific action 2", "..."],
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
        Make API call to LLM service.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response text
        """
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.config.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert data quality analyst with extensive experience in data governance and quality management.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature
        }
        
        response = requests.post(
            self.config.api_url,
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _parse_llm_response(self, response: str) -> List[LLMRecommendation]:
        """
        Parse LLM response into structured recommendations.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of parsed recommendations
        """
        try:
            # Extract JSON from response (sometimes LLM adds extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            recommendations = []
            for rec_data in data.get('recommendations', []):
                recommendation = LLMRecommendation(
                    category=rec_data.get('category', 'data_validation'),
                    priority=rec_data.get('priority', 'medium'),
                    title=rec_data.get('title', 'Data Quality Improvement'),
                    description=rec_data.get('description', ''),
                    action_items=rec_data.get('action_items', []),
                    affected_columns=rec_data.get('affected_columns', []),
                    estimated_impact=rec_data.get('estimated_impact', '')
                )
                recommendations.append(recommendation)
            
            return recommendations
            
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
