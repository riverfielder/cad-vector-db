"""HTML visualization for explainable retrieval results

Generates interactive HTML reports showing:
- Similarity breakdown (Stage 1 + Stage 2)
- Feature-level and sequence-level analysis
- Fusion method contributions
- Detailed metrics visualization
"""
import json
from typing import List, Dict, Optional
from pathlib import Path


# Backward compatibility alias
def generate_explanation_html(results: List[Dict],
                             query_path: str,
                             output_file: str = "explanation.html") -> str:
    """Alias for generate_html_visualization (backward compatibility)
    
    Args:
        results: List of search results with explanation field
        query_path: Path to query file
        output_file: Output HTML file path
        
    Returns:
        html_content: Generated HTML as string
    """
    return generate_html_visualization(results, query_path, output_file)


def generate_html_visualization(results: List[Dict],
                                query_path: str,
                                output_file: str = "explanation.html") -> str:
    """Generate HTML visualization for search results with explanations
    
    Args:
        results: List of search results with explanation field
        query_path: Path to query file
        output_file: Output HTML file path
        
    Returns:
        output_file: Path to generated HTML file
    """
    html_parts = []
    
    # HTML header with CSS
    html_parts.append(_get_html_header(query_path, len(results)))
    
    # Generate each result card
    for rank, result in enumerate(results, 1):
        html_parts.append(_generate_result_card(result, rank))
    
    # HTML footer
    html_parts.append("    </div>\n</body>\n</html>\n")
    
    # Write to file
    html = ''.join(html_parts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Visualization saved to: {output_file}")
    return output_file


def _get_html_header(query_path: str, num_results: int) -> str:
    """Generate HTML header with enhanced CSS styles"""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>可解释检索结果 | Explainable Retrieval Results</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px; margin: 0 auto; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }}
        h1 {{ 
            color: #333; 
            border-bottom: 4px solid #667eea; 
            padding-bottom: 15px;
            margin-top: 0;
            font-size: 32px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .query-info {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px; 
            margin: 20px 0; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }}
        .query-info h3 {{ margin-top: 0; }}
        .query-info code {{
            background: rgba(255,255,255,0.2);
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        .result-card {{ 
            background: white; 
            margin: 25px 0; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
            border-left: 6px solid #4CAF50;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        .result-header {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .rank-badge {{ 
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white; 
            padding: 8px 20px; 
            border-radius: 25px; 
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
        }}
        .score-badge {{ 
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white; 
            padding: 8px 20px; 
            border-radius: 25px; 
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
        }}
        .result-id {{ 
            font-size: 20px; 
            font-weight: bold; 
            color: #333; 
            flex-grow: 1; 
            margin: 0 20px;
            font-family: 'Courier New', monospace;
        }}
        .metadata {{ 
            background: #f9f9f9; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            font-size: 14px;
            border: 1px solid #e0e0e0;
        }}
        .explanation-section {{ 
            margin-top: 25px; 
            padding: 20px; 
            background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
            border-radius: 12px;
            border: 2px solid #81c784;
        }}
        .explanation-section h3 {{
            margin-top: 0;
            color: #2e7d32;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .similarity-bar {{ 
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .bar-label {{ 
            font-weight: bold; 
            margin-bottom: 8px; 
            display: flex; 
            justify-content: space-between;
            font-size: 14px;
            color: #555;
        }}
        .bar-container {{ 
            background: #e0e0e0;
            border-radius: 12px; 
            height: 36px; 
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        .bar-fill {{ 
            height: 100%; 
            display: flex; 
            align-items: center; 
            padding-left: 15px; 
            color: white; 
            font-weight: bold;
            font-size: 14px;
            transition: width 0.5s ease-out;
        }}
        .bar-stage1 {{ 
            background: linear-gradient(90deg, #FF6B6B, #FF8E53);
            box-shadow: 0 2px 8px rgba(255, 107, 107, 0.4);
        }}
        .bar-stage2 {{ 
            background: linear-gradient(90deg, #4ECDC4, #44A08D);
            box-shadow: 0 2px 8px rgba(78, 205, 196, 0.4);
        }}
        .bar-final {{ 
            background: linear-gradient(90deg, #667eea, #764ba2);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }}
        .quality-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .quality-excellent {{ background: #4CAF50; color: white; }}
        .quality-very_good {{ background: #8BC34A; color: white; }}
        .quality-good {{ background: #FFC107; color: white; }}
        .quality-moderate {{ background: #FF9800; color: white; }}
        .quality-weak {{ background: #F44336; color: white; }}
        .interpretation {{ 
            background: linear-gradient(135deg, #fff3cd 0%, #fffacd 100%);
            border-left: 5px solid #ffc107; 
            padding: 12px 18px; 
            margin: 12px 0;
            border-radius: 6px;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(255, 193, 7, 0.2);
        }}
        .analysis-section {{
            background: linear-gradient(135deg, #e3f2fd 0%, #e1f5fe 100%);
            border: 2px solid #64b5f6;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }}
        .analysis-section h4 {{
            margin-top: 0;
            color: #1976d2;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .confidence-meter {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 2px solid #64b5f6;
        }}
        .confidence-level {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }}
        .recommendations {{
            background: linear-gradient(135deg, #f3e5f5 0%, #fce4ec 100%);
            border: 2px solid #ba68c8;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations h4 {{
            margin-top: 0;
            color: #7b1fa2;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .recommendation-item {{
            background: white;
            padding: 12px 16px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #ba68c8;
            font-size: 14px;
            line-height: 1.6;
        }}
        .contributions {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin: 20px 0;
        }}
        .contribution-card {{ 
            background: white;
            padding: 20px; 
            border-radius: 10px; 
            border: 2px solid #e0e0e0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .contribution-card h4 {{
            margin-top: 0;
            color: #555;
        }}
        .stage2-details {{ 
            margin-top: 20px; 
            padding: 20px; 
            background: linear-gradient(135deg, #e1f5fe 0%, #e0f2f1 100%);
            border-radius: 12px;
            border: 2px solid #4dd0e1;
        }}
        .stage2-details h4 {{
            margin-top: 0;
            color: #00838f;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .details-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 15px; 
            margin: 15px 0;
        }}
        .detail-item {{ 
            background: white;
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #26c6da;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        .detail-label {{ 
            font-weight: bold; 
            color: #555; 
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .detail-value {{ 
            font-size: 18px; 
            color: #333; 
            margin-top: 8px;
            font-weight: bold;
        }}
        .fusion-info {{ 
            background: linear-gradient(135deg, #f3e5f5 0%, #ede7f6 100%);
            padding: 15px 20px; 
            border-radius: 8px; 
            margin: 15px 0; 
            border-left: 5px solid #9c27b0;
            font-weight: 500;
        }}
        @media (max-width: 768px) {{
            .contributions {{ grid-template-columns: 1fr; }}
            .result-header {{ flex-direction: column; align-items: flex-start; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 可解释检索结果 | Explainable Retrieval Results</h1>
        
        <div class="query-info">
            <h3>📋 查询信息 | Query Information</h3>
            <p><strong>查询文件 | Query File:</strong> <code>{query_path}</code></p>
            <p><strong>结果数量 | Total Results:</strong> {num_results}</p>
        </div>
"""


def _generate_result_card(result: Dict, rank: int) -> str:
    """Generate HTML for a single result card"""
    parts = []
    
    # Card header
    parts.append(f"""
    <div class="result-card">
        <div class="result-header">
            <span class="rank-badge">Rank #{rank}</span>
            <span class="result-id">{result['id']}</span>
            <span class="score-badge">Score: {result['score']:.4f}</span>
        </div>
        
        <div class="metadata">
            <strong>Metadata:</strong> 
            Subset: {result.get('metadata', {}).get('subset', 'N/A')} | 
            Sequence Length: {result.get('metadata', {}).get('seq_len', 'N/A')}
        </div>
""")
    
    # Add explanation if available
    if 'explanation' in result:
        parts.append(_generate_explanation_section(result))
    
    parts.append("    </div>\n")
    return ''.join(parts)


def _generate_explanation_section(result: Dict) -> str:
    """Generate enhanced explanation section HTML"""
    exp = result['explanation']
    stage1_sim = exp['stage1_similarity']
    stage2_sim = exp['stage2_similarity']
    final_score = exp['final_score']
    
    # Quality badges
    stage1_quality = exp.get('stage1_quality', 'unknown')
    stage2_quality = exp.get('stage2_quality', 'unknown')
    
    parts = [f"""
        <div class="explanation-section">
            <h3>📊 相似度分解 | Similarity Breakdown</h3>
            
            <div class="similarity-bar">
                <div class="bar-label">
                    <span>Stage 1 (特征级 | Feature-level)</span>
                    <span>{stage1_sim:.4f}<span class="quality-badge quality-{stage1_quality}">{stage1_quality.upper()}</span></span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill bar-stage1" style="width: {stage1_sim*100:.1f}%">
                        {stage1_sim*100:.1f}%
                    </div>
                </div>
                <div class="interpretation">{exp.get('stage1_interpretation', '')}</div>
            </div>
            
            <div class="similarity-bar">
                <div class="bar-label">
                    <span>Stage 2 (序列级 | Sequence-level)</span>
                    <span>{stage2_sim:.4f}<span class="quality-badge quality-{stage2_quality}">{stage2_quality.upper()}</span></span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill bar-stage2" style="width: {stage2_sim*100:.1f}%">
                        {stage2_sim*100:.1f}%
                    </div>
                </div>
                <div class="interpretation">{exp.get('stage2_interpretation', '')}</div>
            </div>
            
            <div class="similarity-bar">
                <div class="bar-label">
                    <span>Final (最终融合分数 | Fused Score)</span>
                    <span>{final_score:.4f}</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill bar-final" style="width: {final_score*100:.1f}%">
                        {final_score*100:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="fusion-info">
                <strong>融合方法 | Fusion Method:</strong> {exp['fusion_method']}
"""]
    
    # Add contributions if available
    if 'contributions' in exp:
        parts.append(_generate_contributions_html(exp['contributions']))
    
    parts.append("            </div>\n")
    
    # Match analysis
    if 'match_analysis' in exp:
        parts.append(_generate_match_analysis_html(exp['match_analysis']))
    
    # Confidence section
    if 'confidence' in exp:
        parts.append(_generate_confidence_html(exp['confidence']))
    
    # Recommendations
    if 'recommendations' in exp:
        parts.append(_generate_recommendations_html(exp['recommendations']))
    
    # Feature analysis
    if 'feature_analysis' in exp:
        parts.append(_generate_feature_analysis_html(exp['feature_analysis']))
    
    # Add stage2 details if available
    if 'stage2_details' in result:
        parts.append(_generate_stage2_details_html(result['stage2_details']))
    
    parts.append("        </div>\n")
    return ''.join(parts)


def _generate_match_analysis_html(analysis: Dict) -> str:
    """Generate match analysis HTML"""
    consistency_color = {'high': '#4CAF50', 'medium': '#FFC107', 'low': '#F44336'}.get(analysis['consistency'], '#999')
    
    return f"""
            <div class="analysis-section">
                <h4>🎯 匹配分析 | Match Analysis</h4>
                <p><strong>匹配类型 | Match Type:</strong> {analysis['match_type'].replace('_', ' ').title()}</p>
                <p><strong>描述 | Description:</strong> {analysis['description']}</p>
                <p><strong>一致性 | Consistency:</strong> 
                    <span style="color: {consistency_color}; font-weight: bold;">{analysis['consistency'].upper()}</span>
                    (差异 | Difference: {analysis['similarity_difference']:.4f})
                </p>
            </div>
"""


def _generate_confidence_html(confidence: Dict) -> str:
    """Generate confidence meter HTML"""
    score = confidence['score']
    level = confidence['level']
    
    # Color based on confidence level
    level_colors = {
        'very_high': '#4CAF50',
        'high': '#8BC34A',
        'medium': '#FFC107',
        'low': '#F44336'
    }
    color = level_colors.get(level, '#999')
    
    return f"""
            <div class="confidence-meter">
                <h4 style="margin-top: 0; color: {color};">🎚️ 置信度评估 | Confidence Assessment</h4>
                <div class="bar-container" style="margin: 15px 0;">
                    <div class="bar-fill" style="width: {score*100:.1f}%; background: linear-gradient(90deg, {color}, {color}dd);">
                        {score*100:.1f}%
                    </div>
                </div>
                <div class="confidence-level">
                    <strong>级别 | Level:</strong> 
                    <span style="color: {color}; font-weight: bold;">{level.replace('_', ' ').upper()}</span>
                </div>
                <p><strong>描述 | Description:</strong> {confidence['description']}</p>
                <p><strong>可靠性 | Reliability:</strong> {confidence['reliability']}</p>
            </div>
"""


def _generate_recommendations_html(recommendations: List[str]) -> str:
    """Generate recommendations section HTML"""
    items_html = ''.join([f'<div class="recommendation-item">{rec}</div>' for rec in recommendations])
    
    return f"""
            <div class="recommendations">
                <h4>💡 智能推荐 | Recommendations</h4>
                {items_html}
            </div>
"""


def _generate_feature_analysis_html(analysis: Dict) -> str:
    """Generate feature vector analysis HTML"""
    cosine_sim = analysis['cosine_similarity']
    l2_dist = analysis['l2_distance']
    
    # Generate top divergent dimensions table
    divergent_rows = ''.join([
        f"""<tr>
            <td>维度 {dim['dimension']}</td>
            <td>{dim['query_value']:.4f}</td>
            <td>{dim['result_value']:.4f}</td>
            <td style="color: #F44336; font-weight: bold;">{dim['difference']:.4f}</td>
        </tr>"""
        for dim in analysis['top_divergent_dims']
    ])
    
    # Generate top similar dimensions table
    similar_rows = ''.join([
        f"""<tr>
            <td>维度 {dim['dimension']}</td>
            <td>{dim['query_value']:.4f}</td>
            <td>{dim['result_value']:.4f}</td>
            <td style="color: #4CAF50; font-weight: bold;">{dim['difference']:.4f}</td>
        </tr>"""
        for dim in analysis['top_similar_dims']
    ])
    
    return f"""
            <div class="analysis-section" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-color: #ff9800;">
                <h4 style="color: #e65100;">🔬 特征向量分析 | Feature Vector Analysis</h4>
                
                <div class="details-grid">
                    <div class="detail-item" style="border-left-color: #ff9800;">
                        <div class="detail-label">余弦相似度 | Cosine Similarity</div>
                        <div class="detail-value">{cosine_sim:.4f}</div>
                    </div>
                    <div class="detail-item" style="border-left-color: #ff9800;">
                        <div class="detail-label">L2 距离 | L2 Distance</div>
                        <div class="detail-value">{l2_dist:.4f}</div>
                    </div>
                    <div class="detail-item" style="border-left-color: #ff9800;">
                        <div class="detail-label">平均绝对差异 | Mean Abs Diff</div>
                        <div class="detail-value">{analysis['mean_absolute_difference']:.4f}</div>
                    </div>
                    <div class="detail-item" style="border-left-color: #ff9800;">
                        <div class="detail-label">最大绝对差异 | Max Abs Diff</div>
                        <div class="detail-value">{analysis['max_absolute_difference']:.4f}</div>
                    </div>
                </div>
                
                <p style="background: white; padding: 12px; border-radius: 6px; margin: 15px 0;">
                    <strong>向量解释 | Interpretation:</strong> {analysis['vector_interpretation']}
                </p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                    <div style="background: white; padding: 15px; border-radius: 8px;">
                        <h5 style="margin-top: 0; color: #F44336;">📉 差异最大的维度 | Top Divergent Dimensions</h5>
                        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                            <thead>
                                <tr style="background: #f5f5f5; text-align: left;">
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">维度</th>
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">查询值</th>
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">结果值</th>
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">差异</th>
                                </tr>
                            </thead>
                            <tbody>
                                {divergent_rows}
                            </tbody>
                        </table>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px;">
                        <h5 style="margin-top: 0; color: #4CAF50;">📈 最相似的维度 | Top Similar Dimensions</h5>
                        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                            <thead>
                                <tr style="background: #f5f5f5; text-align: left;">
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">维度</th>
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">查询值</th>
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">结果值</th>
                                    <th style="padding: 8px; border-bottom: 2px solid #ddd;">差异</th>
                                </tr>
                            </thead>
                            <tbody>
                                {similar_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
"""


def _generate_contributions_html(contrib: Dict) -> str:
    """Generate contributions HTML"""
    if 'stage1_weight' in contrib:
        # Weighted fusion
        return f"""
                <div class="contributions">
                    <div class="contribution-card">
                        <h4>Stage 1 Contribution</h4>
                        <p>Weight: {contrib['stage1_weight']:.2f}</p>
                        <p>Contribution: {contrib['stage1_contribution']:.4f}</p>
                        <p>Percentage: {contrib['stage1_percentage']:.1f}%</p>
                    </div>
                    <div class="contribution-card">
                        <h4>Stage 2 Contribution</h4>
                        <p>Weight: {contrib['stage2_weight']:.2f}</p>
                        <p>Contribution: {contrib['stage2_contribution']:.4f}</p>
                        <p>Percentage: {contrib['stage2_percentage']:.1f}%</p>
                    </div>
                </div>
"""
    else:
        # RRF or Borda fusion
        return f"""
                <div class="contributions">
                    <div class="contribution-card">
                        <h4>Stage 1 Contribution</h4>
                        <p>Contribution: {contrib['stage1_contribution']:.4f}</p>
                        <p>Percentage: {contrib['stage1_percentage']:.1f}%</p>
                    </div>
                    <div class="contribution-card">
                        <h4>Stage 2 Contribution</h4>
                        <p>Contribution: {contrib['stage2_contribution']:.4f}</p>
                        <p>Percentage: {contrib['stage2_percentage']:.1f}%</p>
                    </div>
                </div>
"""


def _generate_stage2_details_html(details: Dict) -> str:
    """Generate stage2 details HTML"""
    return f"""
            <div class="stage2-details">
                <h4>🔬 Sequence-Level Analysis</h4>
                <div class="details-grid">
                    <div class="detail-item">
                        <div class="detail-label">Total Distance</div>
                        <div class="detail-value">{details['total_distance']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Command Match Rate</div>
                        <div class="detail-value">{details['cmd_match_rate']:.1%}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Command Matches</div>
                        <div class="detail-value">{details['cmd_matches']} / {details['sequence_length']}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Avg Param Distance</div>
                        <div class="detail-value">{details['avg_param_distance_per_step']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Max Param Distance</div>
                        <div class="detail-value">{details['max_param_distance_per_step']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Sequence Lengths</div>
                        <div class="detail-value">Q:{details['query_seq_len']} / C:{details['candidate_seq_len']}</div>
                    </div>
                </div>
            </div>
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_json_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    query_path = data.get('query_path', 'Unknown')
    
    generate_html_visualization(results, query_path)
