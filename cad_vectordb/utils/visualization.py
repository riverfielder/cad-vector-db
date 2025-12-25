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
    html_parts.append("</body>\n</html>\n")
    
    # Write to file
    html = ''.join(html_parts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Visualization saved to: {output_file}")
    return output_file


def _get_html_header(query_path: str, num_results: int) -> str:
    """Generate HTML header with CSS styles"""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Explainable Retrieval Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .query-info {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .result-card {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid #4CAF50; }}
        .result-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .rank-badge {{ background: #4CAF50; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .score-badge {{ background: #2196F3; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .result-id {{ font-size: 18px; font-weight: bold; color: #333; flex-grow: 1; margin: 0 20px; }}
        .metadata {{ background: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 4px; font-size: 14px; }}
        .explanation-section {{ margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 4px; }}
        .similarity-bar {{ margin: 15px 0; }}
        .bar-label {{ font-weight: bold; margin-bottom: 5px; display: flex; justify-content: space-between; }}
        .bar-container {{ background: #ddd; border-radius: 10px; height: 30px; overflow: hidden; }}
        .bar-fill {{ height: 100%; display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold; }}
        .bar-stage1 {{ background: linear-gradient(90deg, #FF6B6B, #FF8E53); }}
        .bar-stage2 {{ background: linear-gradient(90deg, #4ECDC4, #44A08D); }}
        .bar-final {{ background: linear-gradient(90deg, #667eea, #764ba2); }}
        .interpretation {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 10px 0; }}
        .contributions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }}
        .contribution-card {{ background: white; padding: 15px; border-radius: 4px; border: 2px solid #ddd; }}
        .stage2-details {{ margin-top: 15px; padding: 15px; background: #e3f2fd; border-radius: 4px; }}
        .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin: 10px 0; }}
        .detail-item {{ background: white; padding: 10px; border-radius: 4px; border-left: 3px solid #2196F3; }}
        .detail-label {{ font-weight: bold; color: #555; font-size: 12px; }}
        .detail-value {{ font-size: 16px; color: #333; margin-top: 5px; }}
        .fusion-info {{ background: #f3e5f5; padding: 10px 15px; border-radius: 4px; margin: 10px 0; border-left: 4px solid #9c27b0; }}
    </style>
</head>
<body>
    <h1>🔍 Explainable Retrieval Results</h1>
    
    <div class="query-info">
        <h3>Query Information</h3>
        <p><strong>Query File:</strong> <code>{query_path}</code></p>
        <p><strong>Total Results:</strong> {num_results}</p>
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
    """Generate explanation section HTML"""
    exp = result['explanation']
    stage1_sim = exp['stage1_similarity']
    stage2_sim = exp['stage2_similarity']
    final_score = exp['final_score']
    
    parts = [f"""
        <div class="explanation-section">
            <h3>📊 Similarity Breakdown</h3>
            
            <div class="similarity-bar">
                <div class="bar-label">
                    <span>Stage 1 (Feature-level)</span>
                    <span>{stage1_sim:.4f}</span>
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
                    <span>Stage 2 (Sequence-level)</span>
                    <span>{stage2_sim:.4f}</span>
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
                    <span>Final Fused Score</span>
                    <span>{final_score:.4f}</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill bar-final" style="width: {final_score*100:.1f}%">
                        {final_score*100:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="fusion-info">
                <strong>Fusion Method:</strong> {exp['fusion_method']}
"""]
    
    # Add contributions if available
    if 'contributions' in exp:
        parts.append(_generate_contributions_html(exp['contributions']))
    
    parts.append("            </div>\n")
    
    # Add stage2 details if available
    if 'stage2_details' in result:
        parts.append(_generate_stage2_details_html(result['stage2_details']))
    
    parts.append("        </div>\n")
    return ''.join(parts)


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
