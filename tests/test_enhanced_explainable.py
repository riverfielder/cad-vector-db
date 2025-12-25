"""Test enhanced explainable retrieval features"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_vectordb.core.retrieval import TwoStageRetrieval, macro_distance
from cad_vectordb.core.feature import extract_feature
from cad_vectordb.utils.visualization import generate_html_visualization
import faiss


def create_mock_data():
    """Create mock data for testing"""
    # Create mock vectors
    np.random.seed(42)
    n_samples = 50
    seq_len = 7
    
    vectors = []
    ids = []
    metadata = []
    
    for i in range(n_samples):
        # Create mock CAD vectors (seq_len, 33)
        vec = np.random.randn(seq_len, 33).astype('float32')
        vec[:, 0] = np.random.randint(0, 10, seq_len)  # Commands
        vectors.append(vec)
        ids.append(f'test_{i:04d}.h5')
        metadata.append({
            'id': f'test_{i:04d}.h5',
            'file_path': f'test_{i:04d}.h5',
            'subset': f'{i//10:04d}',
            'seq_len': seq_len
        })
    
    return vectors, ids, metadata


def test_enhanced_explanation():
    """Test enhanced explanation features"""
    print('='*70)
    print('测试增强的可解释检索功能 | Testing Enhanced Explainable Retrieval')
    print('='*70)
    
    # Create mock data
    print('\n1. 创建测试数据...')
    vectors, ids, metadata = create_mock_data()
    
    # Extract features and build index
    print('2. 构建索引...')
    features = np.array([extract_feature(v) for v in vectors], dtype='float32')
    
    # Build FAISS index
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features)
    
    # Create retrieval system
    retrieval = TwoStageRetrieval(index, ids, metadata)
    
    # Monkey patch _load_macro_vec to use our mock data
    def mock_load_macro_vec(file_path):
        idx = ids.index(file_path) if file_path in ids else 0
        return vectors[idx]
    retrieval._load_macro_vec = mock_load_macro_vec
    
    # Test query
    print('3. 执行可解释检索...')
    query_vec = vectors[0]  # Use first vector as query
    query_path = ids[0]
    
    results, explanation = retrieval.search(
        query_vec,
        query_path,
        k=5,
        stage1_topn=20,
        fusion_method='weighted',
        alpha=0.6,
        beta=0.4,
        explainable=True
    )
    
    print(f'\n✅ 返回 {len(results)} 个结果')
    
    # Verify explanation structure
    print('\n4. 验证解释结构...')
    required_fields = [
        'top_match', 'fusion_method', 'stage1_similarity', 'stage2_similarity',
        'final_score', 'stage1_quality', 'stage2_quality',
        'stage1_interpretation', 'stage2_interpretation',
        'match_analysis', 'confidence', 'recommendations', 'feature_analysis'
    ]
    
    for field in required_fields:
        assert field in explanation, f"❌ 缺少字段: {field}"
    print('✅ 所有必需字段都存在')
    
    # Display detailed explanation
    print('\n' + '='*70)
    print('详细解释 | Detailed Explanation')
    print('='*70)
    
    print(f'\n🎯 最佳匹配: {explanation["top_match"]["id"]}')
    print(f'   最终得分: {explanation["final_score"]:.4f}')
    print(f'   融合方法: {explanation["fusion_method"]}')
    
    print(f'\n📊 相似度分解:')
    print(f'   Stage 1: {explanation["stage1_similarity"]:.4f} ({explanation["stage1_quality"]})')
    print(f'   解释: {explanation["stage1_interpretation"]}')
    print(f'   Stage 2: {explanation["stage2_similarity"]:.4f} ({explanation["stage2_quality"]})')
    print(f'   解释: {explanation["stage2_interpretation"]}')
    
    if 'contributions' in explanation:
        contrib = explanation['contributions']
        print(f'\n📈 贡献分析:')
        print(f'   Stage 1: {contrib["stage1_percentage"]:.1f}% (权重: {contrib["stage1_weight"]})')
        print(f'   Stage 2: {contrib["stage2_percentage"]:.1f}% (权重: {contrib["stage2_weight"]})')
    
    analysis = explanation['match_analysis']
    print(f'\n🎯 匹配分析:')
    print(f'   类型: {analysis["match_type"]}')
    print(f'   描述: {analysis["description"]}')
    print(f'   一致性: {analysis["consistency"]} (差异: {analysis["similarity_difference"]:.4f})')
    
    confidence = explanation['confidence']
    print(f'\n🎚️ 置信度评估:')
    print(f'   得分: {confidence["score"]:.4f}')
    print(f'   级别: {confidence["level"]}')
    print(f'   描述: {confidence["description"]}')
    print(f'   可靠性: {confidence["reliability"]}')
    
    feat = explanation['feature_analysis']
    print(f'\n🔬 特征向量分析:')
    print(f'   余弦相似度: {feat["cosine_similarity"]:.4f}')
    print(f'   L2距离: {feat["l2_distance"]:.4f}')
    print(f'   平均差异: {feat["mean_absolute_difference"]:.4f}')
    print(f'   解释: {feat["vector_interpretation"]}')
    
    print(f'\n   差异最大的3个维度:')
    for dim in feat['top_divergent_dims'][:3]:
        print(f'     维度 {dim["dimension"]}: 查询={dim["query_value"]:.4f}, 结果={dim["result_value"]:.4f}, 差={dim["difference"]:.4f}')
    
    print(f'\n💡 智能推荐 ({len(explanation["recommendations"])} 条):')
    for i, rec in enumerate(explanation['recommendations'], 1):
        print(f'   {i}. {rec}')
    
    # Test visualization
    print('\n5. 测试HTML可视化生成...')
    output_file = '/tmp/test_explanation.html'
    try:
        # Add explanation to result for visualization
        results[0]['explanation'] = explanation
        
        generate_html_visualization(
            results[:3],
            query_path=query_path,
            output_file=output_file
        )
        
        # Check file size
        import os
        file_size = os.path.getsize(output_file)
        print(f'✅ HTML文件已生成: {output_file} ({file_size} bytes)')
        
        # Basic content validation
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '可解释检索结果' in content
            assert '相似度分解' in content
            assert '匹配分析' in content
            assert '置信度评估' in content
            assert '特征向量分析' in content
            assert '智能推荐' in content
        print('✅ HTML内容验证通过')
        
    except Exception as e:
        print(f'❌ 可视化生成失败: {e}')
        raise
    
    print('\n' + '='*70)
    print('✅ 所有测试通过! 增强的可解释检索功能正常工作')
    print('='*70)
    
    return True


if __name__ == '__main__':
    test_enhanced_explanation()
