import json
import os

import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))
from get_text_embedding import get_text_embedding

from dotenv import load_dotenv
from openai import OpenAI

import re
from collections import defaultdict
import math
import jieba

# 统一加载项目根目录的.env
load_dotenv()

class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class EmbeddingModel:
    def __init__(self, batch_size: int = 64):
        self.api_key = os.getenv('GUIJI_API_KEY')
        self.base_url = os.getenv('GUIJI_BASE_URL')
        self.embedding_model = os.getenv('GUIJI_EMBEDDING_MODEL')
        self.batch_size = batch_size
        if not self.api_key or not self.base_url:
            raise ValueError('请在.env中配置GUIJI_API_KEY和GUIJI_BASE_URL')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_text_embedding(
            texts,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np
        if not self.embeddings:
            return []
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]
    
    def search_with_scores(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索并返回带得分的结果"""
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np
        
        if not self.embeddings:
            return []
        
        emb_matrix = np.array(self.embeddings)  # 形状为(n, d)的矩阵，其中n是文档数量，d是向量维度
        query_emb = np.array(query_embedding)  # 形状为(d,)的查询向量
        # 结果是一个长度为n的向量，每个元素表示对应文档与查询的相似度，通过除以分母得到归一化后的数值
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        
        # 获取top_k的索引和得分
        idxs = sims.argsort()[::-1][:top_k]  # 对相似度得分进行升序排序，返回索引，再将排序结果反转，变为降序（相似度从高到低），取前k个最高相似度的索引
        scores = sims[idxs]  # 获取对应的相似度得分
        
        results = []
        for i, idx in enumerate(idxs):  # 遍历排序后的索引
            chunk = self.chunks[idx].copy()  # 对每个文档块进行深拷贝，避免修改原始数据
            chunk['similarity_score'] = float(scores[i])  # 添加相似度得分字段 similarity_score
            results.append(chunk)  # 将结果添加到返回列表中
        
        return results

class SimpleRAG:
    def __init__(self, chunk_json_path: str, model_path: str = None, batch_size: int = 32):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()
        self.keyword_search = None  # 关键字检索器
        self.chunks = None  # 存储原始chunks
    def setup(self):
        print("加载所有页chunk...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 个chunk")
        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, embeddings)
        # 初始化关键字检索器
        self.keyword_search = KeywordSearch(self.chunks)
        print("RAG向量库构建完成！")

    def hybrid_search(self, question: str, top_k: int = 10, vector_weight: float = 0.5):
        """
        混合检索：结合向量检索和关键字检索
        """
        # 向量检索
        q_emb = self.embedding_model.embed_text(question)
        vector_results = self.vector_store.search_with_scores(q_emb, top_k * 2)
        
        # 关键字检索
        keyword_results = self.keyword_search.search(question, top_k * 2)
        
        # 使用RRF算法融合结果
        fused_results = self._rrf_fusion(vector_results, keyword_results, top_k, vector_weight)
        
        return fused_results
    
    def _rrf_fusion(self, vector_results, keyword_results, top_k, vector_weight=0.5):
        """
        RRF (Reciprocal Rank Fusion) 算法实现
        """
        # 构建文档ID到排名的映射
        vector_ranks = {}
        keyword_ranks = {}
        
        # 向量检索结果排名 (注意：vector_results是chunks列表，需要找到对应索引)
        # 遍历向量检索结果（已排序的chunks列表）
        for i, chunk in enumerate(vector_results):
            # 找到chunk在原始chunks中的索引
            for idx, original_chunk in enumerate(self.chunks):
                if chunk['id'] == original_chunk['id']:
                    # 建立映射关系：数据块的索引号与它的排名
                    vector_ranks[idx] = i + 1
                    break
        
        # 关键字检索结果排名
        for idx, score in keyword_results:
            keyword_ranks[idx] = idx + 1  # keyword_results已经是(索引, 得分)的列表
        
        # 计算RRF得分
        rrf_scores = defaultdict(float)
        k = 60  # RRF参数
        
        # 向量检索的RRF得分
        for doc_id, rank in vector_ranks.items():
            rrf_scores[doc_id] += vector_weight * (1 / (rank + k))
        
        # 关键字检索的RRF得分
        for doc_id, rank in keyword_ranks.items():
            rrf_scores[doc_id] += (1 - vector_weight) * (1 / (rank + k))
        
        # 按RRF得分排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 返回融合后的结果
        fused_chunks = []
        for doc_id, score in sorted_docs:
            if doc_id < len(self.chunks):
                chunk_with_score = self.chunks[doc_id].copy()
                chunk_with_score['rrf_score'] = score
                fused_chunks.append(chunk_with_score)
        
        return fused_chunks
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {
            "question": question,
            "chunks": results
        }

    def generate_answer(self, question: str, top_k: int = 3, use_hybrid: bool = False) -> Dict[str, Any]:
        """
        检索+大模型生成式回答，返回结构化结果
        """
        if use_hybrid:
            # 使用混合检索
            chunks = self.hybrid_search(question, top_k * 2)  # 检索更多候选用作RRF
            # 取前top_k个结果
            chunks = chunks[:top_k]
        else:
            # 使用原始向量检索
            q_emb = self.embedding_model.embed_text(question)
            chunks = self.vector_store.search(q_emb, top_k)
        
        zhipu_api_key = os.getenv('ZHIPU_API_KEY')
        zhipu_base_url = os.getenv('ZHIPU_BASE_URL')
        zhipu_model = os.getenv('ZHIPU_TEXT_MODEL')
        if not zhipu_api_key or not zhipu_base_url or not zhipu_model:
            raise ValueError('请在.env中配置ZHIPU_API_KEY、ZHIPU_BASE_URL、ZHIPU_TEXT_MODEL')
        # 使用嵌入模型将问题转换为向量表示
        #q_emb = self.embedding_model.embed_text(question)
        # 向量存储中搜索最相似的top_k个文档块
        #chunks = self.vector_store.search(q_emb, top_k)
        '''拼接检索内容，带上元数据，格式化为上下文字符串，格式为：
            [文件名]filename.pdf [页码]5
            content content content...
        '''
        context = "\n".join([
            f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}" for c in chunks
        ])
        # 明确要求输出JSON格式 answer/page/filename
        prompt = (
            f"你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
            f"请严格按照如下JSON格式输出：\n"
            f'{{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}}'"\n"
            f"检索内容：\n{context}\n\n问题：{question}\n"
            f"请确保输出内容为合法JSON字符串，不要输出多余内容。"
        )
        client = OpenAI(api_key=zhipu_api_key, base_url=zhipu_base_url)
        completion = client.chat.completions.create(
            model=zhipu_model,
            messages=[
                {"role": "system", "content": "你是一名专业的金融分析助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        import json as pyjson
        sys.path.append(os.path.dirname(__file__))
        from extract_json_array import extract_json_array
        # 获取模型原始输出，并使用 extract_json_array 工具提取其中的JSON内容。
        raw = completion.choices[0].message.content.strip()
        # 用 extract_json_array 提取 JSON 对象
        json_str = extract_json_array(raw, mode='objects')
        '''
        这个复杂的错误处理逻辑是为了确保即使模型输出不符合预期格式，也能返回合理的结果：
        (1)如果成功提取并解析JSON，则使用解析结果
        (2)如果解析失败，则回退到原始输出
        (3)始终确保返回文件名和页码信息（优先使用解析结果，否则使用第一个检索到的块的元数据）
        '''
        if json_str:
            # 如果提取到了JSON字符串，则尝试使用Python的json模块解析它
            try:
                arr = pyjson.loads(json_str)
                # 检查解析结果是否为列表类型及列表是否非空
                if isinstance(arr, list) and arr:
                    # 获取列表中的第一个对象
                    j = arr[0]
                    # 从对象中提取 answer、filename 和 page 字段，如果字段不存在则返回空字符串
                    answer = j.get('answer', '')
                    filename = j.get('filename', '')
                    page = j.get('page', '')
                else:
                    # 当JSON解析成功但结果不是非空列表时，使用原始输出作为答案，并从检索到的第一个文档块中提取文件名和页码。
                    answer = raw
                    filename = chunks[0]['metadata']['file_name'] if chunks else ''
                    page = chunks[0]['metadata']['page'] if chunks else ''
            # 当JSON解析失败（如格式错误）时，同样使用原始输出作为答案，并从检索到的第一个文档块中提取元数据。
            except Exception:
                answer = raw
                filename = chunks[0]['metadata']['file_name'] if chunks else ''
                page = chunks[0]['metadata']['page'] if chunks else ''
        # 当 extract_json_array 未能提取到任何JSON内容时，直接使用模型的原始输出作为答案，并从检索到的第一个文档块中提取元数据。
        else:
            answer = raw
            filename = chunks[0]['metadata']['file_name'] if chunks else ''
            page = chunks[0]['metadata']['page'] if chunks else ''
        # 结构化输出
        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": chunks
        }

class KeywordSearch:
    # 基于TF-IDF的关键字检索系统，TF-IDF算法能有效区分重要词汇和常见词汇，罕见但相关的词汇权重更高
    def __init__(self, chunks):
        self.chunks = chunks
        self.inverted_index = self._build_inverted_index()
    
    # 常见中文停用词
    STOP_WORDS = {'的', '是', '在', '了', '和', '有', '我', '你', '他', '她', '它'}
    def _build_inverted_index(self):
        """构建倒排索引"""
        index = defaultdict(set)  # 创建一个默认值为集合的字典作为倒排索引
        for i, chunk in enumerate(self.chunks):
            # 简单的分词处理
            content = chunk['content'].lower()  # 将内容转为小写以实现大小写不敏感
            # 使用jieba进行中文分词
            words = jieba.lcut(content.lower())
            # 过滤停用词
            words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
            for word in set(words):  # 对单词去重，避免同一文档中重复单词被多次计数
                index[word].add(i)  # 将文档索引添加到对应单词的倒排列表中
        return index
    
    def search(self, query, top_k=10):
        """基于TF-IDF的关键字检索"""
        # 将查询转为小写，提取查询中的关键词
        query_words = jieba.lcut(query.lower())
        if not query_words:
            return []
        
        # 计算每个chunk的得分
        scores = defaultdict(float)
        doc_count = len(self.chunks)
        
        # 对于查询中的每个词
        for word in query_words:
            if word in self.inverted_index:
                # 计算IDF（逆文档频率），IDF = log(总文档数 / 包含该词的文档数)
                # 罕见词具有更高的IDF值，权重更大
                idf = math.log(doc_count / len(self.inverted_index[word]))
                for doc_idx in self.inverted_index[word]:
                    # 计算TF（词频），TF = 词在文档中出现次数 / 文档总词数
                    # 词在文档中出现频率越高，TF值越大
                    content = self.chunks[doc_idx]['content'].lower()
                    tf = content.count(word) / len(content.split()) if content.split() else 0
                    # TF-IDF得分，综合考虑词频和逆文档频率，既考虑词的重要性（IDF），又考虑词在文档中的重要性（TF）
                    scores[doc_idx] += tf * idf
        
        # 按得分排序并返回top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_idx, score) for doc_idx, score in sorted_docs]

if __name__ == '__main__':
    # 路径可根据实际情况调整
    chunk_json_path = os.path.join(os.path.dirname(__file__), 'all_pdf_page_chunks.json')
    rag = SimpleRAG(chunk_json_path)
    rag.setup()

    # 控制测试时读取的题目数量，默认只随机抽取10个，实际跑全部时设为None
    TEST_SAMPLE_NUM = None  # 设置为None则全部跑
    FILL_UNANSWERED = True  # 未回答的也输出默认内容

    # 批量评测脚本：读取测试集，检索+大模型生成，输出结构化结果
    test_path = os.path.join(os.path.dirname(__file__), 'datas/test.json')
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        import concurrent.futures
        import random

        # 记录所有原始索引
        all_indices = list(range(len(test_data)))
        # 随机抽取部分题目用于测试
        selected_indices = all_indices
        if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0:
            if len(test_data) > TEST_SAMPLE_NUM:
                selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))

        def process_one(idx):
            '''接收一个索引 idx，指向测试数据中的某个问题'''
            # 从测试数据中提取对应的问题文本
            item = test_data[idx]
            question = item['question']
            # 使用 tqdm.write 输出当前处理进度，显示格式为 [当前序号/总数量] 正在处理: 问题前30个字符...
            tqdm.write(f"[{selected_indices.index(idx)+1}/{len(selected_indices)}] 正在处理: {question[:30]}...")
            # 使用 SimpleRAG 的 generate_answer 方法处理问题，检索top 5相关文档并生成答案
            result = rag.generate_answer(question, top_k=5, use_hybrid=True)
            return idx, result

        results = []  # 存储所有处理结果
        if selected_indices:  # 确保有待处理的问题
            # 创建最大10个工作线程的线程池,允许同时处理多个问题，提高处理效率
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                '''
                executor.map(process_one, selected_indices):
                    将 process_one 函数应用到所有选中的索引上
                    自动并发执行，无需手动管理线程
                tqdm(..., total=len(selected_indices), desc='并发批量生成'):
                    显示处理进度条
                    总数量为选中问题的数量
                    描述文字为"并发批量生成"
                list(...) 将迭代器转换为列表
                '''
                results = list(tqdm(executor.map(process_one, selected_indices), total=len(selected_indices), desc='并发批量生成'))

        # 先输出一份未过滤的原始结果（含 idx）
        import json
        raw_out_path = os.path.join(os.path.dirname(__file__), 'rag_top1_pred_raw.json')
        with open(raw_out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'已输出原始未过滤结果到: {raw_out_path}')

        # 只保留结果部分，并去除 retrieval_chunks 字段
        idx2result = {idx: {k: v for k, v in r.items() if k != 'retrieval_chunks'} for idx, r in results}
        filtered_results = []
        for idx, item in enumerate(test_data):
            if idx in idx2result:
                filtered_results.append(idx2result[idx])
            elif FILL_UNANSWERED:
                # 未被回答的，补默认内容
                filtered_results.append({
                    "question": item.get("question", ""),
                    "answer": "",
                    "filename": "",
                    "page": "",
                })
        # 输出结构化结果到json
        out_path = os.path.join(os.path.dirname(__file__), 'rag_top1_pred.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        print(f'已输出结构化检索+大模型生成结果到: {out_path}')
    
        