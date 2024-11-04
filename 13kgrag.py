# 源文档: https://milvus.io/docs/graph_rag_with_milvus.md

import numpy as np

from collections import defaultdict
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm




milvus_client = MilvusClient(uri="./milvus.db")

llm = '1'

from langchain_community.chat_models import ChatZhipuAI


messages = [
    ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
    ("human", "我喜欢编程。"),
]
zhipuai_chat=ChatZhipuAI(model="glm-4-Flash",temperature= 0.99,
                    max_tokens=999999,api_key="9eb77fe2543c68240713c55742979c85.X54mEVPMMJk7WJGw")
llm = zhipuai_chat







# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


nano_dataset = [
    {
        "passage": "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "triplets": [
            ["Jakob Bernoulli", "made significant contributions to", "calculus"],
            [
                "Jakob Bernoulli",
                "made significant contributions to",
                "the theory of probability",
            ],
            ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
            ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
            ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
            ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
        ],
    },
    {
        "passage": "Johann Bernoulli (1667–1748): Johann, Jakob’s younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "triplets": [
            [
                "Johann Bernoulli",
                "was a major figure of",
                "the development of calculus",
            ],
            ["Johann Bernoulli", "was", "Jakob's younger brother"],
            ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
            ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
            ["Johann Bernoulli", "contributed to", "the calculus of variations"],
            ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
        ],
    },
    {
        "passage": "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli’s principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
        "triplets": [
            ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
            ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
            ["Daniel Bernoulli", "made major contributions to", "probability"],
            ["Daniel Bernoulli", "made major contributions to", "statistics"],
            ["Daniel Bernoulli", "is most famous for", "Bernoulli’s principle"],
            [
                "Bernoulli’s principle",
                "is fundamental to",
                "the understanding of aerodynamics",
            ],
        ],
    },
    {
        "passage": "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli’s influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
        "triplets": [
            [
                "Leonhard Euler",
                "had a significant relationship with",
                "the Bernoulli family",
            ],
            ["leonhard Euler", "was born in", "Basel"],
            ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
            ["Johann Bernoulli's influence", "was profound on", "Euler"],
        ],
    },
]





entityid_2_relationids = defaultdict(list)
relationid_2_passageids = defaultdict(list)


# entities  relations  passages这3个作为知识注入即可.
entities = []
relations = []
passages = []
for passage_id, dataset_info in enumerate(nano_dataset):
    passage, triplets = dataset_info["passage"], dataset_info["triplets"]
    passages.append(passage)
    for triplet in triplets:
        if triplet[0] not in entities:
            entities.append(triplet[0])
        if triplet[2] not in entities:
            entities.append(triplet[2])
        relation = " ".join(triplet)
        if relation not in relations:
            relations.append(relation) # 这里看出来关系看做双边, a__关系___b, 那么 这个关系写入a的数组和b的数组.
            entityid_2_relationids[entities.index(triplet[0])].append(
                len(relations) - 1
            )
            entityid_2_relationids[entities.index(triplet[2])].append(
                len(relations) - 1
            )
        relationid_2_passageids[relations.index(relation)].append(passage_id)

print(1)
print('知识')
print( entities,  relations , passages)




import openai

# Assume that the model is already launched.
# The api_key can't be empty, any string is OK.
client = openai.Client(api_key="not empty", base_url="http://192.168.254.206:9997/v1")

from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-small-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def emb(i):
    
    # 单机测试:
    # a=client.embeddings.create(model='bge-base-zh-v1.5', input=[i])
    embeddings_1 = model.encode(i)
    return embeddings_1
    print(1)
    if type(i)==str:
        return list(np.random.rand(768))

    return  [list(np.random.rand(768))]*len(i)
    print(a)


# a=emb('aa')
pass


embedding_dim = len(emb("foo"))


def create_milvus_collection(collection_name: str):#每次都重置数据库.
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        consistency_level="Strong",
    )


entity_col_name = "entity_collection"
relation_col_name = "relation_collection"
passage_col_name = "passage_collection"
create_milvus_collection(entity_col_name)
create_milvus_collection(relation_col_name)
create_milvus_collection(passage_col_name)








def milvus_insert(
    collection_name: str,
    text_list: list[str],
):
    batch_size = 512
    for row_id in tqdm(range(0, len(text_list), batch_size), desc="Inserting"):
        batch_texts = text_list[row_id : row_id + batch_size]
        batch_embeddings = emb(batch_texts)

        batch_ids = [row_id + j for j in range(len(batch_texts))]
        batch_data = [
            {
                "id": id_,
                "text": text,
                "vector": vector,
            }
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        milvus_client.insert(
            collection_name=collection_name,
            data=batch_data,
        )


milvus_insert(
    collection_name=relation_col_name,
    text_list=relations,
)

milvus_insert(
    collection_name=entity_col_name,
    text_list=entities,
)

milvus_insert(
    collection_name=passage_col_name,
    text_list=passages,
)


milvus_client




# 应用点.
# 1.这个是知识图谱的搜索技术. 对图谱信息进行向量化搜索.




query = "What contribution did the son of Euler's teacher make?"

query_ner_list = ["Euler"] # 通过ner技术抽出来的.

query_ner_embeddings = [
    emb(query_ner) for query_ner in query_ner_list
]

top_k = 3

entity_search_res = milvus_client.search(
    collection_name=entity_col_name,
    data=query_ner_embeddings,
    limit=top_k,
    output_fields=["id"],
)

query_embedding = emb(query)

relation_search_res = milvus_client.search(
    collection_name=relation_col_name,
    data=[query_embedding],
    limit=top_k,
    output_fields=["id"],
)[0]


# 2.Expand Subgraph
# 把上一步拿到的节点和关系进行拓展.
print( entities,  relations , passages, '这里面realtion变量虽然指的是3元对,但是下面图中的relation实际指的是边的分类. 或者说就是 三元对的中间部分的字符串, 并且注意这里面重复的字符串算不同的关系,因为他们head , tail不同')

entity_relation_adj = np.zeros((len(entities), len(relations))) # 邻接矩阵: 横坐标是entity, 纵坐标是relation, 如果存在关系就1.
for entity_id, entity in enumerate(entities):
    entity_relation_adj[entity_id, entityid_2_relationids[entity_id]] = 1

entity_relation_adj = csr_matrix(entity_relation_adj)
# entity_relation_adj = entity_relation_adj

entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T # entity之间的关系矩阵, 关系表示可以通过一条边相连的.
relation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj # 两个关系之间的关系, 表示两个关系之间可以 边(i)-->元素-->边(j), 这个路径的数量(也就是中间元素的取法). 表示这个矩阵relation_adj_1_degree(i,j). !!!!!!!!!!!!!!!!!这个理解很重要.

target_degree = 1 # 拓展一个degree.

entity_adj_target_degree = entity_adj_1_degree
for _ in range(target_degree - 1):
    entity_adj_target_degree = entity_adj_target_degree * entity_adj_1_degree
relation_adj_target_degree = relation_adj_1_degree
for _ in range(target_degree - 1):
    relation_adj_target_degree = relation_adj_target_degree * relation_adj_1_degree
# entity_adj_target_degree : 元素到元素之间的链接, entity_relation_adj: 元素到关系支架难的链接. 乘积就是从entity到target的关系的临街矩阵.
entity_relation_adj_target_degree = entity_adj_target_degree @ entity_relation_adj
# entity_relation_adj_target_degree= 元素到边到元素*  元素-->边













# 拓展.

expanded_relations_from_relation = set()
expanded_relations_from_entity = set()

filtered_hit_relation_ids = [
    relation_res["entity"]["id"]
    for relation_res in relation_search_res
    # if relation_res['distance'] > relation_sim_filter_thresh
]
for hit_relation_id in filtered_hit_relation_ids:
    expanded_relations_from_relation.update(
        relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()
    )
  # 过滤
filtered_hit_entity_ids = [
    one_entity_res["entity"]["id"]
    for one_entity_search_res in entity_search_res
    for one_entity_res in one_entity_search_res
    # if one_entity_res['distance'] > entity_sim_filter_thresh
]
# 根据query找到的entity, 拿entity去拓展图entity_relation_adj_target_degree 里面找到信息加到expanded_relations_from_entity里面.
for filtered_hit_entity_id in filtered_hit_entity_ids:
    expanded_relations_from_entity.update(
        entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()
    )

relation_candidate_ids = list(
    expanded_relations_from_relation | expanded_relations_from_entity
)

relation_candidate_texts = [
    relations[relation_id] for relation_id in relation_candidate_ids
]





#=======下面是一个例子： oneshot 的输入
query_prompt_one_shot_input = """I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be useful to answer the given question. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.

Question:
When was the mother of the leader of the Third Crusade born?

Relationship descriptions:
[1] Eleanor was born in 1122.
[2] Eleanor married King Louis VII of France.
[3] Eleanor was the Duchess of Aquitaine.
[4] Eleanor participated in the Second Crusade.
[5] Eleanor had eight children.
[6] Eleanor was married to Henry II of England.
[7] Eleanor was the mother of Richard the Lionheart.
[8] Richard the Lionheart was the King of England.
[9] Henry II was the father of Richard the Lionheart.
[10] Henry II was the King of England.
[11] Richard the Lionheart led the Third Crusade.

"""
# 下面是例子oneshot的输出。
query_prompt_one_shot_output = """{"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born in 1122"]}"""

query_prompt_template = """Question:
{question}

Relationship descriptions:
{relation_des_str}

"""


def rerank_relations(
    query: str, relation_candidate_texts: list[str], relation_candidate_ids: list[str]
) -> list[int]:
    relation_des_str = "\n".join(
        map(
            lambda item: f"[{item[0]}] {item[1]}",
            zip(relation_candidate_ids, relation_candidate_texts),
        )
    ).strip()
    rerank_prompts = ChatPromptTemplate.from_messages(
        [
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_template(query_prompt_template),
        ]
    )
    rerank_chain = (
        rerank_prompts
        | llm.bind(response_format={"type": "json_object"})
        | JsonOutputParser()
    )
    rerank_res = rerank_chain.invoke(
        {"question": query, "relation_des_str": relation_des_str}
    )# 使用大模型进行rerank, 重新排序知识.
    rerank_relation_ids = []
    rerank_relation_lines = rerank_res["useful_relationships"]
    id_2_lines = {}
    for line in rerank_relation_lines:
        id_ = int(line[line.find("[") + 1 : line.find("]")])
        id_2_lines[id_] = line.strip()
        rerank_relation_ids.append(id_)
    return rerank_relation_ids

#======最后通过rerank relation 修改了文档上下文检索, 利用知识图谱增强了prompt
rerank_relation_ids = rerank_relations(
    query,
    relation_candidate_texts=relation_candidate_texts,
    relation_candidate_ids=relation_candidate_ids,
)

final_top_k = 2

final_passages = []
final_passage_ids = []
for relation_id in rerank_relation_ids:
    for passage_id in relationid_2_passageids[relation_id]:
        if passage_id not in final_passage_ids:
            final_passage_ids.append(passage_id)
            final_passages.append(passages[passage_id])
passages_from_our_method = final_passages[:final_top_k]#!!!!!!!!!!passages_from_our_method就是我们图rag得到的上下文.





naive_passage_res = milvus_client.search(
    collection_name=passage_col_name,
    data=[query_embedding],
    limit=final_top_k,
    output_fields=["text"],
)[0]
passages_from_naive_rag = [res["entity"]["text"] for res in naive_passage_res]

print(
    f"Passages retrieved from naive RAG: \n{passages_from_naive_rag}\n\n"
    f"Passages retrieved from our method GRAPHRAG: \n{passages_from_our_method}\n\n"
)

print('可以看到直接从query来做向量搜索，那么大概率是找不到euler信息的。从rag，先ner出euler那么是可以找到euler精确的上下文的。')



#===下面进行问答:
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Use the following pieces of retrieved context to answer the question. If there is not enough information in the retrieved context to answer the question, just say that you don't know.
Question: {question}
Context: {context}
Answer:""",
        )
    ]
)

rag_chain = prompt | llm | StrOutputParser()

answer_from_naive_rag = rag_chain.invoke(
    {"question": query, "context": "\n".join(passages_from_naive_rag)}
)
answer_from_our_method = rag_chain.invoke(
    {"question": query, "context": "\n".join(passages_from_our_method)}
)

print(
    f"Answer from naive RAG: {answer_from_naive_rag}\n\nAnswer from our method: {answer_from_our_method}"
)

print('看出效果是强了不少')















