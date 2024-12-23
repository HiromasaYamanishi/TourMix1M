import networkx as nx
from collections import defaultdict
import openai
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import fire
from tqdm import tqdm
import spacy
from pyvis.network import Network
from collections import deque
import japanize_matplotlib
#from lmm import SentenceBertJapanese
import os
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import random
from utils import extract_words_from_response
from utils import english_to_japanese
import re
from typing import List, Tuple, Dict
from collections import OrderedDict


class SAKGBase:
    def __init__(self, load=False):
        
        self.graph_path = '../data/kg/sakg_0724.pkl'
        if load:
            self.load_graph()
        else:
            self.df_review = pd.read_pickle(
                "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
            )
            with open('../data/pairs.pkl', 'rb') as f:
                pairs = pickle.load(f)
                
            le_user = LabelEncoder()
            le_spot = LabelEncoder()
            self.df_review['user_id'] = le_user.fit_transform(self.df_review['url'])
            self.df_review['spot_id'] = le_spot.fit_transform(self.df_review['spot'])
            self.df_review['pairs'] = pairs
            self.nlp = spacy.load('ja_ginza')
                
            self.graph = nx.Graph()
            self.build_graph()
            self.clean_up_edges()
            self.visualize_graph(num_nodes=20)
            self.print_graph_statistics()
            self.save_graph()

    def build_graph(self):
        test_reviews = pd.read_csv('../data/df_review_feature_eval.csv')['conversations'].values
        print(test_reviews)
        for i, row in tqdm(self.df_review.iterrows()):
            user_id, spot, pairs, review = row['user_id'], row['spot'], row['pairs'], row['review']
            if review in test_reviews:continue
            user_node = 'user_' + str(user_id)
            # ノードカテゴリを追加
            self.graph.add_node(user_node, category='user')
            self.graph.add_node(spot, category='spot')
            for adj, noun in pairs:
                self.graph.add_node(noun, category='word')
                self._add_edge_with_count(user_node, noun, adj)
                self._add_edge_with_count(spot, noun, adj)
            #if i==100000:break
            
    def clean_up_edges(self, count_thresh=3):
        for node in self.graph.nodes:
            if self.graph.nodes[node].get('category') == 'spot':
                edges_to_remove = []
                for neighbor in self.graph.neighbors(node):
                    relations = self.graph[node][neighbor]['relations']
                    relations_to_remove = [rel for rel, count in relations.items() if count < count_thresh]
                    for rel in relations_to_remove:
                        del relations[rel]
                    if not relations:
                        edges_to_remove.append((node, neighbor))
                self.graph.remove_edges_from(edges_to_remove)
    
    def _add_edge_with_count(self, node1, node2, relation):
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2, relations={})
        # エッジが既に存在する場合、relationごとのカウントを更新
        if relation in self.graph[node1][node2]['relations']:
            self.graph[node1][node2]['relations'][relation] += 1
        else:
            self.graph[node1][node2]['relations'][relation] = 1

    def save_graph(self, graph_path):
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        # nx.write_gml(self.graph, self.graph_path)
        print(f"Graph saved to {graph_path}")
        
    def load_graph(self):
        with open(self.graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        #graph = nx.read_gml(self.graph_path)
        print(f"Graph loaded from {self.graph_path}")

    def visualize_graph(self, num_nodes=20):
        # グラフの一部を抽出するために、ノード数を制限
        subgraph = self.graph.subgraph(list(self.graph.nodes)[:num_nodes])
        
        # ポジションを設定（レイアウトを指定）
        pos = nx.spring_layout(subgraph)
        
        # エッジのラベルを取得
        edge_labels = { (u, v): str(data['relations']) for u, v, data in subgraph.edges(data=True) }

        net = Network(notebook=True, width="100%", height="800px", font_color="black")
    
        net.from_nx(subgraph)
        
        # ノードのラベルを設定
        for node in net.nodes:
            node['label'] = node['id']
            node['title'] = node['id']
            node['font'] = {'size': 10, 'face': 'Hiragino Sans'}
            node['color'] = 'skyblue'
        
        # エッジのラベルを設定
        for edge in net.edges:
            edge['title'] = edge_labels.get((edge['from'], edge['to']), '')
            edge['font'] = {'size': 8, 'face': 'Hiragino Sans'}
            edge['color'] = 'red'
        
        # グラフを HTML ファイルとして保存
        net.show("../data/kg/sakg.html")
        # pyvis_G = Network()
        # pyvis_G.from_nx(self.graph)
        # pyvis_G.show("../data/kg/sakg.html")
        # ノードの描画
        #nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10,font_family='Hiragino Sans',  font_weight='bold')
        
        # エッジのラベルの描画
        #nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red',font_family='Hiragino Sans', font_size=8)
        
        #plt.savefig('../data/kg/sakg.jpg')

    def get_graph_statistics(self):
        stats = {
            "number_of_nodes": self.graph.number_of_nodes(),
            "number_of_edges": self.graph.number_of_edges(),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "graph_density": nx.density(self.graph),
            #"connected_components": nx.number_connected_components(self.graph),
        }
        return stats

    def print_graph_statistics(self):
        stats = self.get_graph_statistics()
        print("Graph Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    

    def get_n_hop_entities_and_relations(self, start_node, n):
        """
        指定したノードからnホップ離れた全てのエンティティとその関係を取得する

        :param start_node: 開始ノード
        :param n: ホップ数
        :return: nホップ以内のエンティティと関係のリスト
        """
        visited = set()
        queue = deque([(start_node, 0, [])])
        results = []

        while queue:
            node, depth, path = queue.popleft()

            if depth > n:
                continue

            if node not in visited:
                visited.add(node)

                if depth > 0:
                    results.append((node, path))

                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        new_path = path + [(node, edge_data['relations'], neighbor)]
                        queue.append((neighbor, depth + 1, new_path))

        return results

    def print_n_hop_results(self, start_node, n):
        """
        結果を見やすく表示するためのヘルパーメソッド
        """
        results = self.get_n_hop_entities_and_relations(start_node, n)
        for entity, path in results:
            print(f"Entity: {entity}")
            print("Path:")
            for source, relation, target in path:
                print(f"  {source} --[{relation}]--> {target}")
            print()
            
    def get_embedding_prompt(self, split=1):
        all_prompt_df = pd.read_csv('../data/all_prompts.csv')
        sbert = SentenceBertJapanese()
        chunk_size = len(all_prompt_df)//3 + 1
        prompt_emb = sbert.encode(all_prompt_df['prompt'][chunk_size*(split-1):chunk_size*(split)], batch_size=100)
        with open(f'../data/all_prompt_emb_{split}.pkl', 'wb') as f:
            pickle.dump(prompt_emb, f)

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup(self, ):
        dist.destroy_process_group()

    def encode_chunk(self, rank, world_size, all_prompts, chunk_size, model):
        self.setup(rank, world_size)

        # Determine the chunk of data this process will handle
        start_idx = rank * chunk_size
        end_idx = min((rank + 1) * chunk_size, len(all_prompts))
        chunk_prompts = all_prompts[start_idx:end_idx]
        model = SentenceBertJapanese()
        # Encode the prompts
        chunk_embeddings = model.encode(chunk_prompts, show_progress_bar=True, convert_to_tensor=True, device=f'cuda:{rank}')

        # Gather results from all processes
        gathered_embeddings = [torch.zeros_like(chunk_embeddings) for _ in range(world_size)]
        dist.all_gather(gathered_embeddings, chunk_embeddings)

        if rank == 0:
            # Combine gathered embeddings
            embeddings = torch.cat(gathered_embeddings, dim=0).cpu().numpy()
            with open('../data/all_prompt_emb.pkl', 'wb') as f:
                pickle.dump(embeddings, f)
            
    def get_embedding_parallel(self, world_size):    
        all_prompt_df = pd.read_csv('../data/all_prompts.csv')
        all_prompts = all_prompt_df['prompt'].tolist()

        # Initialize SentenceBertJapanese model
        model = SentenceTransformer()

        # world_size = torch.cuda.device_count()
        chunk_size = int(np.ceil(len(all_prompts) / world_size))

        torch.multiprocessing.spawn(self.encode_chunk, args=(world_size, all_prompts, chunk_size, model), nprocs=world_size, join=True)


    def get_embedding_entity(self):
        entity_df = pd.read_csv('../data/sakg_entity.csv')
        sbert = SentenceBertJapanese()
        entity_emb = sbert.encode(entity_df['entity'], batch_size=100)
        with open('../data/all_entity_emb.pkl', 'wb') as f:
            pickle.dump(entity_emb, f)
            

class SAKGSpot(SAKGBase):
    def __init__(self, load=False):
        #self.graph_path = '../data/kg/sakg_noun_adj.pkl'
        if load:
            self.load_graph()
        else:
            self.df_review = pd.read_pickle(
                "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
            )
            with open('../data/kg/pairs_adj.pkl', 'rb') as f:
                pairs_adj = pickle.load(f)
            with open('../data/kg/pairs_noun.pkl', 'rb') as f:
                pairs_noun = pickle.load(f)
            le_user = LabelEncoder()
            le_spot = LabelEncoder()
            self.df_review['user_id'] = le_user.fit_transform(self.df_review['url'])
            self.df_review['spot_id'] = le_spot.fit_transform(self.df_review['spot'])
            self.df_review['pairs_noun'] = pairs_noun
            self.df_review['pairs_adj'] = pairs_adj
            print('pairs_noun', pairs_noun[:5])
            print( 'pairs_adj', pairs_adj[:5])
            self.nlp = spacy.load('ja_ginza')
                
            
            #self.build_graph()
            #self.clean_up_edges()
            #self.visualize_graph(num_nodes=20)
            #self.print_graph_statistics()
            #self.save_graph()

    def build_graph(self, adj=True, noun=False, save=False, graph_path='../data/kg/sakg_noun_adj.pkl'):
        self.graph = nx.Graph()
        if noun:
            for i, row in tqdm(self.df_review.iterrows()):
                user_id, spot, pairs_noun, review = row['user_id'], row['spot'], row['pairs_noun'], row['review']
                user_node = 'user_' + str(user_id)
                # ノードカテゴリを追加
                self.graph.add_node(user_node, category='user')
                self.graph.add_node(spot, category='spot')
                for noun1,p1, noun2,p2 in pairs_noun:
                    self.graph.add_node(noun1, category='word')
                    self.graph.add_node(noun2, category='word')
                    self._add_edge_with_count(user_node, noun1, noun2)
                    self._add_edge_with_count(user_node, noun2, noun1)
                    self._add_edge_with_count(spot, noun1, noun2)
                    self._add_edge_with_count(spot, noun2, noun1)

        if adj:
            for i, row in tqdm(self.df_review.iterrows()):
                user_id, spot, pairs_adj, review = row['user_id'], row['spot'], row['pairs_adj'], row['review']
                user_node = 'user_' + str(user_id)
                # ノードカテゴリを追加
                self.graph.add_node(user_node, category='user')
                self.graph.add_node(spot, category='spot')
                for adj, noun in pairs_adj:
                    self.graph.add_node(noun, category='word')
                    self._add_edge_with_count(user_node, noun, adj)
                    self._add_edge_with_count(spot, noun, adj)

        self.clean_up_edges(count_thresh=3)
        if save:
            self.save_graph(graph_path)

class SAKG(SAKGBase):
    def __init__(self, load=False):
        #self.graph_path = '../data/kg/sakg_noun_adj.pkl'
        #self.graph_path = '../data/kg/sakg_unified.pkl'
        self.graph_path = '../data/kg/sakg_with_image.pkl'
        if load:
            self.load_graph()
        else:
            self.spot_df = pd.read_pickle(
                "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
            )
            with open('../data/kg/pairs_adj.pkl', 'rb') as f:
                spot_pairs_adj = pickle.load(f)
            with open('../data/kg/pairs_noun.pkl', 'rb') as f:
                spot_pairs_noun = pickle.load(f)
            #le_user = LabelEncoder()
            #le_spot = LabelEncoder()
            #self.spot_df['user_id'] = le_user.fit_transform(self.df_review['url'])
            #self.spot_df['spot_id'] = le_spot.fit_transform(self.df_review['spot'])
            self.spot_df['pairs_noun'] = spot_pairs_noun
            self.spot_df['pairs_adj'] = spot_pairs_adj
            print('pairs_noun', spot_pairs_noun[:5])
            print( 'pairs_adj', spot_pairs_adj[:5])
            self.nlp = spacy.load('ja_ginza')
                
            self.user_df = pd.read_csv(
                '../data/kg/user_review.csv'
            )
            with open('../data/kg/user_pairs_adj.pkl', 'rb') as f:
                user_pairs_adj = pickle.load(f)
            with open('../data/kg/pairs_noun.pkl', 'rb') as f:
                user_pairs_noun = pickle.load(f)
                
            print(len(user_pairs_adj), len(user_pairs_noun))
            self.user_df['pairs_noun'] = user_pairs_noun
            self.user_df['pairs_adj'] = user_pairs_adj

            #self.clean_up_edges()
            #self.visualize_graph(num_nodes=20)
            #self.print_graph_statistics()
            #self.save_graph()

    def clean_up_edges(self, count_thresholds={'image': 1, 'user': 2, 'spot': 3}):
        """
        ノードの種類に応じて異なる閾値でエッジをクリーンアップ
        
        Args:
            count_thresholds: ノードカテゴリごとの閾値を指定する辞書
        """
        edges_to_remove = []
        
        for node in self.graph.nodes:
            node_category = self.graph.nodes[node].get('category')
            if node_category in ['spot', 'user', 'image']:
                threshold = count_thresholds[node_category]
                
                for neighbor in self.graph.neighbors(node):
                    if node_category == 'image':
                        # imageノードの場合は直接weightを確認
                        weight = self.graph[node][neighbor].get('weight', 0)
                        if weight < threshold:
                            edges_to_remove.append((node, neighbor))
                    else:
                        # userまたはspotノードの場合はrelationsを確認
                        relations = self.graph[node][neighbor]['relations']
                        relations_to_remove = [rel for rel, count in relations.items() 
                                            if count < threshold]
                        
                        for rel in relations_to_remove:
                            del relations[rel]
                        
                        if not relations:
                            edges_to_remove.append((node, neighbor))
        
        self.graph.remove_edges_from(edges_to_remove)

    def build_graph(self, include_image=True, adj=True, noun=False, save=True, graph_path='../data/kg/sakg_noun_adj.pkl'):
        """
        ユーザーレビューと観光地レビューからグラフを構築
        
        Args:
            user_df: ユーザーレビューのDataFrame (user_id, reviewカラムが必要)
            spot_df: 観光地レビューのDataFrame (spot, reviewカラムが必要)
            adj: 形容詞-名詞のペアを使用するか
            noun: 名詞-名詞のペアを使用するか
        """
        self.graph = nx.Graph()

        test_reviews = set(list(pd.read_csv('../data/df_review_feature_eval.csv')['conversations'].values))
        # ユーザーレビューからエッジを追加
        if include_image:
            self.build_graph_with_images(excluded_labels={'人', '車', '観光地'})


        if self.user_df is not None:
            
            for i, row in tqdm(self.user_df.iterrows(), desc='Processing user reviews'):
                if row['review'] in test_reviews:continue
                user_id = str(row['user'])
                user_node = user_id
                self.graph.add_node(user_node, category='user')

                if noun:
                    for noun1, p1, noun2, p2 in row['pairs_noun']:
                        self.graph.add_node(noun1, category='word')
                        self.graph.add_node(noun2, category='word')
                        self._add_edge_with_count(user_node, noun1, noun2)
                        self._add_edge_with_count(user_node, noun2, noun1)

                if adj:
                    for adj, noun in row['pairs_adj']:
                        self.graph.add_node(noun, category='word')
                        self._add_edge_with_count(user_node, noun, adj)
                # if i==10:
                #     break

        # 観光地レビューからエッジを追加 
        if self.spot_df is not None:
            for i, row in tqdm(self.spot_df.iterrows(), desc='Processing spot reviews'):
                if row['review'] in test_reviews:continue
                spot = row['spot']
                self.graph.add_node(spot, category='spot')

                if noun:
                    for noun1, p1, noun2, p2 in row['pairs_noun']:
                        self.graph.add_node(noun1, category='word')
                        self.graph.add_node(noun2, category='word')
                        self._add_edge_with_count(spot, noun1, noun2)
                        self._add_edge_with_count(spot, noun2, noun1)

                if adj:
                    for adj, noun in row['pairs_adj']:
                        self.graph.add_node(noun, category='word')
                        self._add_edge_with_count(spot, noun, adj)
                # if i==10:
                #     break

        # self.clean_up_edges(count_thresholds={
        #         'image': 1,
        #         'user': 2,
        #         'spot': 3
        #     })

        if save:
            self.save_graph(self.graph_path)


    def build_graph_with_images(self, excluded_labels={'人', '観光地'}, save=False):
        """
        画像情報からグラフを構築する追加メソッド
        
        Args:
            image_paths: 画像パスのリスト
            img2caption: 画像パスをキー、キャプションを値とする辞書
            img2neighbors: 画像パスをキー、近傍画像パスのリストを値とする辞書
            detic_result_dir: Deticの結果が保存されているディレクトリ
        """
        
        captions = pd.read_csv('../data/image_caption/caption.csv')
        captions['image_suffix'] = captions['image_path'].apply(lambda x:x.split('/')[-1])
        img2caption = dict(zip(captions['image_suffix'], captions['caption']))
    
        image_text_df = pd.read_pickle('../data/image_review_pair_all_filter.pkl')
        image_text_df['image_suffix'] = image_text_df['image_path'].apply(lambda x:x.split('/')[-1])
        image_paths = list(image_text_df['image_suffix'].unique())
        

        with open('../data/image_retrieval/retrieve_result.pkl', 'rb') as f:
            retrieve_result = pickle.load(f)

        img2neighbors = defaultdict(list)
        for i, neighbors in enumerate(retrieve_result['indice']):
            for n in neighbors:
                # print(i, n)
                img2neighbors[image_paths[i]].append(image_paths[n])


        with open('../data/detic/img2labels.pkl', 'rb') as f:
            self.img2detic = pickle.load(f)

        for i,image_path in tqdm(enumerate(image_paths), desc='Processing images'):
            #if i==10:break
            image_node = image_path
            self.graph.add_node(image_node, category='image')
            
            # キャプションからの単語抽出
            if image_path in img2caption:
                # キャプションが存在する場合
                caption_words = extract_words_from_response(img2caption[image_path])
                caption_words = [w for w in caption_words if w not in excluded_labels]
                if caption_words:
                    # 最大2つまで抽出
                    for word in caption_words[:2]:
                        self.graph.add_node(word, category='word')
                        self._add_edge_with_count(image_node, word, relation=None)
            else:
                # 近傍画像のキャプションから抽出
                neighbor_words = []
                for neighbor in img2neighbors.get(image_path, []):
                    if neighbor in img2caption:
                        words = extract_words_from_response(img2caption[neighbor])
                        neighbor_words+=words
                        #neighbor_words.update([w for w in words if w not in excluded_labels])
                
                if neighbor_words:
                    # 最大1つ抽出
                    word = neighbor_words[0] #random.sample(neighbor_words, min(1, len(neighbor_words)))[0]
                    self.graph.add_node(word, category='word')
                    self._add_edge_with_count(image_node, word, relation=None)
            
            label = self.img2detic[image_path]
            if label:
                self.graph.add_node(label[0], category='word')
                self._add_edge_with_count(image_node, label[0], relation=None)
            # Deticの結果からの単語抽出
            # try:
            #     detic_path = os.path.join(detic_result_dir, f'{image_path}.json')
            #     if os.path.exists(detic_path):
            #         with open(detic_path) as f:
            #             result = json.load(f)
                    
            #         # スコアが0.6以上のラベルを抽出
            #         high_score_labels = [
            #             english_to_japanese[class_name] 
            #             for class_name, score in zip(result['class_names'], result['scores'])
            #             if score > 0.6 and english_to_japanese[class_name] not in excluded_labels
            #         ]
                    
            #         if high_score_labels:
            #             # 最大1つ抽出
            #             label = high_score_labels[0]
            #             self.graph.add_node(label, category='word')
            #             self._add_edge_with_count(image_node, label, relation=None)
                        
            # except Exception as e:
            #     print(f"Error processing Detic result for {image_path}: {e}")
            #     continue

        if save:
            with open('../data/kg/sakg_with_image.pkl', 'wb') as f:
                pickle.dump(self.graph, f)

    def clean_up_edges(self, count_thresholds={'image': 1, 'user': 2, 'spot': 3}):
        """
        ノードの種類に応じて異なる閾値でエッジをクリーンアップ
        
        Args:
            count_thresholds: ノードカテゴリごとの閾値を指定する辞書
        """
        edges_to_remove = []
        
        for node in self.graph.nodes:
            node_category = self.graph.nodes[node].get('category')
            if node_category in ['spot', 'user', 'image']:
                threshold = count_thresholds[node_category]
                
                for neighbor in self.graph.neighbors(node):
                    edge = self.graph[node][neighbor]
                    if 'weight' in edge:
                        # imageノードの場合は直接weightを確認
                        weight = self.graph[node][neighbor].get('weight', 0)
                        if weight < threshold:
                            edges_to_remove.append((node, neighbor))
                    elif 'relations' in edge:
                        # userまたはspotノードの場合はrelationsを確認
                        relations = self.graph[node][neighbor]['relations']
                        relations_to_remove = [rel for rel, count in relations.items() 
                                            if count < threshold]
                        
                        for rel in relations_to_remove:
                            del relations[rel]
                        
                        if not relations:
                            edges_to_remove.append((node, neighbor))
        
        self.graph.remove_edges_from(edges_to_remove)

    def _add_edge_with_count(self, node1, node2, relation=None):
        # エッジが既に存在する場合、relationごとのカウントを更新
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2, relations={}, weight=0)
        if relation is None:
            self.graph[node1][node2]['weight'] += 1
        else:
            if relation in self.graph[node1][node2]['relations']:
                self.graph[node1][node2]['relations'][relation] += 1
            else:
                self.graph[node1][node2]['relations'][relation] = 1
            


if __name__=='__main__':
    fire.Fire(SAKG)
