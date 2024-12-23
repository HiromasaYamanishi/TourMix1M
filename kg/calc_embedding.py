import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

# [Previous SAKGEmbeddingDataset class remains the same]
class SAKGEmbeddingDataset(Dataset):
    def __init__(self, graph, triples, entity2id, relation2id, negative_samples=5):
        self.graph = graph
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.negative_samples = negative_samples
        
        # Build triple dictionary for efficient negative sampling
        self.head_rel_to_tails = defaultdict(set)
        for head, rel, tail, _ in self.triples:
            self.head_rel_to_tails[(head, rel)].add(tail)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        head, rel, tail, count = self.triples[idx]
        neg_tails = []
        
        # More efficient negative sampling
        existing_tails = self.head_rel_to_tails[(head, rel)]
        num_entities = len(self.entity2id)
        
        for _ in range(self.negative_samples):
            neg_tail = np.random.randint(num_entities)
            retry_count = 0
            while neg_tail in existing_tails and retry_count < 100:
                neg_tail = np.random.randint(num_entities)
                retry_count += 1
            neg_tails.append(neg_tail)
            
        return {
            'head': head,
            'relation': rel,
            'tail': tail,
            'neg_tails': neg_tails,
            'count': count
        }

    
class SAKGEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, image_feature_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Count scaling factor for relations
        self.count_scale = nn.Parameter(torch.ones(num_relations))
        
        # Image projection layer
        self.image_projection = nn.Linear(image_feature_dim, embedding_dim)
        
        # SBERT projection layers
        self.entity_projection = nn.Linear(768, embedding_dim)  # SBERT dim -> embedding dim
        self.relation_projection = nn.Linear(768, embedding_dim)
        
        self.init_weights()
    
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.xavier_uniform_(self.entity_projection.weight)
        nn.init.xavier_uniform_(self.relation_projection.weight)
    
    def get_scaled_relation_embedding(self, relation_ids, counts=None):
        rel_emb = self.relation_embeddings(relation_ids)
        if counts is not None:
            scale = self.count_scale[relation_ids].unsqueeze(-1)
            rel_emb = rel_emb * (1 + torch.log1p(counts.unsqueeze(-1) * scale))
        return rel_emb
    
    def forward(self, heads, relations, tails, neg_tails, counts=None):
        # Get embeddings
        head_emb = self.entity_embeddings(heads)
        tail_emb = self.entity_embeddings(tails)
        neg_tail_emb = self.entity_embeddings(neg_tails)
        relation_emb = self.get_scaled_relation_embedding(relations, counts)
        
        # TransE score calculation
        pos_score = torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=-1).unsqueeze(1)
        neg_score = torch.norm(head_emb.unsqueeze(1) + relation_emb.unsqueeze(1) - neg_tail_emb, p=2, dim=-1)
        
        return pos_score, neg_score
    
    def contrastive_loss(self, emb1, emb2, temperature=0.07):
        # Normalize embeddings
        emb1_norm = F.normalize(emb1, dim=-1)
        emb2_norm = F.normalize(emb2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(emb1_norm, emb2_norm.transpose(0, 1)) / temperature
        
        # Contrastive loss
        labels = torch.arange(len(emb1)).to(emb1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    # [Rest of SAKGEmbedding class remains the same]

class SAKGEmbeddingTrainer:
    def __init__(self, graph, image_features_dict, embedding_dim=128, lr=0.0005, 
                 negative_samples=5, batch_size=64, use_sbert_contrast=False, 
                 sbert_weight=0.1, val_ratio=0.1):
        self.graph = graph
        
        # Create entity and relation vocabularies
        self.entity2id = {node: idx for idx, node in enumerate(graph.nodes())}
        self.id2entity = {idx: node for node, idx in self.entity2id.items()}
        
        #print('entity2id', list(self.entity2id.keys())[:100])

        #print('image feature dict', list(image_features_dict.keys())[:100])
        converted_image_features = {}
        for filename, features in image_features_dict.items():
            # もしファイル名がグラフのノードとして存在する場合
            if filename in self.entity2id:
                node_id = self.entity2id[filename]
                converted_image_features[node_id] = features
        
        self.image_features_dict = converted_image_features
        # print('image feature dict', list(converted_image_features.keys())[:100])
        # exit()
        image_feature_dim = next(iter(image_features_dict.values())).shape[0]
        # Create relation vocabulary
        self.relation_counts = defaultdict(int)
        self.relation2id = {}
        rel_idx = 0
        for _, _, edge_data in graph.edges(data=True):
            if 'relations' in edge_data:
                for relation in edge_data['relations'].keys():
                    if relation not in self.relation2id:
                        self.relation2id[relation] = rel_idx
                        rel_idx += 1
                    self.relation_counts[relation] += edge_data['relations'][relation]
        
        # Add special relation for image nodes
        self.relation2id['image_relation'] = len(self.relation2id)
        
        # Create all triples
        all_triples = []
        for i,(head, tail, edge_data) in enumerate(graph.edges(data=True)):
            # print(head, tail)
            # if i<100:
            #     print(head.get('category', None))
            #     print(tail.get('category', None))
            # if head.get('category', None) == 'image':
            #     print('image head', head)
            # if tail.get('category', None) == 'image':
            #     print('tail head', tail)
            head_id = self.entity2id[head]
            tail_id = self.entity2id[tail]
            
            if 'weight' in edge_data:  # For image nodes
                rel_id = self.relation2id['image_relation']
                all_triples.append((head_id, rel_id, tail_id, edge_data['weight']))
            elif 'relations' in edge_data:
                for relation, count in edge_data['relations'].items():
                    rel_id = self.relation2id[relation]
                    all_triples.append((head_id, rel_id, tail_id, count))
        
        # Split triples into train and validation sets
        np.random.shuffle(all_triples)
        val_size = int(len(all_triples) * val_ratio)
        train_triples = all_triples[val_size:]
        val_triples = all_triples[:val_size]
        
        # Create train and validation datasets
        self.train_dataset = SAKGEmbeddingDataset(graph, train_triples, self.entity2id, self.relation2id, negative_samples)
        self.val_dataset = SAKGEmbeddingDataset(graph, val_triples, self.entity2id, self.relation2id, negative_samples)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.model = SAKGEmbedding(
            num_entities=len(self.entity2id),
            num_relations=len(self.relation2id),
            embedding_dim=embedding_dim,
            image_feature_dim=image_feature_dim
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.initial_lr = lr
        # 学習率減衰のパラメータ
        self.lr_decay_factor = 0.5  # 学習率を半分にする
        self.lr_decay_patience = 3   # 3回改善がなければ減衰
        self.no_improvement_count = 0
        self.best_metric = 0

        self.use_sbert_contrast = use_sbert_contrast
        self.sbert_weight = sbert_weight
        
        # Create mapping from entity IDs to image IDs for image nodes
        self.entity_to_image_id = {}
        for node, idx in self.entity2id.items():
            if isinstance(node, str) and self.graph.nodes[node].get('category') == 'image':
                self.entity_to_image_id[idx] = node
        #print('image num', self.entity_to_image_id)
        if use_sbert_contrast:
            self._initialize_sbert_embeddings(graph)

    def evaluate(self, device='cuda', num_negative_samples=100, show_samples=True, num_samples=20):
        """Evaluate the model using random negative samples instead of all entities"""
        self.model.eval()
        total_samples = 0
        hits_1 = 0
        hits_5 = 0
        hits_10 = 0
        mrr = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                heads = batch['head'].to(device)
                relations = batch['relation'].to(device)
                tails = batch['tail'].to(device)
                counts = batch['count'].float().to(device)
                batch_size = len(heads)
                
                # Generate random negative samples
                random_entities = torch.randint(
                    0, self.model.num_entities, 
                    (batch_size, num_negative_samples),
                    device=device
                )
                # Add the correct tail to the candidates
                candidates = torch.cat([tails.unsqueeze(1), random_entities], dim=1)  # [batch_size, num_negative_samples + 1]
                
                # Get embeddings
                head_emb = self.model.entity_embeddings(heads).unsqueeze(1)  # [batch_size, 1, dim]
                rel_emb = self.model.get_scaled_relation_embedding(relations, counts).unsqueeze(1)  # [batch_size, 1, dim]
                cand_emb = self.model.entity_embeddings(candidates)  # [batch_size, num_negative_samples + 1, dim]
                
                # Calculate scores
                scores = torch.norm(
                    head_emb + rel_emb - cand_emb,
                    p=2,
                    dim=-1
                )  # [batch_size, num_negative_samples + 1]
                
                # Get rankings (correct tail is at index 0)
                rankings = (scores <= scores[:, 0].unsqueeze(1)).sum(1)
                
                # Calculate metrics
                hits_1 += (rankings <= 1).sum().item()
                hits_5 += (rankings <= 5).sum().item()
                hits_10 += (rankings <= 10).sum().item()
                mrr += (1.0 / rankings.float()).sum().item()
                total_samples += batch_size
        
        metrics = {
            'Hits@1': hits_1 / total_samples * 100,
            'Hits@5': hits_5 / total_samples * 100,
            'Hits@10': hits_10 / total_samples * 100,
            'MRR': mrr / total_samples
        }
        
        if show_samples:
            print("\n=== Sampling Examples ===")
            # 画像からのサンプリング
            image_triplets = self.sample_triplets_from_embeddings(
                num_samples=num_samples, 
                source_type='image',
            min_connections=200,
                device=device
            )
            print("\nSamples from Images:")
            self.print_sampled_triplets(image_triplets)
            
            # スポットからのサンプリング
            spot_triplets = self.sample_triplets_from_embeddings(
                num_samples=num_samples, 
                source_type='spot',
                    min_connections=200,
                device=device
            )
            print("\nSamples from Spots:")
            self.print_sampled_triplets(spot_triplets)
        
        self.model.train()
        return metrics
    
    def get_node_importance(self, node_ids, min_connections=5):
        """
        ノードの重要性（接続数）に基づく重みを計算
        
        Args:
            node_ids: ノードIDのリスト
            min_connections: 最小接続数の閾値
        
        Returns:
            weights: 各ノードの重み（接続数が少ないノードは0）
        """
        weights = []
        for node_id in node_ids:
            node = self.id2entity[node_id.item()]
            connection_count = 0
            
            # エッジの重みの合計を計算
            for _, _, edge_data in self.graph.edges(node, data=True):
                if 'weight' in edge_data:
                    connection_count += edge_data['weight']
                elif 'relations' in edge_data:
                    connection_count += sum(edge_data['relations'].values())
            
            # 最小接続数以上の場合のみ重みを設定
            weight = connection_count if connection_count >= min_connections else 0
            weights.append(weight)
        
        weights = torch.tensor(weights, dtype=torch.float)
        return weights

    def weighted_sampling(self, node_ids, weights, num_samples=1):
        """
        重み付きサンプリングを実行
        
        Args:
            node_ids: ノードIDのテンソル
            weights: 重みのテンソル
            num_samples: サンプリング数
        
        Returns:
            sampled_indices: サンプリングされたインデックス
        """
        if (weights == 0).all():
            # すべての重みが0の場合はランダムサンプリング
            return torch.randint(0, len(node_ids), (num_samples,))
        
        # 重みを確率に変換
        probs = weights / weights.sum()
        
        # 重み付きサンプリングを実行
        sampled_indices = torch.multinomial(probs, num_samples, replacement=True)
        return sampled_indices

    def sample_triplets_from_embeddings(self, num_samples=5, source_type='both', device='cuda', min_connections=5):
        """
        埋め込みに基づいてトリプレットをサンプリング（重要なノードを優先）
        
        Args:
            num_samples: サンプリング数
            source_type: 'image', 'spot', 'both' のいずれか
            device: 使用デバイス
            min_connections: 最小接続数の閾値
        """
        self.model.eval()
        with torch.no_grad():
            # エンティティの埋め込みを取得
            entity_embeddings = self.model.entity_embeddings.weight
            relation_embeddings = self.model.relation_embeddings.weight
            
            # ノードの分類とIDの取得
            image_nodes = []
            spot_nodes = []
            word_nodes = []
            for node_id in range(len(self.id2entity)):
                node = self.id2entity[node_id]
                node_category = self.graph.nodes[node].get('category')
                if isinstance(node, str) and node_category == 'image':
                    image_nodes.append(node_id)
                elif isinstance(node, str) and node_category == 'word':
                    word_nodes.append(node_id)
                else:
                    spot_nodes.append(node_id)
            
            # 各ノードタイプの重要度を計算
            image_nodes = torch.tensor(image_nodes)
            spot_nodes = torch.tensor(spot_nodes)
            word_nodes = torch.tensor(word_nodes)
            
            image_weights = self.get_node_importance(image_nodes, min_connections)
            spot_weights = self.get_node_importance(spot_nodes, min_connections)
            word_weights = self.get_node_importance(word_nodes, min_connections)
            
            # デバイスに移動
            image_nodes = image_nodes.to(device)
            spot_nodes = spot_nodes.to(device)
            word_nodes = word_nodes.to(device)
            image_weights = image_weights.to(device)
            spot_weights = spot_weights.to(device)
            word_weights = word_weights.to(device)
            
            sampled_triplets = []
            samples_per_type = num_samples
            if source_type == 'both':
                samples_per_type = num_samples // 2

            # 画像からのサンプリング
            if source_type in ['image', 'both']:
                for _ in range(samples_per_type):
                    # 重み付きサンプリングで画像ノードを選択
                    image_idx = self.weighted_sampling(image_nodes, image_weights, 1)[0]
                    image_id = image_nodes[image_idx].item()
                    image_node = self.id2entity[image_id]
                    
                    # 画像ノードから接続している実際のエッジを取得
                    connected_spots = []
                    connected_relations = []
                    connected_weights = []
                    
                    for _, spot, edge_data in self.graph.edges(image_node, data=True):
                        if 'weight' in edge_data:
                            connected_spots.append(self.entity2id[spot])
                            connected_relations.append(self.relation2id['image_relation'])
                            connected_weights.append(edge_data['weight'])
                        elif 'relations' in edge_data:
                            for relation, count in edge_data['relations'].items():
                                connected_spots.append(self.entity2id[spot])
                                connected_relations.append(self.relation2id[relation])
                                connected_weights.append(count)
                    
                    if not connected_spots:
                        continue
                    
                    # 接続されているスポットの埋め込みを取得
                    connected_spots = torch.tensor(connected_spots).to(device)
                    spot_embs = entity_embeddings[connected_spots]
                    image_emb = entity_embeddings[image_id]
                    
                    # 類似度計算
                    similarities = torch.matmul(spot_embs, image_emb)
                    most_similar_indices = torch.topk(similarities, k=min(3, len(connected_spots))).indices
                    
                    for idx in most_similar_indices:
                        spot_id = connected_spots[idx].item()
                        relation_id = connected_relations[idx]
                        weight = connected_weights[idx]
                        
                        sampled_triplets.append({
                            'source_type': 'image',
                            'source': image_node,
                            'relation': list(self.relation2id.keys())[relation_id],
                            'target': self.id2entity[spot_id],
                            'weight': weight,
                            'similarity_score': similarities[idx].item(),
                            'target_category': 'spot'
                        })

            # スポットからのサンプリング
            if source_type in ['spot', 'both']:
                for _ in range(samples_per_type):
                    # 重み付きサンプリングでスポットノードを選択
                    spot_idx = self.weighted_sampling(spot_nodes, spot_weights, 1)[0]
                    spot_id = spot_nodes[spot_idx].item()
                    spot_node = self.id2entity[spot_id]
                    spot_emb = entity_embeddings[spot_id]
                    
                    # Image ノードとの類似度計算
                    image_embs = entity_embeddings[image_nodes]
                    image_similarities = torch.matmul(image_embs, spot_emb)
                    top_image_indices = torch.topk(image_similarities, k=min(3, len(image_nodes))).indices
                    
                    # Word ノードとの類似度計算
                    word_embs = entity_embeddings[word_nodes]
                    word_similarities = torch.matmul(word_embs, spot_emb)
                    top_word_indices = torch.topk(word_similarities, k=min(3, len(word_nodes))).indices
                    
                    # 実際のエッジ情報の取得
                    edge_info = {}
                    for _, target, edge_data in self.graph.edges(spot_node, data=True):
                        target_id = self.entity2id[target]
                        if 'weight' in edge_data:
                            edge_info[target_id] = {
                                'relation': 'image_relation',
                                'weight': edge_data['weight']
                            }
                        elif 'relations' in edge_data:
                            for relation, count in edge_data['relations'].items():
                                edge_info[target_id] = {
                                    'relation': relation,
                                    'weight': count
                                }
                    
                    # Image トリプレットの追加
                    for idx in top_image_indices:
                        image_id = image_nodes[idx].item()
                        image_node = self.id2entity[image_id]
                        
                        # エッジ情報の取得（なければデフォルト値）
                        edge_data = edge_info.get(image_id, {
                            'relation': 'image_relation',
                            'weight': 1.0
                        })
                        
                        sampled_triplets.append({
                            'source_type': 'spot',
                            'source': spot_node,
                            'relation': edge_data['relation'],
                            'target': image_node,
                            'weight': edge_data['weight'],
                            'similarity_score': image_similarities[idx].item(),
                            'target_category': 'image'
                        })
                    
                    # Word トリプレットの追加
                    for idx in top_word_indices:
                        word_id = word_nodes[idx].item()
                        word_node = self.id2entity[word_id]
                        
                        # エッジ情報の取得（なければデフォルト値）
                        edge_data = edge_info.get(word_id, {
                            'relation': 'appears_with',
                            'weight': 1.0
                        })
                        
                        sampled_triplets.append({
                            'source_type': 'spot',
                            'source': spot_node,
                            'relation': edge_data['relation'],
                            'target': word_node,
                            'weight': edge_data['weight'],
                            'similarity_score': word_similarities[idx].item(),
                            'target_category': 'word'
                        })
                
            return sampled_triplets

    def print_sampled_triplets(self, triplets):
        """サンプリングされたトリプレットを表示"""
        print("\nSampled Triplets based on Embeddings:")
        print("-" * 80)
        
        # ソースタイプごとにグループ化して表示
        for source_type in ['image', 'spot']:
            type_triplets = [t for t in triplets if t['source_type'] == source_type]
            if type_triplets:
                print(f"\n=== Sampled from {source_type.upper()} ===")
                
                # ターゲットカテゴリごとにグループ化
                if source_type == 'spot':
                    for category in ['image', 'word']:
                        category_triplets = [t for t in type_triplets if t['target_category'] == category]
                        if category_triplets:
                            print(f"\n--- {category.upper()} Targets ---")
                            for i, triplet in enumerate(category_triplets, 1):
                                print(f"\nTriplet {i}:")
                                print(f"Source (Spot): {triplet['source']}")
                                print(f"Relation: {triplet['relation']}")
                                print(f"Target ({category}): {triplet['target']}")
                                print(f"Weight: {triplet['weight']:.3f}")
                                print(f"Similarity Score: {triplet['similarity_score']:.3f}")
                else:
                    for i, triplet in enumerate(type_triplets, 1):
                        print(f"\nTriplet {i}:")
                        print(f"Source: {triplet['source']}")
                        print(f"Relation: {triplet['relation']}")
                        print(f"Target: {triplet['target']}")
                        print(f"Weight: {triplet['weight']:.3f}")
                        print(f"Similarity Score: {triplet['similarity_score']:.3f}")

    def _get_batch_image_features(self, entity_indices, device):
        """Get image features for a batch of entity indices"""
        batch_features = []
        valid_indices = []
        #print('entity', entity_indices)
        #print('idx', entity_indices)
        for idx in entity_indices:
            #print('entity to image', idx, list(self.image_features_dict.keys())[:10])

            if idx.item() in self.image_features_dict:
                batch_features.append(self.image_features_dict[idx.item()])
                valid_indices.append(idx)
            # if idx in self.entity_to_image_id:
            #     image_id = self.entity_to_image_id[idx]
            #     if image_id in self.image_features_dict:
            #         batch_features.append(self.image_features_dict[image_id])
            #         valid_indices.append(idx)
        
        if not batch_features:
            return None, None
        return (torch.tensor(batch_features, device=device), 
                torch.tensor(valid_indices, device=device))
    
    def train(self, num_epochs, eval_steps=1000, device='cuda'):
        self.model = self.model.to(device)
        best_mrr = 0
        steps = 0
        
        if self.use_sbert_contrast:
            self.entity_sbert_emb = self.entity_sbert_emb.to(device)
            self.relation_sbert_emb = self.relation_sbert_emb.to(device)
        
        for epoch in range(num_epochs):
            total_loss = 0
            epoch_steps = 0
            
            for batch in tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                self.optimizer.zero_grad()
                
                # Move batch to device
                heads = batch['head'].to(device)
                relations = batch['relation'].to(device)
                tails = batch['tail'].to(device)
                neg_tails = torch.stack(batch['neg_tails']).t().to(device)
                counts = batch['count'].float().to(device)
                
                # TransE loss
                pos_score, neg_score = self.model(heads, relations, tails, neg_tails, counts)
                transe_loss = torch.mean(torch.max(pos_score - neg_score + 1, torch.zeros_like(pos_score)))
                
                loss = transe_loss
                
                # Add SBERT contrastive loss if enabled
                if self.use_sbert_contrast:
                    entity_emb = self.model.entity_embeddings.weight
                    entity_proj = self.model.entity_projection(self.entity_sbert_emb)
                    entity_contrast_loss = self.model.contrastive_loss(entity_emb, entity_proj)
                    
                    relation_emb = self.model.relation_embeddings.weight
                    relation_proj = self.model.relation_projection(self.relation_sbert_emb)
                    relation_contrast_loss = self.model.contrastive_loss(relation_emb, relation_proj)
                    
                    loss += self.sbert_weight * (entity_contrast_loss + relation_contrast_loss)
                    
                # Add image contrastive loss for image nodes
                batch_entity_ids = torch.cat([heads, tails, neg_tails.reshape(-1)])
                image_features, valid_indices = self._get_batch_image_features(batch_entity_ids, device)
                
                if image_features is not None:
                    image_emb = self.model.entity_embeddings(valid_indices)
                    image_proj = self.model.image_projection(image_features)
                    image_contrast_loss = self.model.contrastive_loss(image_emb, image_proj)
                    loss += 0.05 * image_contrast_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                steps += 1
                epoch_steps += 1
                
                # Print loss every 10 steps
                if steps % 50 == 0:
                    print(f'Step {steps}, Loss: {loss.item():.4f}')
                
                # Evaluate every eval_steps
                if steps % eval_steps == 0:
                    metrics = self.evaluate(device=device)
                    print(f"\nEvaluation at step {steps}:")
                    for metric_name, value in metrics.items():
                        print(f"{metric_name}: {value:.2f}")
                    
                    # 学習率の調整
                    if metrics['MRR'] > self.best_metric:
                        self.best_metric = metrics['MRR']
                        self.no_improvement_count = 0
                        # Save best model
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_mrr': best_mrr,
                        }, 'best_model.pt')
                    else:
                        self.no_improvement_count += 1
                        
                        if self.no_improvement_count >= self.lr_decay_patience:
                            # 学習率を減衰
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] *= self.lr_decay_factor
                            current_lr = self.optimizer.param_groups[0]['lr']
                            print(f'Reducing learning rate to {current_lr}')
                            self.no_improvement_count = 0

            
            avg_loss = total_loss / epoch_steps
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
            
            # Evaluate at the end of each epoch
            metrics = self.evaluate(device=device)
            print(f"\nEvaluation at epoch {epoch+1}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.2f}")

def filter_low_frequency_entities(graph, min_weight=5):
    """
    categoryがwordのノードのうち、重みの総和が一定値未満のエンティティを除去する
    
    Args:
        graph: NetworkXグラフ
        min_weight: 最小重み閾値
    
    Returns:
        filtered_graph: フィルタリング後のグラフ
    """
    filtered_graph = graph.copy()
    nodes_to_remove = []
    
    # 各カテゴリーのノード数をカウント
    category_counts = {'word': 0, 'removed_word': 0}
    for node in graph.nodes():
        if graph.nodes[node].get('category') == 'word':
            category_counts['word'] += 1
    
    for node in graph.nodes():
        # wordカテゴリーのノードのみを処理
        if graph.nodes[node].get('category') == 'word':
            total_weight = 0
            
            # ノードから出るエッジの重みを集計
            for _, _, edge_data in graph.edges(node, data=True):
                if 'weight' in edge_data:
                    total_weight += edge_data['weight']
                elif 'relations' in edge_data:
                    total_weight += sum(edge_data['relations'].values())
            
            # 重みが閾値未満なら除去対象に追加
            if total_weight < min_weight:
                nodes_to_remove.append(node)
                category_counts['removed_word'] += 1
    
    # 該当ノードを除去
    filtered_graph.remove_nodes_from(nodes_to_remove)
    
    # 結果の表示
    print(f"Original word nodes: {category_counts['word']}")
    print(f"Removed {category_counts['removed_word']} word nodes with total weight < {min_weight}")
    print(f"Remaining word nodes: {category_counts['word'] - category_counts['removed_word']}")
    
    return filtered_graph

if __name__=='__main__':
    image_text_df = pd.read_pickle('/home/yamanishi/project/airport/src/analysis/LLaVA/data/image_review_pair_all_filter.pkl')
    print(len(image_text_df))
    image_paths = list(image_text_df['image_path'].unique())
    image_paths = [i.split('/')[-1] for i in image_paths]
    all_vectors = np.concatenate([np.load(f'../data/image_retrieval/clip_image_embeddings_part{suffix}.npy') for suffix in range(4)])

    assert len(image_paths) == len(all_vectors)
    image_features_dict = dict(zip(image_paths, all_vectors))
    with open('../data/kg/sakg_with_image.pkl', 'rb') as f:
        graph = pickle.load(f)

    for image_path in image_paths:
        spot_name = image_path.split('_')[0]
        if graph.has_edge(image_path, spot_name):
            graph.add_edge(image_path, spot_name, relations={}, weight=0)
            graph[image_path][spot_name]['weight']+=1

    graph = filter_low_frequency_entities(graph)
    # トレーナーの初期化
    trainer = SAKGEmbeddingTrainer(
        graph=graph,
        image_features_dict=image_features_dict,
        embedding_dim=128,
        lr=0.0005,
        use_sbert_contrast=False,
        sbert_weight=0.1
    )

    # 学習実行
    trainer.train(num_epochs=50)