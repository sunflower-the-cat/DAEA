import numpy as np
import networkx as nx
import math
from tqdm import tqdm, trange
from collections import defaultdict
import pdb
import os
from laserembeddings import Laser
import re
import torch

def read_doublet(file):
    def ext_url(url):
        return url.strip().split("/")[-1]
    with open(file) as f:
        lines = f.readlines()
    doublets = []
    for line in lines:
        items = line.strip().split('\t')
        if len(items) < 2: 
            items.append("_")
        doublets.append((items[0], ext_url(items[1])))
    return doublets
def read_attr(file):
    def ext_url(url):
        return url.strip("<>").split("/")[-1]         

    with open(file) as f:
        lines = f.readlines()
    triples = []
    attr_triples = []
    for line in lines:
        items = line.strip().split(" ")
        entity = ext_url(items[0])
        attribute = ext_url(items[1])
        value = " ".join(items[2:-1]) 
        attr_triples.append((entity, attribute, value))
    return attr_triples 
def build_entity_map(args, language_id):
    ent_ids = read_doublet( os.path.join(args.DBP15K_clean_dir, args.subset, 'ent_ids_%s' % str(language_id)) )
    ent_names = read_doublet( os.path.join(args.DBP15K_clean_dir, args.subset,'id_features_' + str(language_id)) )
    assert len(ent_ids) == len(ent_names)
    id2ent = {}
    dbpid2entid = {}
    for i, (ent_id_tup, ent_name_tup) in enumerate(zip(ent_ids, ent_names)):
        ent_id_a, end_label = ent_id_tup
        ent_id_b, ent_name = ent_name_tup
        assert ent_id_a == ent_id_b
        id2ent[i] = {"dbpid": int(ent_id_a), "label": end_label, "name": ent_name}
        dbpid2entid[int(ent_id_a)] = i
    ent2id = { v["label"]: k for k, v in id2ent.items()}
    return id2ent, ent2id, dbpid2entid
def build_relation_map(args, language_id):
    rel_ids = read_doublet( os.path.join(args.DBP15K_clean_dir, args.subset,'rel_ids_' + str(language_id)) )
    id2rel = {}
    dbpid2relid = {}
    for i, (rel_id, rel_label) in enumerate(rel_ids):
        id2rel[i] = { "dbpid": int(rel_id), "label": rel_label }
        dbpid2relid[int(rel_id)] = i
    rel2id = { v["label"]:  k for k, v in id2rel.items()}
    return id2rel, rel2id, dbpid2relid
def build_attribute_map(args, language_name):
    raw_attr_triples = read_attr( os.path.join(args.DBP15K_dir, args.subset, "%s_att_triples" % language_name) )
    all_attributes = list(set([ a for e, a, v in raw_attr_triples]))
    id2attr = {}
    for i, attr in enumerate(all_attributes):
        id2attr[i] = attr
    attr2id = { v: k for k, v in id2attr.items()}
    return id2attr, attr2id, raw_attr_triples           

class Graph:
    def __init__(self, args, language_id, language_name): 
        self.language_name = language_name
        self.id2ent, self.ent2id, self.dbpid2entid = build_entity_map(args, language_id)
        self.id2rel, self.rel2id, self.dbpid2relid = build_relation_map(args, language_id)
        self.id2attr, self.attr2id, self.raw_attr_triples = build_attribute_map(args, language_name)
        self.rel_triples = self.read_triple(args, language_id)
        self.attr_triples = self.build_attr_triples()
    def read_triple(self, args, language_id):
        file = os.path.join(args.DBP15K_clean_dir, args.subset, "triples_%s" % str(language_id))     
        with open(file) as f:
            lines = f.readlines()
        triples = []
        for line in lines:
            items = line.strip().split('\t')
            triples.append(( self.dbpid2entid[int(items[0])], self.dbpid2relid[int(items[1])], self.dbpid2entid[int(items[2])] ) )
        return triples          
    def build_attr_triples(self):
        # map raw_attr_triples to entity and attribute ids
        return [(self.ent2id[e], self.attr2id[a])  for e, a, v in self.raw_attr_triples]   

def ent_adj_matrix(graph, is_self_mapping=True):
    matrix_shape = [len(graph.id2ent), len(graph.id2ent)]
    coord_set = set()
    degree_count = defaultdict(int)    
    i_matrix = []
    v_matrix = []

    if is_self_mapping:
        for i in range(matrix_shape[0]):
            coord = (i,i)
            i_matrix.append(coord)
            coord_set.add(coord)
            degree_count[coord[0]] += 1

    skipped_triple = 0
    for h, r, t in graph.rel_triples:
        if h not in graph.id2ent or t not in graph.id2ent or r not in graph.id2rel:
            skipped_triple += 1
            continue
        coord = (h, t)
        if coord not in coord_set:
            i_matrix.append(coord)
            degree_count[coord[0]] += 1

        coord_set.add(coord)

    for i, coord in enumerate(i_matrix):
        # symmtric normalize
        v_matrix.append(1 / \
                        ( ( 1 if degree_count[coord[0]] == 0 else math.sqrt(degree_count[coord[0]]) ) * ( 1 if degree_count[coord[1]] == 0 else math.sqrt(degree_count[coord[1]]) ) )
                       )        

    i_matrix_two_rows = [[],[]]
    for x, y in i_matrix:
        i_matrix_two_rows[0].append(x)
        i_matrix_two_rows[1].append(y)    

    adj_sparse = {"i": i_matrix_two_rows, "v": v_matrix, "shape": matrix_shape}  
    return adj_sparse

def rel_adj_matrix(graph, is_self_mapping=True):
    matrix_shape = [len(graph.id2ent), len(graph.id2rel)]
    coord_set = set()
    degree_count = defaultdict(int)    
    i_matrix = []
    v_matrix = []

    skipped_triple = 0
    for h, r, t in graph.rel_triples:
        if h not in graph.id2ent or t not in graph.id2ent or r not in graph.id2rel:
            skipped_triple += 1
            continue
        coord = (h, r)
        if coord not in coord_set:
            i_matrix.append(coord)
            degree_count[coord[0]] += 1

        coord_set.add(coord)

    for i, coord in enumerate(i_matrix):
        # mean normalize
        v_matrix.append( 1 / ( 1 if degree_count[coord[0]] == 0 else degree_count[coord[0]] ) )        

    i_matrix_two_rows = [[],[]]
    for x, y in i_matrix:
        i_matrix_two_rows[0].append(x)
        i_matrix_two_rows[1].append(y)    

    adj_sparse = {"i": i_matrix_two_rows, "v": v_matrix, "shape": matrix_shape}  
    return adj_sparse

def attr_adj_matrix(graph, is_self_mapping=True):
    matrix_shape = [len(graph.id2ent), len(graph.id2attr)]
    coord_set = set()
    degree_count = defaultdict(int)    
    i_matrix = []
    v_matrix = []

    skipped_triple = 0
    for e, a in graph.attr_triples:
        if e not in graph.id2ent or a not in graph.id2attr:
            skipped_triple += 1
            continue
        coord = (e, a)
        if coord not in coord_set:
            i_matrix.append(coord)
            degree_count[coord[0]] += 1

        coord_set.add(coord)

    for i, coord in enumerate(i_matrix):
        # mean normalize
        v_matrix.append( 1 / ( 1 if degree_count[coord[0]] == 0 else degree_count[coord[0]] ) )       

    i_matrix_two_rows = [[],[]]
    for x, y in i_matrix:
        i_matrix_two_rows[0].append(x)
        i_matrix_two_rows[1].append(y)    

    adj_sparse = {"i": i_matrix_two_rows, "v": v_matrix, "shape": matrix_shape}  
    return adj_sparse

def build_adj_matrices(graph):
    ent_matrix = ent_adj_matrix(graph)
    rel_matrix = rel_adj_matrix(graph)
    attr_matrix = attr_adj_matrix(graph)
    matriecs = {
        'ent': ent_matrix,
        'rel': rel_matrix,
        'attr': attr_matrix
    }
    return matriecs

def is_en(text, default_lang):
    p = re.compile('([a-zA-Z0-9]+)')
    result = p.findall(text)
    en_length = len(''.join(result))
    text_length = len(text)
    return 'en' if en_length / text_length > 0.5 else default_lang    
    
def laser_embeddings(args, graph, language_name):
    save_dir = "saved_data/laser_embeddings/%s" % args.subset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file = os.path.join(save_dir, "%s_embeddings.pth" % language_name)
    if os.path.exists(file):
        all_embeddings = torch.load(file)
    else:
        ents = [ graph.id2ent[i]['name'] for i in range(len(graph.id2ent))]
        rels = [ graph.id2rel[i]['label'] for i in range(len(graph.id2rel))]
        rels_lang = [ is_en(rel, language_name) for rel in rels]
        attrs = [ graph.id2attr[i] for i in range(len(graph.id2attr))]
        attrs_lang = [ is_en(attr, language_name) for attr in attrs]
        
        laser = Laser()
        print("start creating %s embeedings..." % language_name)
        ent_embeddings = laser.embed_sentences(ents, lang="en")
        print("ents done...")
        rel_embeddings = laser.embed_sentences(rels, lang=rels_lang)
        print("rels done...")
        attr_embeddings = laser.embed_sentences(attrs, lang=attrs_lang)
        print("attrs done...")
        all_embeddings = {
            'ents': ent_embeddings,
            'rels': rel_embeddings,
            'attrs': attr_embeddings
        }        
        
        torch.save(all_embeddings, os.path.join(save_dir, "%s_embeddings.pth" % language_name))
        print("embeddings saved.")
        
    return all_embeddings
def train_test_sets(args, g1, g2):
    train_set = read_doublet( os.path.join(args.DBP15K_clean_dir, args.subset, 'ref_ent_ids') )
    train_set = [ (g1.dbpid2entid[int(ent_1)], g2.dbpid2entid[int(ent_2)]) for ent_1, ent_2 in train_set]
    test_set = read_doublet( os.path.join(args.DBP15K_clean_dir, args.subset, 'sup_ent_ids') )
    test_set = [ (g1.dbpid2entid[int(ent_1)], g2.dbpid2entid[int(ent_2)]) for ent_1, ent_2 in test_set]
    return train_set, test_set
    
def generate_data(args):
    data = {}
    graphs = []
    for language_id, language_name in enumerate(args.subset.split("_")):
        language_id += 1
        g = Graph(args, language_id, language_name)
        adj_matrices = build_adj_matrices(g)
        embeddings = laser_embeddings(args, g, language_name)
        data[language_name] = (adj_matrices, embeddings)
        graphs.append(g)
    train, test = train_test_sets(args, *graphs)
    data['train_test'] = (train, test)
    return data

def generate_walks(args):
    source, target = args.subset.split("_")
    source_g, target_g = Graph(args, 1, source), Graph(args, 2, target)
    train, test = train_test_sets(args, source_g, target_g)

    save_dir = "saved_walks/%s" % args.subset
    walks_file = os.path.join(save_dir, "walks.pth")
    if os.path.exists(walks_file):
        print("load saved walks...")
        node2walks = torch.load(walks_file)
    else:
        print("generate walks...")
        all_anchors = test + train
        target_anchors = set([ t for s, t in all_anchors ])    

        target_nx = nx.Graph()
        target_nx.add_edges_from(
            [ (h,t, {"rel": target_g.id2rel[r]}) 
            for h, r, t in target_g.rel_triples if h in target_anchors and t in target_anchors
            ])
        nx.set_node_attributes(target_nx, False, 'anchor')

        for s_id, t_id in train:
            if target_nx.has_node(t_id):
                target_nx.nodes[t_id]["anchor"] = True    

        node2walks = []
        longest = 0
        len_sum = 0
        total_walks = 0
        target2source = { t:s for s, t in all_anchors}
        L, R = 10, 10

        def random_draw(item_set, prev, n_try=4):
            for i in range(n_try):
                node_drawn = item_set[np.random.randint(0, len(item_set))]
                if not prev or (prev and node_drawn != prev) or (node_drawn == prev and len(item_set) == 1):
                    break
            return node_drawn

        for s_id, t_id in tqdm(train): 
            if target_nx.has_node(t_id):
                for _ in range(R):
                    walk_t = [t_id]
                    walk_s = [s_id]
                    mask = [1]
                    while len(walk_t) < L:
                        cur = walk_t[-1]
                        prev = walk_t[-2] if len(walk_t) > 1 else None
                        nbr_anchors = [ n for n in target_nx.adj[cur] if target_nx.nodes[n]["anchor"] == True]
                        nbr_nonanchors = [ n for n in target_nx.adj[cur] if target_nx.nodes[n]["anchor"] != True]

                        if target_nx.nodes[cur]["anchor"]:
                            sample_anchor = np.random.choice(2,p=[0.9, 0.1]) # hyperparameter
                        else:
                            sample_anchor = np.random.choice(2,p=[0.1, 0.9])
                            
                        if sample_anchor == 1 and nbr_anchors:
                            node_drawn = random_draw(nbr_anchors, prev)
                            walk_t.append(node_drawn)
                            walk_s.append(target2source[node_drawn])
                            mask.append(1)
                        elif sample_anchor == 1 and nbr_nonanchors:
                            node_drawn = random_draw(nbr_nonanchors, prev)
                            walk_t.append(node_drawn)
                            walk_s.append(target2source[node_drawn])           
                            mask.append(0)                         
                        elif sample_anchor == 0 and nbr_nonanchors:
                            node_drawn = random_draw(nbr_nonanchors, prev)
                            walk_t.append(node_drawn)
                            walk_s.append(target2source[node_drawn])  
                            mask.append(0)                      
                        elif sample_anchor == 0 and nbr_anchors:
                            node_drawn = random_draw(nbr_anchors, prev)
                            walk_t.append(node_drawn)
                            walk_s.append(target2source[node_drawn])
                            mask.append(1) 
                        else:
                            assert True, 'startring from an isolated node, not enough walk length'
                            break

                    len_sum += len(walk_t)
                    if len(walk_t) > longest:
                        longest = len(walk_t)
                    total_walks += 1  

                    node2walks.append((walk_t, walk_s, mask))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(node2walks, walks_file)

    return node2walks, train, test
                
if __name__ == "__main__":     
    class DBP15KReaderArgs:
        def __init__(self):
            self.DBP15K_clean_dir = 'datasets/DBP15K_clean'
            self.DBP15K_dir = 'datasets/DBP15k'
            self.subset = 'zh_en'

    args = DBP15KReaderArgs() 

    data = generate_data(args)              
