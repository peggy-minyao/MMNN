import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from hgraph.chemutils import *
from hgraph.nnutils import *

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object): #输入的是一个smile

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 20

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles) #将smile转换为mol，且把芳香键转换为单键和双键。

        self.mol_graph = self.build_mol_graph() #将分子转换为graph
        self.clusters, self.atom_cls = self.find_clusters() 
        self.mol_tree = self.tree_decomp() #将分子转换为tree
        self.order = self.label_tree()

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: #special case
            return [(0,)], [[0]]

        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing(): #如果不在环里，则将该键的前后原子序号加入clusters里
                clusters.append( (a1,a2) ) #有点像一个邻接矩阵

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)] #查看所有最小环，返回所有组成最小环的原子index
        clusters.extend(ssr) #用于在末尾一次性追加另一个序列的多个值,这里的clusters里包含所有的键和环的序号。

        if 0 not in clusters[0]: #root is not node[0] 不懂
            for i,cls in enumerate(clusters):
                if 0 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    #clusters[i], clusters[0] = clusters[0], clusters[i]
                    break

        atom_cls = [[] for i in range(n_atoms)] #根据原子数量创建列表，假设n_atoms=6，那么atom_cls = [[], [], [], [], [], []]
        for i in range(len(clusters)): #返回每个原子属于哪几个clusters
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls #clusters：将分子分成几个不同的部分，即打成片段（键和环）；atom_cls返回每个原子属于哪个cluster

    def tree_decomp(self): #将分子中的模体表征为一个节点，并连接起来，对于其中一个原子来说，满足某个条件则单独作为一个节点。#mol_tree
        clusters = self.clusters
        graph = nx.empty_graph( len(clusters) ) #返回n个节点，0条边的图
        for atom, nei_cls in enumerate(self.atom_cls): #原子编号与其属于的clusters编号,
            if len(nei_cls) <= 1: continue #如果原子只属于一个或者不属于哪个clusters（一般是边缘的原子）,则跳过该原子执行下面的代码，而从下一个循环开始。
            bonds = [c for c in nei_cls if len(clusters[c]) == 2] #返回该原子内属于键的clusters,假设bonds=[2,3],则在该原子属于的clusters中，clusters2，3是键
            rings = [c for c in nei_cls if len(clusters[c]) > 4] #need to change to 2，返回属于环的clusters，len(clusters[c]) > 4代表clusters[c]内的原子数目大于4。

            if len(nei_cls) > 2 and len(bonds) >= 2: #如果该原子属于两个以上的clusters并且，键的clusters大于2，涉及两种情况，一个原子连接三个键，或连接两个键一个环
                clusters.append([atom]) #符合上面条件的原子单独拎出来，作为一个clusters，接到前面的存放clusters的list中去
                c2 = len(clusters) - 1 #c2等于一个数字
                graph.add_node(c2) #加一个节点，节点标签为c2
                for c1 in nei_cls: #将新加的节点与该原子与其周围的clusters连接起来
                    graph.add_edge(c1, c2, weight = 100) #新加的原子的边的权重设置为100。（如果一个原子周围有三个及三个以上的键相连接时，单独提取出来作为一个clusters，且与周围的权重设为100.

            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2  如果该原子属于的clusters里，有两以上的环
                clusters.append([atom]) #temporary value, need to change，则该原子再次成为一个新的clusters，
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)#新加的 原子再次加上100的权重，即加上上面的情况，有两种情况会加上100的权重。
            else:
                for i,c1 in enumerate(nei_cls): #c1为该原子属于的多个clusters中的第一个，c2为第二个。
                    for c2 in nei_cls[i + 1:]: 
                        inter = set(clusters[c1]) & set(clusters[c2]) # &是一个逻辑符号，输出左右两个集合的交集。所以这个weight应该是和周围clusters的公用原子个数，大部分为1，
                        graph.add_edge(c1, c2, weight = len(inter)) #len(inter) 有多少？？作者的数据有1, 2, 3, 4, 5, 6, 7, 8, 100，我的数据有1, 2, 3, 4, 5, 6, 7, 8, 12, 23, 100

        n, m = len(graph.nodes), len(graph.edges)
        assert n - m <= 1 #must be connected
        return graph if n - m == 1 else nx.maximum_spanning_tree(graph) #如果n-m等于1就返回graph，否则返回最大生成树。
        #最小生成树，所有的顶点，以最短的路径连接，最大生成树与其相反。

    def label_tree(self): #给生成的树加标签
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) #better performance with fixed order
            for idx,y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0 
                self.mol_tree[y][x]['label'] = idx + 1 #position encoding
                prev_sib[y] = sorted_child[:idx] 
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x,y,1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y,x,0) )

        order, pa = [], {} #建立空的order和pa
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))] #有几个clusters就创建几个空的[ ]
        dfs(order, pa, prev_sib, 0, -1)

        order.append( (0, None, 0) ) #last backtrack at root
        
        mol = get_mol(self.smiles)
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() + 1 ) #给原子打上数字标签

        tree = self.mol_tree
        for i,cls in enumerate(self.clusters):
            inter_atoms = set(cls) & set(self.clusters[pa[i]]) if pa[i] >= 0 else set([0])
            cmol, inter_label = get_inter_label(mol, cls, inter_atoms)
            tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
            tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            tree.nodes[i]['cluster'] = cls 
            tree.nodes[i]['assm_cands'] = []

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in self.clusters[c]] 
                pa_cls = self.clusters[ pa[i] ]
                tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.mol_graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        return order
       
    def build_mol_graph(self): #创造分子图，并为节点加入原子类型和原子电荷的信息，为键加上键的类型的信息。
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index( bond.GetBondType() ) 
            graph[a1][a2]['label'] = btype #储存的是整数，假设是单键，则返回0，双键则返回1
            graph[a2][a1]['label'] = btype

        return graph
    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab): #avocab :conmmon atom vocab
        mol_batch = [MolGraph(x) for x in mol_batch] #[MolGraph(x)返回一个类，包含了分子x的各种特性
        tree_tensors, tree_batchG = MolGraph.tensorize_graph([x.mol_tree for x in mol_batch], vocab) #
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab) #graph_batchG是将一个batch中的所有的graph放在一起，graph_tensors是包括键的信息、节点信息等打包放在一起的tensors
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max( [len(c) for x in mol_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)

        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch): #该循环输入一个batch的2量，graph_batch中的molecular_graph或者tree_graph，bid是该batch内的编号，G是图。
            offset = len(fnode)
            scope.append( (offset, len(G)) ) #？？len(G)返回该分子的原子数
            G = nx.convert_node_labels_to_integers(G, first_label=offset) #返回图G的副本，并以offset为偏移量重新标记节点为连续的整数
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] ) #该G有几个node就加几个None到fnode数据框中去？

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid #给图中的节点编上号，注意这里是batch_id, 该batch内的第一个分子全部编号1，第二个编号2，以此类推
                fnode[v] = vocab[attr] #根据vocab的编号，在fnode中填上编号，每个编号代表一种原子类型和其带有的形式电荷
                agraph.append([]) #有几个原子在agraph中加上几个数据框

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else:
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1 #给边编上号，1，2，3，4，5，注意双向边两个方向的编号是不一样的
                G[u][v]['mess_idx'] = eid #把编号打到G上
                agraph[v].append(eid) #agraph在上面加上了原子个数的空框，这里给每个原子由哪些键连接赋上信息。
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)] #eid是一个数字，上面定义
                for w in G.predecessors(u): #？？这个函数
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode) #转换为一种tensor
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

if __name__ == "__main__":
    import sys
    
    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1','O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1', 'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1', 'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1', 'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl', 'CCOCCCNC(=O)c1cc(OC)ccc1Br', 'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1', 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F', 'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N', 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1', 'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1', 'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1', 'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1', 'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']

    for s in sys.stdin:#test_smiles:
        print(s.strip("\r\n "))
        #mol = Chem.MolFromSmiles(s)
        #for a in mol.GetAtoms():
        #    a.SetAtomMapNum( a.GetIdx() )
        #print(Chem.MolToSmiles(mol))

        hmol = MolGraph(s)
        print(hmol.clusters)
        #print(list(hmol.mol_tree.edges))
        print(nx.get_node_attributes(hmol.mol_tree, 'label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'inter_label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'assm_cands'))
        #print(hmol.order)
