# src/data_loaders/graph_builder.py
import pandas as pd
import networkx as nx
import numpy as np

class GraphBuilder:
    """
    Advanced Graph Feature Extractor for Financial Fraud.
    Captures topological importance and flow dynamics to resist adversarial evasion.
    """
    def __init__(self, df):
        self.df = df
        self.G = nx.from_pandas_edgelist(
            df, 
            source='nameOrig', 
            target='nameDest', 
            edge_attr=['amount'], 
            create_using=nx.DiGraph()
        )

    def get_graph_features(self):
        """Extracts 8 specific graph features that closed the 'Recall Gap' in Notebook 4."""
        print("Starting Research-Grade Graph Feature Extraction...")
        
        pr = nx.pagerank(self.G, alpha=0.85)
        
        in_degree = dict(self.G.in_degree())
        out_degree = dict(self.G.out_degree())
        
        clustering = nx.clustering(self.G.to_undirected())

        self.df['orig_in_degree'] = self.df['nameOrig'].map(in_degree).fillna(0)
        self.df['orig_out_degree'] = self.df['nameOrig'].map(out_degree).fillna(0)
        self.df['orig_pagerank'] = self.df['nameOrig'].map(pr).fillna(1e-9)
        self.df['orig_clustering'] = self.df['nameOrig'].map(clustering).fillna(0)

        self.df['dest_in_degree'] = self.df['nameDest'].map(in_degree).fillna(0)
        self.df['dest_out_degree'] = self.df['nameDest'].map(out_degree).fillna(0)
        self.df['dest_pagerank'] = self.df['nameDest'].map(pr).fillna(1e-9)

        self.df['orig_flow_ratio'] = self.df['orig_in_degree'] / (self.df['orig_out_degree'] + 1e-9)
        
        print(f"Graph Construction Complete. Total Nodes: {self.G.number_of_nodes()}")
        return self.df

    def get_adversarial_threshold(self):
        """Returns mean PageRank for the Calibration Defense found in Notebook 5."""
        pr_values = list(nx.pagerank(self.G).values())
        return np.mean(pr_values)
