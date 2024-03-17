import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx
from .context_management import Context
from .structured2 import StructuredReasoner
from . import chatgpt
from openai import OpenAI, AsyncOpenAI, OpenAIError
import os
import dotenv
dotenv.load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
aclient = AsyncOpenAI()
# from context_management import Context
# from structured2 import StructuredReasoner
# import chatgpt


class ProteinGraph:
    def __init__(self, info_file_path, links_file_path, start_protein):
        # Load protein information
        self.protein_info = pd.read_csv(
            # info_file_path, sep='\t', compression='gzip')
            info_file_path, sep='\t')

        # Load protein links
        self.protein_links = pd.read_csv(
            # links_file_path, sep=' ', compression='gzip')
            links_file_path, sep=' ')

        # Create a mapping from preferred names to protein IDs
        self.name_to_id = dict(
            zip(self.protein_info['preferred_name'], self.protein_info['#string_protein_id']))

        # Create a mapping from protein IDs to preferred names
        self.id_to_name = dict(
            zip(self.protein_info['#string_protein_id'], self.protein_info['preferred_name']))

        # Initialize an empty graph
        self.graph = nx.Graph()

        self.start_protein = start_protein

        self.visited = set()

    def get_interacting_proteins(self, protein_name):
        if protein_name not in self.name_to_id:
            return f"Protein '{protein_name}' not found in the dataset."

        protein_id = self.name_to_id[protein_name]

        # Filter interactions involving the protein of interest
        interacting_proteins = self.protein_links[
            (self.protein_links['protein1'] == protein_id) |
            (self.protein_links['protein2'] == protein_id)
        ]

        # Flip the proteins if necessary
        def flip_proteins(row, target_protein):
            if row['protein2'] == target_protein:
                return row['protein2'], row['protein1'], row['combined_score']
            return row['protein1'], row['protein2'], row['combined_score']

        # Apply the flip_proteins function and explicitly cast the 'combined_score' to float to avoid dtype incompatibility issues
        flipped_proteins = interacting_proteins.apply(
            lambda row: flip_proteins(row, protein_id), axis=1, result_type='expand')
        interacting_proteins['protein1'] = flipped_proteins[0]
        interacting_proteins['protein2'] = flipped_proteins[1]
        interacting_proteins['combined_score'] = flipped_proteins[2].astype(
            float)

        # Group by protein pairs and keep the interaction with the highest score
        interacting_proteins = interacting_proteins.groupby(
            ['protein1', 'protein2'], as_index=False
        ).agg({'combined_score': 'max'})

        # Map protein IDs to their preferred names
        interacting_proteins['protein1'] = interacting_proteins['protein1'].map(
            self.id_to_name)
        interacting_proteins['protein2'] = interacting_proteins['protein2'].map(
            self.id_to_name)

        # Sort by combined_score in descending order before returning
        interacting_proteins = interacting_proteins.sort_values(
            by='combined_score', ascending=False).reset_index(drop=True)

        # Return the dataframe with protein names instead of IDs
        return interacting_proteins

    def get_protein_info(self, protein_name):
        if protein_name not in self.name_to_id:
            return f"Protein '{protein_name}' not found in the dataset."

        protein_id = self.name_to_id[protein_name]
        return self.protein_info[self.protein_info['#string_protein_id'] == protein_id]

    def get_protein_info_by_id(self, protein_id):
        return self.protein_info[self.protein_info['#string_protein_id'] == protein_id]

    def explore_protein(self, protein_name=None, top_n=5):
        if protein_name is None:
            if not self.graph.nodes:
                print("No starting protein specified.")
                return

            # Find all leaf nodes (nodes with degree 1)
            not_vis_nodes = [
                node for node in self.graph.nodes() if node not in self.visited]

            # If there are no leaf nodes, end the function
            if not not_vis_nodes:
                print("No leaf nodes to explore from current graph.")
                return

            # Calculate the depth of each leaf node from 'ARF5' and sort them
            not_vis_nodes_depth = {node: nx.shortest_path_length(
                self.graph, source=self.start_protein, target=node) for node in not_vis_nodes}
            # Sort the leaf nodes by depth (or any other criteria you have)
            sorted_not_vis_nodes = sorted(
                not_vis_nodes_depth, key=not_vis_nodes_depth.get)

            # Get the leaf node with the lowest depth (or other criteria)
            protein_name = sorted_not_vis_nodes[0]

        # Add node and edges for the specified protein name
        self.visited.add(protein_name)
        edges = self.add_protein_and_edges(protein_name, top_n)
        return protein_name, edges

    def add_protein_and_edges(self, protein_name, top_n):
        if protein_name not in self.name_to_id:
            print(f"Protein '{protein_name}' not found in the dataset.")
            return

        # Get interacting proteins, map to names, and sort by score
        potential_edges = self.get_interacting_proteins(
            protein_name).head(top_n)

        # Add edges to the graph
        for _, row in potential_edges.iterrows():
            self.graph.add_edge(
                row['protein1'], row['protein2'], weight=row['combined_score'])

        return potential_edges

    def visualize_graph(self):
        plt.figure(figsize=(10, 10))
        # Positioning the nodes using the spring layout
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue',
                node_size=2000, font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.title("Protein Interaction Graph")
        plt.show()

    def get_graph_data_for_visualization(self):
        # Convert the networkx graph into a format suitable for D3.js
        nodes = [{"id": node, "label": node} for node in self.graph.nodes()]
        edges = [{"source": u, "target": v, "weight": d["weight"]}
                 for u, v, d in self.graph.edges(data=True)]
        return nodes, edges


class ProteinExplorerAgent:
    # def __init__(self, protein_graph: ProteinGraph, start_protein, start_interaction, stopping_condition, model='gpt-4-turbo-preview'):
    def __init__(self, protein_graph: ProteinGraph, start_protein, start_interaction, stopping_condition, model='gpt-3.5-turbo'):
        self.protein_graph = protein_graph
        self.current_protein = start_protein
        self.start_protein = start_protein
        self.start_interaction = start_interaction
        self.stopping_condition = stopping_condition
        self.model = model
        self.context = Context()
        self.system_prompt = self.clean_indent("""
            You are a gene perturbation simulator for human cells. You are exploring the effects of perturbing genes in a protein interaction network. You will base your analyses with respect to information and conclusions you have seen so far.
            """)
        self.reasoner = StructuredReasoner(
            system_prompt=self.system_prompt, model=self.model)
        self.explored_proteins = set()

    def start(self, start_protein, edges):
        # Add initial message about the start protein
        self.context.add_message('system', self.system_prompt, idx=0)
        prompt = self.clean_indent(f"""
        ```md
        {self.get_gene_description([start_protein, *edges['protein2']])}
        ```
        {start_protein} has been {self.start_interaction}
        Simulate what happens on the whole and what happens to the other genes in its network which include {', '.join(edges['protein2'])}
        """)
        self.context.add_message('user', prompt)
        # ret = chatgpt.acomplete(self.context.messages,
        #                         model=self.model, stream=True)
        ret = aclient.chat.completions.create(messages=self.context.messages,
                                              model=self.model,
                                              temperature=0,
                                              stream=True)
        return ret

    def get_gene_description(self, gene):
        if isinstance(gene, list):
            descriptions = []
            for g in gene:
                annotation = self.protein_graph.get_protein_info(g)[
                    'annotation'].iloc[0]
                descriptions.append(f"gene: {g}\n{annotation}\n")
            return "\n".join(descriptions)
        else:
            annotation = self.protein_graph.get_protein_info(gene)[
                'annotation'].iloc[0]
            return f"gene: {gene}\n{annotation}\n"

    def explore(self, start_protein=None):
        next_protein, edges = self.protein_graph.explore_protein(
            protein_name=start_protein)
        if next_protein in self.explored_proteins:
            return None, None
        self.current_protein = next_protein
        self.explored_proteins.add(next_protein)
        return next_protein, edges

    def simulate_state_change(self, protein):
        # Use the reasoner to simulate the state change for the protein
        nodes, edges = self.protein_graph.get_graph_data_for_visualization()
        # Get the direct nodes to the protein
        direct_nodes = [edge['target']
                        for edge in edges if edge['source'] == protein]
        prompt = self.clean_indent(f"""
        ```md
        {self.get_gene_description([protein] + direct_nodes)}
        ```
        You talked about the effects the initial perturbation had on {protein}. Although, these effects are indirect and are not directly caused by the perturbation of {self.start_protein}, let's think step by step about those effects with reference to all the previous chain of events that has happened so far.
        We want to see the propogation of the effects on the gene network of {protein}

        These are the genes in its network:
        {', '.join(direct_nodes)}

        Simulate what is happening to the whole and what happens to the other genes with reference to {protein}.
        """)
        self.context.add_message('user', prompt)
        # ret = chatgpt.acomplete(self.context.messages,
        #                         model=self.model, stream=True)
        ret = aclient.chat.completions.create(messages=self.context.messages,
                                              model=self.model,
                                              temperature=0,
                                              stream=True)
        return ret

    def check_for_condition(self):
        # Use the reasoner to check if the state change causes damage
        self.reasoner.messages = self.context.messages.copy()
        self.reasoner.add_message(
            'user', f'Has the current analysis shown whether the state has reached the condition "{self.stopping_condition}"?')
        answer = self.reasoner.extract_info(
            'The condition has not been reached: {answer}', bool)
        return answer

    def clean_indent(self, s):
        return '\n'.join([line.lstrip() for line in s.split('\n')])


def printj(json_obj):
    print(json.dumps(json_obj, indent=4))


def clean_indent(s):
    return '\n'.join([line.lstrip() for line in s.split('\n')])


if __name__ == '__main__':
    # info_file_path = '/Users/danielgeorge/Documents/work/bio/futurehouse/gene_stepper/graph_db/9606.protein.info.v12.0.txt.gz'
    # links_file_path = '/Users/danielgeorge/Documents/work/bio/futurehouse/gene_stepper/graph_db/9606.protein.links.v12.0.txt.gz'
    info_file_path = '/Users/danielgeorge/Documents/work/bio/futurehouse/gene_stepper/graph_db/9606.protein.info.v12.0.txt'
    links_file_path = '/Users/danielgeorge/Documents/work/bio/futurehouse/gene_stepper/graph_db/9606.protein.links.v12.0.txt'
    protein_graph = ProteinGraph(
        info_file_path, links_file_path, start_protein='ARF5')
    protein_explorer = ProteinExplorerAgent(
        protein_graph, start_protein='ARF5', start_interaction='knocked out', stopping_condition='apoptosis')
    start_protein, edges = protein_explorer.explore(
        start_protein='ARF5')
    start = protein_explorer.start(start_protein=start_protein, edges=edges)
    print(start)

    for _ in range(3):
        next_protein, edges = protein_explorer.explore()
        print(next_protein)
        sim = protein_explorer.simulate_change(next_protein, edges)
        print(sim)
        answer = protein_explorer.check_for_condition()
        print(f'condition of apoptosis reached: {answer}')
        protein_graph.visualize_graph()
