import pandas as pd
from structured2 import StructuredReasoner
from context_management import Context
import networkx as nx
import matplotlib.pyplot as plt
import asyncio
import json
import os
import time
import hashlib
from datetime import datetime

import diskcache
from openai import OpenAI, AsyncOpenAI, OpenAIError
import dotenv
dotenv.load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
aclient = AsyncOpenAI()


logs_dir = os.path.join(os.getcwd(), '.chatgpt_history/logs')
cache_dir = os.path.join(os.getcwd(), '.chatgpt_history/cache')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

cache = diskcache.Cache(cache_dir)


def get_key(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def retry_on_exception(retries=5, initial_wait_time=1):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                wait_time = initial_wait_time
                for attempt in range(retries):
                    try:
                        return await func(*args, **kwargs)
                    except OpenAIError as e:
                        if attempt == retries - 1:
                            raise e
                        print(e)
                        await asyncio.sleep(wait_time)
                        wait_time *= 2
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                wait_time = initial_wait_time
                for attempt in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except OpenAIError as e:
                        if attempt == retries - 1:
                            raise e
                        print(e)
                        time.sleep(wait_time)
                        wait_time *= 2
            return sync_wrapper
    return decorator


@retry_on_exception()
def complete(messages=None, model='gpt-4-turbo-preview', temperature=0, use_cache=True, **kwargs):
    if use_cache:
        key = get_key(messages)
        if key in cache:
            return cache.get(key)
    response = client.chat.completions.create(
        messages=messages, model=model, temperature=temperature, **kwargs)
    return parse_response(response, messages, **kwargs)


@retry_on_exception()
async def acomplete(messages=None, model='gpt-4-turbo-preview', temperature=0, use_cache=True, **kwargs):
    if use_cache:
        key = get_key(messages)
        if key in cache:
            return cache.get(key)
    response = await aclient.chat.completions.create(messages=messages,
                                                     model=model,
                                                     temperature=temperature,
                                                     **kwargs)
    print(response)
    print(type(response))
    return parse_response(response, messages, **kwargs)


def parse_response(response, messages, **kwargs):
    n = kwargs.get('n', 1)
    stream = kwargs.get('stream', False)
    if stream:
        strm = parse_stream(response, messages, n=n)
        print(strm)
        print(type(strm))
        return strm

    results = []
    for choice in response.choices:
        message = choice.message
        if message.function_call:
            name = message.function_call.name
            try:
                args = json.loads(message.function_call.arguments)
            except json.decoder.JSONDecodeError as e:
                print('ERROR: OpenAI returned invalid JSON for function call arguments')
                raise e
            results.append({'role': 'function', 'name': name, 'args': args})
            log_completion(messages + [results[-1]])
        else:
            results.append(message.content)
            log_completion(messages + [message])

    output = results if n > 1 else results[0]
    cache.set(get_key(messages), output)
    return output


def parse_stream(response, messages, n=1):
    results = ['' for _ in range(n)]
    for chunk in response:
        for choice in chunk.choices:
            if not choice.delta:
                continue
            text = choice.delta.content
            if not text:
                continue
            idx = choice.index
            results[idx] += text
            if n == 1:
                yield text
            else:
                yield (text, idx)

    for r in results:
        log_completion(messages + [{'role': 'assistant', 'content': r}])
    cache.set(get_key(messages), results)


def log_completion(messages):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    save_path = os.path.join(logs_dir, timestamp + '.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    log = ""
    for message in messages:
        if not isinstance(message, dict):
            message = dict(message)
            message = {k: v for k, v in message.items() if v is not None}

        log += message['role'].upper() + ' ' + '-'*100 + '\n\n'
        if 'name' in message:
            log += f"Called function: {message['name']}("
            if 'args' in message:
                log += '\n'
                for k, v in message['args'].items():
                    log += f"\t{k}={repr(v)},\n"
            log += ')'
            if 'content' in message:
                log += '\nContent:\n' + message['content']
        elif 'function_call' in message:
            log += f"Called function: {message['function_call'].get('name', 'UNKNOWN')}(\n"
            log += ')'
        else:
            log += message["content"]
        log += '\n\n'

    with open(save_path, 'w') as f:
        f.write(log)


# from .context_management import Context
# from .structured2 import StructuredReasoner
# from . import chatgpt


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
            leaf_nodes = [node for node,
                          degree in self.graph.degree() if degree == 1]

            # If there are no leaf nodes, end the function
            if not leaf_nodes:
                print("No leaf nodes to explore from current graph.")
                return

            # Calculate the depth of each leaf node from 'ARF5' and sort them
            leaf_nodes_depth = {node: nx.shortest_path_length(
                self.graph, source=self.start_protein, target=node) for node in leaf_nodes}
            # Sort the leaf nodes by depth (or any other criteria you have)
            sorted_leaf_nodes = sorted(
                leaf_nodes_depth, key=leaf_nodes_depth.get)

            # Get the leaf node with the highest depth (or other criteria)
            protein_name = sorted_leaf_nodes[-1]

        # Add node and edges for the specified protein name
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
    def __init__(self, protein_graph: ProteinGraph, start_protein, start_interaction, stopping_condition, model='gpt-4'):
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
        return acomplete(self.context.messages, model=self.model, stream=True)

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

    def simulate_change(self, next_protein, edges):
        sim = self.simulate_state_change(next_protein, edges)
        return sim

    async def simulate_state_change(self, protein, edges):
        # Use the reasoner to simulate the state change for the protein
        prompt = self.clean_indent(f"""
        ```md
        {self.get_gene_description([protein, *edges['protein2']])}
        ```
        We're going to look specifically at {protein} with respect to the previous context.
        These are the genes in its network:
        {', '.join(edges['protein2'])}
        Simulate what is happening to the whole and what happens to the other genes in its network.
        """)
        self.context.add_message('user', prompt)
        ret = acomplete(self.context.messages,
                        model=self.model, stream=True)
        print(ret)
        print(type(ret))
        ret2 = await acomplete(self.context.messages,
                               model=self.model, stream=True)
        print(ret2)
        print(type(ret2))

        return ret

    def check_for_condition(self):
        # Use the reasoner to check if the state change causes damage
        self.reasoner.messages = self.context.messages
        self.reasoner.add_message(
            'user', f'Has the current analysis shown whether the state has reached the condition "{self.stopping_condition}"?')
        answer = self.reasoner.extract_info(
            'The condition has not been reached: {answer}', bool)
        return answer

    def clean_indent(self, s):
        return '\n'.join([line.lstrip() for line in s.split('\n')]).strip()


def printj(json_obj):
    print(json.dumps(json_obj, indent=4))


def clean_indent(s):
    return '\n'.join([line.lstrip() for line in s.split('\n')])


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

asyncio.run(start)
