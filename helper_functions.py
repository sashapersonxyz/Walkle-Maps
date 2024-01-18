import requests
from PIL import Image
from io import BytesIO
import torch
import blip_inference as blip
import networkx as nx
import osmnx as ox
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import argparse


def get_embedding(image, model, preprocess, device):
    '''Function to get image embedding'''
    with torch.no_grad():
        image = preprocess(image)
        image = image.unsqueeze(0).to(device)
        image_embedding = model.encode_image(image)
    return image_embedding




def sample_within_group(group, max_num_images_per_path):
    '''Function to sample at most max_num_images_per_path random rows within each unique value of the second column'''
    return group.sample(min(max_num_images_per_path, len(group)))



def get_routes_within_tolerance(start, end, graph, weight='length', plot=False, n_paths_max = 1, tolerance = 0.1):
    '''Function to get all shortest routes within (1+tolerance) of the shortest route. Will return at most n_paths_max routes.'''
    source_node = ox.nearest_nodes(graph, *start)
    target_node = ox.nearest_nodes(graph, *end)
    

    routes = nx.shortest_simple_paths(graph, source_node, target_node, weight=weight)
    possible_routes = []
    graph = nx.MultiDiGraph(graph)
    for i,route in tqdm(enumerate(routes), desc = "computing possible routes", total = n_paths_max):
        if i==0:
            shortest_length = sum(ox.utils_graph.get_route_edge_attributes(graph, route, 'length'))
        if sum(ox.utils_graph.get_route_edge_attributes(graph, route, 'length')) <= (1+tolerance)*shortest_length:
            possible_routes.append(route)
        else:
            break
        if i >= n_paths_max-1:
            break
    
    # edge_lengths = ox.utils_graph.get_route_edge_attributes(graph, route, 'length') 
    # edge_travel_time = ox.utils_graph.get_route_edge_attributes( graph_type[go_type], route, 'travel_time') 
    # total_route_length = round(sum(edge_lengths), 1)
    # route_travel_time  = round(sum(edge_travel_time)/60, 2)
    
    if plot:
        for route in possible_routes:
            ox.plot_graph_route(graph, route, node_size=0, figsize=(40,40))
    return possible_routes#, total_route_length, route_travel_time


def get_similarity_scores(embeddings_dict, image_edge_subset, model, device, positive_traits = [], negative_traits = []):
    '''Function to compute similarity scores between the precomputed image embeddings and the positive and negative traits. Note, the negative traits contribute negatively to the overall score'''
    all_traits = positive_traits+negative_traits
    assert len(all_traits)

    tokenized_text = blip.tokenize(all_traits).to(device)
    text_embedding = model.encode_text(tokenized_text)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    
    similarities = dict()
    for edge, dat in image_edge_subset.groupby('edge', group_keys=False):
        for image_ID in dat['image_ID']:
            if image_ID in embeddings_dict:
                image_embedding = embeddings_dict[image_ID]
                with torch.no_grad():
                    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                    similarity = image_embedding @ text_embedding.T
                    similarity = similarity.squeeze().numpy()#similarity = similarity.squeeze().item()
        
        
                similarity = (similarity+1)/2
                
                if len(all_traits) == 1:
                    similarity = [similarity]
                
                #make sure the negative traits count negatively
                S = []
                for i, s in enumerate(similarity):
                    if i >= len(positive_traits):
                         s = -s
                    S.append(s)
                similarities[edge] = similarities.get(edge,[])+[np.mean(S)]
    return similarities


def get_image(image_id,MLY_ACCESS_TOKEN):
    header = {'Authorization' : f'OAuth {MLY_ACCESS_TOKEN}'}
    url = f'https://graph.mapillary.com/{image_id}?fields=thumb_2048_url'
    r = requests.get(url, headers=header)
    data = r.json()
    image_data = None
    if 'thumb_2048_url' in data:
        image_url = data['thumb_2048_url']
    
        image_data = requests.get(image_url, stream=True).content
        image_data=BytesIO(image_data)
    return image_data

def get_top_bottom_k_indices(lst, k):
    if len(lst) <= k:
        return list(range(len(lst))), list(range(len(lst)))

    top_k_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:k]
    bottom_k_indices = sorted(range(len(lst)), key=lambda i: lst[i])[:k]

    return top_k_indices, bottom_k_indices



def get_top_bottom_images(image_edge_subset, route, graph, MLY_ACCESS_TOKEN, k=4, attr = 'similarities'):
    scores = ox.utils_graph.get_route_edge_attributes(nx.MultiDiGraph(graph), route, attr)
    top_bottom_inds = get_top_bottom_k_indices(scores,k)
    top_bottom_images = []
    for inds in top_bottom_inds:
        images = []
        for i in inds:
            edge = (route[i],route[i+1])
            if edge in set(image_edge_subset['edge']):
                image_id = image_edge_subset[image_edge_subset['edge']==edge].iloc[0,0]
                images.append(get_image(image_id, MLY_ACCESS_TOKEN))
            else:
                images.append(None)
        top_bottom_images.append(images)
    return top_bottom_images






def get_image_grid(list1, list2):
    k = len(list1)

    fig, axarr = plt.subplots(2, k, figsize=(15, 6))

    for ax in axarr.flat:
        ax.axis('off')

    for i in range(2):
        for j in range(k):
            image_path = list1[j] if i == 0 else list2[j]

            if image_path is not None:
                img = Image.open(image_path)
                axarr[i, j].imshow(np.asarray(img))
    axarr[0, 0].set_title("Highest Scoring", loc='center', fontsize=12)
    axarr[1, 0].set_title("Lowest scoring", loc='center', fontsize=12)
    
    return fig

def predict_missing_weights_recursive(graph, edge, attr, order=1):
    u, v = edge
    
    # Get neighbors based on the specified order
    neighbors = set()
    for node in list(graph.neighbors(u)) + list(graph.neighbors(v)):
        neighbors.update(list(graph.neighbors(node)))
    
    # Filter out edges with missing weights
    neighbor_weights = [graph[u][v][attr] for u, v in graph.edges(neighbors) if attr in graph[u][v]]
    
    # Predict missing weight by averaging neighboring edge weights
    if neighbor_weights:
        return sum(neighbor_weights) / len(neighbor_weights)
    elif order < len(graph.nodes):
        # If no weights found and order < number of nodes, recurse with a higher-order neighborhood
        return predict_missing_weights_recursive(graph, edge, order=order+1)
    else:
        return None  # No neighboring edges with weights





def parse_float_tuple(arg):
    try:
        # Split the input string into a list of floats
        floats = [float(x) for x in arg.split(',')]
        return tuple(floats)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float tuple: {arg}")

def get_route_street_names(graph, route, return_length = True):
    '''get the names (if exist) of each of the edges in route.'''
    street_names = []
    lengths = []
    for i in range(len(route)-1):
        try:
            street_name = graph.get_edge_data(route[i], route[i+1])['name']
        except:
            street_name = 'unnamed'
        try:
            length = int(graph.get_edge_data(route[i], route[i+1])['length'])
        except:
            length = 0
        if len(street_names):
            if street_names[-1] != street_name:
                street_names.append(street_name)
                lengths.append(length)
            else:
                lengths[-1] = lengths[-1] + length
        else:
            street_names.append(street_name)
            lengths.append(length)
            
    if return_length:
        return_value = [(name, length) for name, length in zip(street_names, lengths)]
    else:
        return_value = street_names
    
    return return_value