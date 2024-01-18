import os
import mapillary.interface as mly

from PIL import Image

import torch
import blip_inference as blip
    
import pickle as pk

import networkx as nx
import osmnx as ox
from tqdm import tqdm


import geopy.distance

from matplotlib import pyplot as plt


import numpy as np


import argparse

import helper_functions as hf
import json
from random import sample



def main():
    parser = argparse.ArgumentParser(description='Find the optimal route under semantic constraints')


    parser.add_argument('--MLY_ACCESS_TOKEN', type=str, required = True, help='''Your Mapilary token. To get one, visit https://www.mapillary.com/dashboard/developer, go to 'developers', then 'register application', register a new application (use fake website if needed)''')

    parser.add_argument('--lat_long_start', type=hf.parse_float_tuple, required=True, help='latitude-longitude coordinates of the start point. Can be easily obtained by right clicking a point on google maps and clicking on the coordinates that appear.')
    
    parser.add_argument('--lat_long_destination', type=hf.parse_float_tuple, required=True, help='latitude-longitude coordinates of the end point. Can be easily obtained by right clicking a point on google maps and clicking on the coordinates that appear.')

    parser.add_argument('--positive_traits', nargs='+', type=str, help='List of desirable traits for your route. Specifically, a space separated list (using quotes ") of zero or more desireable traits.', default = ['pretty nature'])

    parser.add_argument('--negative_traits', nargs='+', type=str, help='List of UNdesirable traits for your route. Specifically, a space separated list (using quotes ") of zero or more desireable traits.', default = ['ugly construction'])

    parser.add_argument('--acceptable_deviation', type=float, help='''A decimal number representing the emphasis on the shortness of the route (larger number = potentially longer, but nicer, route). Specifically, this value represents the proportion longer than the shortest possible route that the returned route is allowed to be (e.g. 0.1 means the route can be at most 10% longer than the shortest possible route)''', default = 0.1)
    
    parser.add_argument('--n_paths_max', type=int, help='Maximum number of shortest paths to consider. Generally, this number should be set to as high as possible, however this may increase computation time.', default = 10000)
    
    parser.add_argument('--max_num_images_per_path', type=int, help='Maximum number of images to consider per path segment.', default = 2)
    
    parser.add_argument('--decision_strategy', type=str, default = 'mean_weighted', help='''Method of deciding on the optimal path. Want the best single street? Then choose maxmax. Want to avoid the worst? Then choose maxmin. Want the best on average? Choose mean. Want to take into consideration how long you're walking on the street? Then all _weighted to any of these.''')
                    
    parser.add_argument('--save_plots', action='store_true', help='Whether to save the plots as a jpg for future reference.')
                        
    parser.add_argument('--ignore_previous_embeddings', action='store_true', help='Whether to potentially recompute edge embeddings. Useful if increasing max_num_images_per_path.')
        
    parser.add_argument('--run_name', type=str, help='Name of run for file naming purposes.', default = 'my_route')
    

    args = parser.parse_args()
    
    MLY_ACCESS_TOKEN = args.MLY_ACCESS_TOKEN
    lat_long_start = args.lat_long_start
    lat_long_destination = args.lat_long_destination    
    positive_traits = args.positive_traits
    negative_traits = args.negative_traits
    acceptable_deviation = args.acceptable_deviation
    n_paths_max = args.n_paths_max
    max_num_images_per_path = args.max_num_images_per_path
    decision_strategy = args.decision_strategy
    save_plots = args.save_plots
    ignore_previous_embeddings = args.ignore_previous_embeddings
    run_name = args.run_name


    
    # Your token here!
    # To get one, visit https://www.mapillary.com/dashboard/developer, go to 'developers',
    # Then 'register application', register a new application (read access atleast),
    # then copy & paste the 'Client Token' here

    mly.set_access_token(MLY_ACCESS_TOKEN)
    
    
    
    
    #fudge factor (percentage increase) for bounding box
    eps = 0.1
    
    
    
    #download map/image data
    print('downloading map data...')
    #iamge search radius (m). Adding a small fudge factor so our POIs aren't on the boundary
    radius = geopy.distance.geodesic(lat_long_start, lat_long_destination).m/2+100
    
    mid_point = tuple(np.mean([np.array(lat_long_start),np.array(lat_long_destination)],axis=0))
    
    #sadly this method doesnt work for some reason due to an issue with mly
    #images = mly.get_image_close_to(*mid_point, radius=radius).to_dict()

    north_south_fudge = eps*(max([lat_long_start[0],lat_long_destination[0]])-min([lat_long_start[0],lat_long_destination[0]]))
    east_west_fudge = eps*(max([lat_long_start[1],lat_long_destination[1]])-min([lat_long_start[1],lat_long_destination[1]]))
    images = mly.images_in_bbox({'north': max([lat_long_start[0],lat_long_destination[0]])+north_south_fudge, 'south': min([lat_long_start[0],lat_long_destination[0]])-north_south_fudge, 'east': max([lat_long_start[1],lat_long_destination[1]])+east_west_fudge, 'west': min([lat_long_start[1],lat_long_destination[1]])-east_west_fudge})
    images = json.loads(images)
    
    osm_graph = ox.graph_from_point(mid_point, dist=radius, network_type="bike", simplify=True)
    
    
    
    
    #Subset by possible shortest routes
    print('subsetting map data...')
    osm_graph_subset = nx.DiGraph(osm_graph)
    routes = hf.get_routes_within_tolerance((lat_long_start[1],lat_long_start[0]), (lat_long_destination[1],lat_long_destination[0]), osm_graph_subset, weight = 'length', n_paths_max=n_paths_max, plot=False, tolerance=acceptable_deviation)
    

    keep_nodes = sum(routes,[])
    remove_nodes = [node for node in osm_graph_subset.nodes() if not(node in keep_nodes)]
    osm_graph_subset.remove_nodes_from(remove_nodes)
    osm_graph_subset = nx.MultiDiGraph(osm_graph_subset)
    
    #check out new subgraph
    fig, ax = ox.plot_graph(osm_graph_subset, node_size=0, figsize=(40,40), show = False, close = False)
    ax.set_title('Subgraph of Possible Routes', fontsize=50, color = 'white')
    if save_plots:
        plt.savefig(f'{run_name}_Subgraph_of_Possible_Routes.jpg', format='jpg', bbox_inches='tight')

    plt.show()
    
    if os.path.isfile('./embeddings.pkl') and not ignore_previous_embeddings:
        with open('./embeddings.pkl', 'rb') as pickle_file:
            embeddings_dict = pk.load(pickle_file)  
    else:
        embeddings_dict = dict()
        
    image_edge = []
    
    
    for I in tqdm(images['features'], desc = 'indexing steet images'):
        edge, dist = ox.nearest_edges(osm_graph_subset, *I['geometry']['coordinates'], return_dist=True)
        image_id = I['properties']['id']
        if image_id not in embeddings_dict:
            #this requiremnt ensures that the edge is within about 8ft. (0.00001~2ft)
            if dist<0.00004:
                image_edge.append([image_id, edge[:2]])
    
    
    import pandas as pd
    
    df = pd.DataFrame(image_edge, columns=['image_ID', 'edge'])
    
    
    # Apply the sampling function to each group (using the images with precomputed embeddings if existing)
    image_edge_subset = []
    for rows in df.groupby('edge', group_keys=False):
    
        precomputed = list(set(rows[1].image_ID).intersection(set(embeddings_dict.keys())))
        new = list(set(rows[1].image_ID).difference(set(embeddings_dict.keys())))
        additional = []
        if len(precomputed)<max_num_images_per_path:
            additional = sample(new,k=min(max_num_images_per_path,len(new)))
        
        all_images = precomputed+additional
        image_edge_subset += [[i,rows[0]] for i in all_images]
    
    image_edge_subset = pd.DataFrame(image_edge_subset, columns=['image_ID', 'edge'])
    
    
    
    #now it's time to get the image embeddings :)
    all_image_data = []
    
    
    # Define the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the BLIP model and define the image preprocessing pipeline
    model, preprocess = blip.load('base', device)
    
    
    
    for image_id in tqdm(image_edge_subset['image_ID'], desc = 'downloading street images'):
        if image_id not in embeddings_dict:
            image_data = hf.get_image(image_id, MLY_ACCESS_TOKEN)
            if image_data is not None:
                all_image_data.append([image_id,image_data])
        
    
    for imdat in tqdm(all_image_data, desc = 'computing image embeddings', total = len(all_image_data)):
        image_id, image_data = imdat
        image = Image.open(image_data)
    
        image_embedding = hf.get_embedding(image, model, preprocess, device)
        

        embeddings_dict[image_id] = image_embedding


    with open('./embeddings.pkl', 'wb') as file:
        pk.dump(embeddings_dict, file)
    
    
    
    
    
    similarities = hf.get_similarity_scores(embeddings_dict, image_edge_subset, model, device, positive_traits = positive_traits, negative_traits = negative_traits)
    similarities = {edge:np.mean(similarities[edge]) for edge in similarities}
    
    #impute the missing edges
    impute_value = np.mean(list(similarities.values()))
    for edge in osm_graph_subset.edges():
        if edge not in similarities:
            similarities[edge] = impute_value
    
    #convert to simple digraph for convience
    osm_graph_subset = nx.DiGraph(osm_graph_subset)

    nx.set_edge_attributes(osm_graph_subset, similarities, 'similarities')
    # Predict missing edge weights for all edges
    attr = 'similarities'
    for edge in osm_graph_subset.edges():
        if attr not in osm_graph_subset[edge[0]][edge[1]]:
            osm_graph_subset[edge[0]][edge[1]][attr] = hf.predict_missing_weights_recursive(osm_graph_subset, edge, attr)

    if 'weighted' in decision_strategy:
        score = {e:osm_graph_subset.get_edge_data(*e)['length']*similarities[e] for e in similarities}
    else:
        score = similarities

            
    nx.set_edge_attributes(osm_graph_subset, score, 'score')



    
    
    
    #get the best scoring route (weighted average of similarities)
    route_scores = []
    for route in routes:
        path_length = sum(ox.utils_graph.get_route_edge_attributes(nx.MultiDiGraph(osm_graph_subset), route, 'length'))
        edge_scores = ox.utils_graph.get_route_edge_attributes(nx.MultiDiGraph(osm_graph_subset), route, 'score')
        if decision_strategy == 'mean_weighted':
            route_score = sum(edge_scores)/path_length
        if decision_strategy == 'mean':
            route_score = np.mean(edge_scores)
        if 'maxmax' in decision_strategy:
            route_score = np.max(edge_scores)
        if 'maxmin' in decision_strategy:
            route_score = np.min(edge_scores)
        route_scores.append(route_score)
    
    
    #plot the best scoring route
    optimal_route = routes[np.argmax(route_scores)]
    fig, ax = ox.plot_graph_route(nx.MultiDiGraph(osm_graph_subset), optimal_route, node_size=0, figsize=(40,40), show = False, close = False)
    ax.set_title('Optimal Route', fontsize=50, color = 'white')
    if save_plots:
        plt.savefig(f'{run_name}_Optimal_Route.jpg', format='jpg', bbox_inches='tight')

    plt.show()
    
    #plot the shortest route
    fig, ax = ox.plot_graph_route(nx.MultiDiGraph(osm_graph_subset), routes[0], node_size=0, figsize=(40,40), show = False, close = False)
    ax.set_title('Shortest Route', fontsize=50, color = 'white')
    if save_plots:
        plt.savefig(f'{run_name}_Shortest_Route.jpg', format='jpg', bbox_inches='tight')

    plt.show()
    
    
    print(f'Here is the best route and distances (m): {hf.get_route_street_names(osm_graph_subset, optimal_route)}\n\n')
    print(f'Here is the shortest route and distances (m): {hf.get_route_street_names(osm_graph_subset, routes[0])}')
    
    #get best and worst scoring images for the optimal route
    top_bottom_images = hf.get_top_bottom_images(image_edge_subset, optimal_route, osm_graph_subset, MLY_ACCESS_TOKEN, k=4, attr = 'similarities')
    image_grid = hf.get_image_grid(*top_bottom_images)
    image_grid.suptitle('Best and worst images for Optimal Route', fontsize=16)
    if save_plots:
        plt.savefig(f'{run_name}_Optimal_Route_best_worst.jpg', format='jpg', bbox_inches='tight')
    plt.show()
    

    #do the same but for the shortest route
    top_bottom_images = hf.get_top_bottom_images(image_edge_subset, routes[0], osm_graph_subset, MLY_ACCESS_TOKEN, k=4, attr = 'similarities')
    image_grid = hf.get_image_grid(*top_bottom_images)
    image_grid.suptitle('Best and worst images for Shortest Route', fontsize=16)
    if save_plots:
        plt.savefig(f'{run_name}_Shorteset_Route_best_worst.jpg', format='jpg', bbox_inches='tight')
    plt.show()
    
    

if __name__ == '__main__':
    main()

