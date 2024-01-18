Walkle Maps: Semantically Constrained Routing.

The goal of this project is to leverage pretrained multimodal language-image models in order to find the best routes under semantic (positive or negative) constraints.

**Example Usage**: python main.py --MLY_ACCESS_TOKEN ### --lat_long_start 47.60705499736696, -122.34109826277559 --lat_long_destination 47.630117702184506, -122.31510402574143 --positive_traits "pretty trees" --negative_traits "construction zones" "big streets" --acceptable_deviation 0.1 --n_paths_max 10000 --max_num_images_per_path 2 --decision_strategy mean_weighted --save_plots --run_name example_run


This will compute the "optimal" route between lat_long_start and lat_long_destination, where "optimal" is defined to be the route with the highest weighted mean semantic score among the top n_paths_max routes which are within acceptable_deviation of the shortest route. The semantic score is Calculated by considering max_num_images_per_path images on each edge and computing similarity with the positive_traits and subtracting the similarity with the negative_traits. This will return plots and example images of the shortest and optimal route and additionally print the street names of the optimal route. ![my_route_Optimal_Route](https://github.com/sashapersonxyz/Walkle-Maps/assets/156481390/47262c6c-5c2a-43b1-ab1a-5c93628ab0a0)
