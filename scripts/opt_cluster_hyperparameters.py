"""
Script to optimize the hyperparameters of a clusteralgorithm on a superpatch
"""
# Core library imports
import argparse
import json
import os

# External Library imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from tilseg import model_selection


def parse_args():
    """
    Function to parse the command line arguments
    Parameters
    ----------
    None
    Resturns
    --------
    argparse.Namespace object containing the parsed arguments
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Script to optimize the hyperparameters of a clusteralgorithm on a superpatch")
    # Add argument for filepath
    parser.add_argument("-f", "--file",
                        dest="patch_path",
                        default=os.path.join(
                            "..", "abi_patches",
                            "patch", "TCGA-3C-AALI-01Z-00-DX1_T370.tif"),
                        help="File path to the superpatch")
    # Add argument for out directory
    parser.add_argument("-o", "--output",
                        dest="out_file",
                        default=os.path.join(".",
                                             "output",
                                             "optimized_hyperparameters.json"),
                        help="File to output json containing optimum hyperparameters to")
    # Add argument for cluster algorithm to optimize
    parser.add_argument("-c", "--clusterer",
                        dest="cluster_algorithm",
                        default="OPTICS",
                        help="Algorithm to optimize hyperparameters for")
    # Add argument for hyperparameter input file
    parser.add_argument("-p", "--hyperparameters",
                        dest="hyperparameter_path",
                        default=os.path.join(".", "hyperparameters.json"),
                        help="""File containig hyperparameters to try, should be a json.
                An object of hyperparameter:values[array]""")
    # Add argument for scoring method
    parser.add_argument("-m", "--metric",
                        dest="metric",
                        default="ch",
                        help="Metric to use for scoring the clustering")
    # Add argument for taking a sample
    parser.add_argument("-s", "--sample",
                        dest="sample",
                        default=None,
                        help="""
                        Number of samples to take from the data, reduces
                        accuracy but can dramatically increase speed. Value
                        will be the number of points sampled to perform the
                        optimization. If None, entire patch is used. 
                        Default None.                        
                        """)
    parser.add_argument("--combinations",
                        action="store_true",
                        dest = "combinations",
                        help = """
                        Whether to use all possible combinations of
                        the hyperparameters"""
                        )
    return parser.parse_args()


def main():
    """
    Main function, for when the script is called
    """
    # Parse the command line arguments
    args = parse_args()
    # Read in the patch, linearize, and normalize
    patch = np.float32(plt.imread(args.patch_path).reshape((-1, 3))/255.)
    # Read the hyperparameter json file
    with open(args.hyperparameter_path, "r") as file:
        hyperparameters = json.load(file)
    # adjust for "inf" and "None" which can't be naturally encoded in json
    for key, item in hyperparameters.items():
        modified_list = []
        for i in item:
            if i in ["inf", "Inf", "np.inf", "np.Inf"]:
                modified_list+=[np.inf]
            elif i in ["None", "none", "NaN", "NA"]:
                modified_list+=[None]
            else:
                modified_list+=[i]
        hyperparameters[key] = modified_list
    if args.combinations:
        hyperparameters = model_selection.\
            generate_hyperparameter_combinations(
            hyperparameter_dict=hyperparameters
            )
    # Take a sample if args.sample isn't None,
    if args.sample:
        patch = model_selection.sample_patch(patch, sample = args.sample)
    # Find which clustering algorithm is desired and
    if args.cluster_algorithm in ["KMeans", "Kmeans", "KMEANS", "KM", "km", "kmeans"]:
        # Transform result into dictionary so it can be written to
        #   a json like the others
        result = {
            "n_cluster": model_selection.opt_kmeans(
                patch,
                n_clusters=hyperparameters["n_clusters"])
        }
    elif args.cluster_algorithm in ["DBSCAN", "dbscan", "Dbscan"]:
        result = model_selection.opt_dbscan(patch,
                                            eps=hyperparameters["eps"],
                                            metric=args.metric)
    elif args.cluster_algorithm in ["OPTICS", "optics", "Optics"]:
        result = model_selection.opt_optics(
            patch,
            min_samples=hyperparameters["min_samples"],
            max_eps=hyperparameters["max_eps"],
            metric=args.metric
        )
    elif args.cluster_algorithm in ["BIRCH", "birch", "Birch"]:
        result = model_selection.opt_birch(
            patch,
            threshold=hyperparameters["threshold"],
            branching_factor=hyperparameters["branching_factor"],
            n_clusters=hyperparameters["n_clusters"],
            metric=args.metric)
    else:
        raise AttributeError("Couldn't parse provided clusterer name")
    # Change np.inf and None to strings so they can be written to json
    for key, value in result.items():
        if value == np.inf:
            result[key] = "inf"
        if value is None:
            result[key] = "None"
    # Write result
    with open(args.out_file, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
