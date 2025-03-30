import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_performance_file(file_path, metrics=["auroc", "auprc"]):
    """
    Read performance results from a file for the specified metrics.

    Args:
        file_path (str): Path to the performance file.
        metrics (list): List of metric names to extract (e.g., ["accuracy", "mcc"]).

    Returns:
        dict: Dictionary with metric names as keys and their values (or NaN if not found).
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        result = {}
        for metric in metrics:
            # Search for the metric followed by '=' or ':' and a number, case-insensitive
            pattern = rf"{metric}\s*[=:]\s*([\d.]+)"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result[metric] = float(match.group(1))
            else:
                result[metric] = np.nan
        return result
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {metric: np.nan for metric in metrics}

def extract_target_from_filename(filename,target_type):
    """
    Extract the target dataset name from the filename.

    Args:
        filename (str): Name of the file.

    Returns:
        str: Extracted target name or "unknown" if not found.
    """
    targets = ["gpcr", "ionchannel", "kinase", "nuclearreceptor", "protease", "transporter"]
    for target in targets:
        if target in filename.lower() and target != target_type:
            return target
    
    pattern = r"results-.*?-([a-zA-Z]+)-\["
    match = re.search(pattern, filename)
    if match and match.group(1) in targets:
        return match.group(1)
    
    pattern = r"-([a-zA-Z]+)-\['chemprop'\]"
    match = re.search(pattern, filename)
    if match and match.group(1) in targets:
        return match.group(1)
    
    print(f"Warning: Could not extract target from filename: {filename}")
    return "unknown"

ALL_TARGETS = ["gpcr", "ionchannel", "kinase", "nuclearreceptor", "protease", "transporter"]

# Define methodology groups for better organization
METHODOLOGY_GROUPS = {
    "shallow": ["svm", "rf", "gb"],
    "fine-tuned": ["fine-tuned"],
    "freeze": ["freeze"],
    "scratch": ["scratch"]
}

# Define method order for consistent display
METHOD_ORDER = ["scratch", "shallow",  "fine-tuned", "freeze"]

def visualize_set_results(base_dir, output_dir="visualization_results", metric="auroc", all_metrics=["auroc", "auprc"]):
    """
    Visualize the performance comparison between different methods for the specified metric.
    Results are primarily grouped by methodology (shallow/fine-tune/freeze/scratch).

    Args:
        base_dir (str): Directory containing result files (e.g., "result_files/transporter/dataSubset500").
        output_dir (str): Directory to save visualizations.
        metric (str): The metric to plot (e.g., "auroc").
        all_metrics (list): List of all metrics to collect (e.g., ["auroc", "auprc"]).

    Returns:
        dict: Performance results for all methods and targets.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir,base_dir.split('/')[1])
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(base_dir))
    
    path_parts = base_dir.split(os.sep)
    target_type = next((part for part in path_parts if part in ALL_TARGETS), "unknown")
    if target_type == "unknown":
        print(f"Warning: Could not identify target type from path {base_dir}. Using 'unknown'.")
    else:
        print(f"Identified target type: {target_type}")
    
    # Define methods and their directories
    methods = {
        "shallow": ["svm", "rf", "gb"],
        "fine-tuned": ["fine-tuned"],
        "freeze": ["freeze"]
    }
    
    targets = [t for t in ALL_TARGETS if t != target_type]
    print(f"Using target datasets: {targets}")
    
    # Dictionary to store results
    results = {}
    
    # Process shallow, fine-tuned, and freeze methods
    for method_type, method_prefixes in methods.items():
        method_dir = os.path.join(base_dir, method_type)
        if not os.path.exists(method_dir):
            print(f"Warning: Directory {method_dir} does not exist. Skipping.")
            continue
            
        if method_type == "shallow":
            for algorithm in method_prefixes:
                for target in targets:
                    if target == "kinase":
                        layer0_pattern = f"{algorithm}_layer_0_per_results-{target}-{target_type}"
                        for file in os.listdir(method_dir):
                            if layer0_pattern in file:
                                key = f"{method_type}_{algorithm}_layer0_{target}"
                                results[key] = read_performance_file(os.path.join(method_dir, file), metrics=all_metrics)
                    
                    layer1_pattern = f"{algorithm}_layer_1_per_results-{target}-{target_type}"
                    for file in os.listdir(method_dir):
                        if layer1_pattern in file:
                            key = f"{method_type}_{algorithm}_layer1_{target}"
                            results[key] = read_performance_file(os.path.join(method_dir, file), metrics=all_metrics)
        else:
            for file in os.listdir(method_dir):
                target = extract_target_from_filename(file, target_type)
                if target in targets:
                    key = f"{method_type}_{target}"
                    filepath = os.path.join(method_dir, file)
                    results[key] = read_performance_file(filepath, metrics=all_metrics)
    
    # Process scratch results separately
    scratch_dir = os.path.join(base_dir, "scratch")
    if os.path.exists(scratch_dir):
        for file in os.listdir(scratch_dir):
            target = extract_target_from_filename(file,target_type)
            if target in targets:
                key = f"scratch_baseline"
                results[key] = read_performance_file(os.path.join(scratch_dir, file), metrics=all_metrics)
    
    # Organize data for plotting
    plot_data = {
        "methodology": [],  # Primary grouping (shallow, fine-tuned, freeze, scratch)
        "algorithm": [],    # Specific algorithm (svm, rf, gb) - only for shallow
        "target": [],       # Target dataset
        "layer": [],        # Layer info
    }
    
    for m in all_metrics:
        plot_data[m] = []
    
    for key, values in results.items():
        parts = key.split('_')
        
        if key.startswith("shallow_"):
            # Format: shallow_algorithm_layer_target
            methodology, algorithm, layer_info, target = parts[0], parts[1], f"Layer {parts[2][-1]}", parts[3]
        else:
            # Format: methodology_target
            methodology, target = parts[0], parts[1]
            algorithm = "N/A"
            layer_info = "N/A"
            
        plot_data["methodology"].append(methodology)
        plot_data["algorithm"].append(algorithm)
        plot_data["target"].append(target)
        plot_data["layer"].append(layer_info)
        
        for m in all_metrics:
            plot_data[m].append(values[m])
    
    # Plotting
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Group by Methodology, show targets within each group
    plt.subplot(2, 1, 1)
    
    # Use predefined method order
    methodology_list = [m for m in METHOD_ORDER if m in set(plot_data["methodology"])]
    targets_list = sorted(set(plot_data["target"]))
    
    # Colors for targets
    colors = sns.color_palette("Set2", len(targets_list))
    target_colors = dict(zip(targets_list, colors))
    
    # Set bar positions
    methodology_positions = np.arange(len(methodology_list))
    bar_width = 0.8 / len(targets_list)
    
    # Create labels for x-axis (show "baseline" for scratch)
    methodology_labels = ["baseline" if m == "scratch" else m for m in methodology_list]
    
    # Plot bars for each target within methodology groups
    for i, target in enumerate(targets_list):
        target_values = []
        target_positions = []
        for j, methodology in enumerate(methodology_list):
            
            # Find all entries for this methodology and target
            indices = [k for k in range(len(plot_data["methodology"])) 
                       if plot_data["methodology"][k] == methodology and plot_data["target"][k] == target]
            
            if indices:
                # Get the best result for this methodology/target combination
                best_idx = max(indices, key=lambda idx: plot_data[metric][idx])
                target_values.append(plot_data[metric][best_idx])
                
                # Position the bar
                if target == "baseline":
                    position = methodology_positions[j] + i * bar_width - (len(targets_list)) * bar_width / 2 + 5 * bar_width
                else:
                    position = methodology_positions[j] + i * bar_width - (len(targets_list)) * bar_width / 2 
                target_positions.append(position)
        # Plot the target bars
        plt.bar(target_positions, target_values, bar_width, label=target, color=target_colors[target],alpha = 0.7)

    plt.xlabel('Methodology')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} by Methodology and Target Dataset')
    adjust_positions = list(methodology_positions.copy())
    adjust_positions[0] = methodology_positions[0]+ 2 * bar_width
    plt.xticks(adjust_positions, methodology_labels)
    plt.legend(title = 'Source',loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=plot_data[metric][-1], color='gray', linestyle='--', linewidth=1.5, alpha=0.9)
    
    # Plot 2: Targets on x-axis, methodologies grouped within each target
    plt.subplot(2, 1, 2)
    
    # Set positions
    target_positions = np.array(np.arange(len(targets_list)))
    methodology_colors = sns.color_palette("Set2", len(methodology_list))
    methodology_color_map = dict(zip(methodology_list, methodology_colors))
    
    # Plot bars for each methodology within target groups
    for i, methodology in enumerate(methodology_list):
        methodology_values = []
        methodology_positions = []
        
        for j, target in enumerate(targets_list):
            # Find all entries for this methodology and target
            indices = [k for k in range(len(plot_data["methodology"]))
                       if plot_data["methodology"][k] == methodology and plot_data["target"][k] == target]
            
            if indices:
                # Get the best result for this methodology/target combination
                best_idx = max(indices, key=lambda idx: plot_data[metric][idx])
                methodology_values.append(plot_data[metric][best_idx])
                
                # Position the bar
                if methodology == "scratch":
                    position = target_positions[j] + i * bar_width - (len(methodology_list) * bar_width) / 2 + bar_width / 2 + 3 * bar_width
                else:
                    position = target_positions[j] + i * bar_width - (len(methodology_list) * bar_width) / 2 - bar_width * j
                methodology_positions.append(position)
        # Plot the methodology bars with "baseline" label for scratch
        label = "baseline" if methodology == "scratch" else methodology
        plt.bar(methodology_positions, methodology_values, bar_width, label=label, color=methodology_color_map[methodology],alpha=0.7)
    
    plt.xlabel('Target Datasets')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} by Target Dataset and Methodology')
    adjust_positions = list(target_positions.copy())
    adjust_positions[0] = target_positions[0]+ 1.5 * bar_width
    for i in range(1,len(adjust_positions)):
        adjust_positions[i] = adjust_positions[i] - i * bar_width
    plt.xticks(adjust_positions, targets_list)
    plt.legend(title = "Mode", loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=plot_data[metric][-1], color='gray', linestyle='--', linewidth=1.5, alpha=0.9)
    
    plt.tight_layout()
    filename = f"{target_type}_{dataset_name}_{metric}_by_methodology.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    # Create methodology-focused plots - show best performing algorithm for each methodology
    create_methodology_focused_plot(plot_data, target_type, dataset_name, output_dir, metric)
    
    # Detailed table
    print("\nDetailed Performance Results:")
    print("-" * 120)
    header = f"{'Methodology':<15} {'Algorithm':<15} {'Target':<15} {'Layer':<10}"
    for m in all_metrics:
        header += f" {m.upper():<10}"
    print(header)
    print("-" * 120)
    
    # Sort by methodology first, then target
    sorted_indices = sorted(range(len(plot_data["methodology"])), 
                          key=lambda i: (METHOD_ORDER.index(plot_data["methodology"][i]) 
                                        if plot_data["methodology"][i] in METHOD_ORDER else 999,
                                        plot_data["target"][i]))
    
    for idx in sorted_indices:
        row = f"{plot_data['methodology'][idx]:<15} {plot_data['algorithm'][idx]:<15} {plot_data['target'][idx]:<15} {plot_data['layer'][idx]:<10}"
        for m in all_metrics:
            row += f" {plot_data[m][idx]:<10.4f}"
        print(row)
    
    return results

def create_methodology_focused_plot(plot_data, target_type, dataset_name, output_dir, metric):
    """
    Create a plot that focuses on methodology comparison, aggregating the best results 
    for each methodology across targets.

    Args:
        plot_data (dict): Data organized for plotting
        target_type (str): Current target type being processed
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save visualizations
        metric (str): Metric to visualize
    """
    plt.figure(figsize=(12, 8))
    
    # Get unique methodologies in the preferred order
    methodology_list = [m for m in METHOD_ORDER if m in set(plot_data["methodology"])]
    targets_list = sorted(set(plot_data["target"]))
    
    # Calculate aggregate scores for each methodology
    methodology_scores = {}
    
    for methodology in methodology_list:
        # Get all scores for this methodology across targets
        scores = []
        for target in targets_list:
            indices = [i for i in range(len(plot_data["methodology"])) 
                      if plot_data["methodology"][i] == methodology and plot_data["target"][i] == target]
            
            if indices:
                # Take the best score for this methodology/target
                best_score = max(plot_data[metric][i] for i in indices)
                scores.append(best_score)
        
        if scores:
            # Calculate mean score across targets
            methodology_scores[methodology] = np.mean(scores)
    
    # Set positions for the bars
    positions = np.arange(len(methodology_list))
    
    # Create a color palette
    colors = sns.color_palette("Set2", len(methodology_list))
    
    # Plot the bars with modified labels
    bar_labels = ["baseline" if m == "scratch" else m for m in methodology_list]
    plt.bar(positions, [methodology_scores[m] for m in methodology_list], color=colors, alpha = 0.7)
    
    # Add labels and title
    plt.xlabel('Methodology')
    plt.ylabel(f'Mean {metric.upper()} across targets')
    plt.title(f'Comparison of Methodologies - {target_type} {dataset_name} - {metric.upper()}')
    plt.xticks(positions, bar_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate([methodology_scores[m] for m in methodology_list]):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    filename = f"{target_type}_{dataset_name}_{metric}_methodology_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved methodology comparison to {output_path}")
    plt.close()

def process_multiple_datasets(base_results_dir, datasets=None, targets=None, output_dir="visualization_results", metrics=None):
    """
    Process multiple datasets and generate visualizations for each metric.

    Args:
        base_results_dir (str): Base directory with results.
        datasets (list, optional): List of dataset names.
        targets (list, optional): List of target types.
        output_dir (str): Directory for outputs.
        metrics (list, optional): Metrics to process (default: ["auroc", "auprc"]).

    Returns:
        dict: Aggregated results.
    """
    if metrics is None:
        metrics = ["auroc", "auprc"]
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "aggregated_results.csv")
    with open(csv_path, "w") as f:
        header = "Dataset,TargetType,Target,Methodology,Algorithm,Layer"
        for m in metrics:
            header += f",{m.upper()}"
        f.write(header + "\n")
    
    all_results = {}
    targets = targets or [d for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d)) and d in ALL_TARGETS]
    
    for target_type in targets:
        target_dir = os.path.join(base_results_dir, target_type)
        if not os.path.isdir(target_dir):
            print(f"Warning: {target_dir} is not a directory. Skipping.")
            continue
        
        datasets_for_target = datasets or [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        for dataset in datasets_for_target:
            dataset_dir = os.path.join(target_dir, dataset)
            if not os.path.isdir(dataset_dir):
                print(f"Warning: {dataset_dir} is not a directory. Skipping.")
                continue
            
            print(f"\nProcessing dataset: {dataset} for target type: {target_type}")
            for metric in metrics:
                print(f"Generating visualization for metric: {metric}")
                results = visualize_set_results(dataset_dir, output_dir, metric, all_metrics=metrics)
                dataset_key = f"{target_type}_{dataset}_{metric}"
                all_results[dataset_key] = results
                
                with open(csv_path, "a") as f:
                    for key, values in results.items():
                        parts = key.split('_')
                        
                        if key.startswith("shallow_"):
                            # Format: shallow_algorithm_layer_target
                            methodology, algorithm, layer_info, target_name = parts[0], parts[1], f"Layer {parts[2][-1]}", parts[3]
                        else:
                            # Format: methodology_target
                            methodology, target_name = parts[0], parts[1]
                            algorithm = "N/A"
                            layer_info = "N/A"
                        
                        row = f"{dataset},{target_type},{target_name},{methodology},{algorithm},{layer_info}"
                        for m in metrics:
                            row += f",{values[m]}"
                        f.write(row + "\n")
    
    for metric in metrics:
        create_summary_visualization(all_results, output_dir, metric)
    
    return all_results

def create_summary_visualization(all_results, output_dir, metric="auroc"):
    """
    Create a summary visualization across all datasets, grouped by methodology.

    Args:
        all_results (dict): Aggregated results.
        output_dir (str): Directory for output.
        metric (str): Metric to visualize.
    """
    summary_data = {}
    relevant_results = {k: v for k, v in all_results.items() if metric.lower() in k.lower()}
    
    for dataset_key, results in relevant_results.items():
        target_type, dataset = dataset_key.split('_')[:2]
        dataset_id = f"{target_type}_{dataset}"
        
        # Group by methodology
        methodology_results = {methodology: [] for methodology in METHOD_ORDER}
        
        for key, values in results.items():
            # Extract methodology
            if key.startswith("shallow_"):
                methodology = "shallow"
            else:
                methodology = key.split('_')[0]
                
            if methodology in methodology_results:
                methodology_results[methodology].append(values[metric])
        
        # Calculate mean for each methodology, ignoring NaNs
        summary_data[dataset_id] = {}
        for methodology, values in methodology_results.items():
            if values and not np.all(np.isnan(values)):
                summary_data[dataset_id][methodology] = np.nanmean(values)
    
    plt.figure(figsize=(15, 10))
    
    datasets = list(summary_data.keys())
    methodology_list = [m for m in METHOD_ORDER if any(m in d for d in summary_data.values())]
    
    x = np.arange(len(datasets))
    width = 0.8 / len(methodology_list)
    
    # Plot bars for each methodology
    for i, methodology in enumerate(methodology_list):
        values = [summary_data[d].get(methodology, np.nan) for d in datasets]
        if not np.all(np.isnan(values)):
            positions = x + i * width - (len(methodology_list) * width) / 2 + width / 2
            plt.bar(positions, values, width, label=methodology)
    
    plt.xlabel('Datasets')
    plt.ylabel(f'Mean {metric.upper()}')
    plt.title(f'Comparison of Methodologies Across Datasets - {metric.upper()}')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend(title='Methodology')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"summary_{metric}_by_methodology.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved summary visualization to {output_path}")
    plt.close()

if __name__ == "__main__":
    process_multiple_datasets(
        base_results_dir="result_files",
        targets=["gpcr", "ionchannel", "kinase", "nuclearreceptor", "protease", "transporter"],
        datasets=["dataSubset500"],
        output_dir="visualization_results",
        metrics=["Accuracy", "MCC"]
    )