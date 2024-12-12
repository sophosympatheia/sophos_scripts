# This script enables a comparison of the similarity between two exl2 measurement files produced from LLMs with the same underlying architecture.
# This was a quick and dirty job using Claude AI. It could probably be improved.

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_accuracy_differences(data1, data2):
    differences = {}
    
    # Get all layers from first file's measurement
    measurements1 = data1.get('measurement', {})
    measurements2 = data2.get('measurement', {})
    
    # Combine all layer names from both files
    all_layers = set(measurements1.keys()) | set(measurements2.keys())
    
    for layer_name in all_layers:
        # Get measurements from both files, defaulting to empty list if missing
        layer_measurements1 = measurements1.get(layer_name, []) or []
        layer_measurements2 = measurements2.get(layer_name, []) or []
        
        # Skip if either measurement list is empty
        if not layer_measurements1 or not layer_measurements2:
            print(f"Warning: Missing measurements for layer {layer_name} in one or both files")
            differences[layer_name] = 0
            continue
        
        try:
            # Create dictionaries mapping total_bits to accuracy for both files
            bits_to_acc1 = {m['total_bits']: m['accuracy'] for m in layer_measurements1 if 'total_bits' in m and 'accuracy' in m}
            bits_to_acc2 = {m['total_bits']: m['accuracy'] for m in layer_measurements2 if 'total_bits' in m and 'accuracy' in m}
            
            layer_diffs = []
            # Calculate differences for matching total_bits
            for total_bits in bits_to_acc1:
                if total_bits in bits_to_acc2:
                    acc1 = bits_to_acc1[total_bits]
                    acc2 = bits_to_acc2[total_bits]
                    # Calculate percentage difference
                    diff_percent = abs(acc1 - acc2) * 100 # We're going to express the values as "1%" instead of "0.01" for readability
                    layer_diffs.append(diff_percent)
            
            # Calculate average difference for this layer
            if layer_diffs:
                differences[layer_name] = sum(layer_diffs) / len(layer_diffs)
            else:
                print(f"Warning: No matching total_bits found for layer {layer_name}")
                differences[layer_name] = 0
                
        except Exception as e:
            print(f"Error processing layer {layer_name}: {str(e)}")
            differences[layer_name] = 0
            
    return differences

def plot_differences(differences):
    # Create DataFrame for plotting
    df = pd.DataFrame(list(differences.items()), columns=['Layer', '% Difference'])
    
    # Convert any list values to their average
    df['% Difference'] = df['% Difference'].apply(lambda x: x if not isinstance(x, list) else sum(x)/len(x))
    
    # Sort the DataFrame by layer name and number
    def get_layer_num(layer_name):
        try:
            # Extract numbers from the layer name
            nums = [int(s) for s in layer_name.split('.') if s.isdigit()]
            return nums[0] if nums else float('inf')  # Put layers without numbers at the end
        except:
            return float('inf')
    
    # Sort by the extracted number
    df = df.sort_values(by='Layer', key=lambda x: x.map(get_layer_num))
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(df)), df['% Difference'])
    plt.xticks(range(len(df)), df['Layer'], rotation=45, ha='right')
    plt.xlabel('Model Layer')
    plt.ylabel('Average Accuracy Difference (%)')
    plt.title('Accuracy Differences Between Files by Layer')
    plt.tight_layout()
    
    return df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare accuracy measurements between two ExLlamaV2 measurement JSON files')
    parser.add_argument('file1', help='Path to first measurement JSON file')
    parser.add_argument('file2', help='Path to second measurement JSON file')
    args = parser.parse_args()
    
    # Load the JSON files
    print(f"Loading files: {args.file1} and {args.file2}")
    data1 = load_json_file(args.file1)
    data2 = load_json_file(args.file2)
    
    # Calculate differences
    print("Calculating differences...")
    differences = calculate_accuracy_differences(data1, data2)
    
    # Create plot and get DataFrame
    print("Creating visualization...")
    df = plot_differences(differences)
    
    # Print table
    print("\nAccuracy Differences by Layer:")
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False))
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()