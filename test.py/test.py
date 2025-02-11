import numpy as np
import matplotlib.pyplot as plt

# Function to parse the file and extract the data
def parse_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []
    except Exception as e:
        print(f"Error reading the file {filename}: {e}")
        return []

    data = []
    in_data_section = False

    for line in lines:
        # Check if we are in the [DATA] section of CDAT0
        if line.strip().startswith("[CDAT0"):
            in_data_section = True
            continue  # Skip the header line

        if in_data_section:
            # Split the line into x, y, and counts
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    x, y, counts = map(int, parts)
                    data.append((x, y, counts))
                except ValueError:
                    print(f"Error: Invalid data format in line: {line.strip()}")
            if line.strip().startswith("["):  # End of data section
                break

    return data

# Function to plot the dE/E final data
def plot_dE_E_final(data):
    # Convert data to numpy arrays
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    counts = np.array([d[2] for d in data])

    # Create a 2D histogram
    plt.figure(figsize=(10, 8))
    plt.hist2d(x, y, bins=(1024, 1024), weights=counts, cmap='viridis', cmin=1)
    plt.colorbar(label='Counts')
    plt.xlabel('E final')
    plt.ylabel('dE')
    plt.title('dE vs E final')
    plt.show()

# Main script
if __name__ == "__main__":
    filename = "test.txt"  # Replace with your file name
    data = parse_file(filename)
    plot_dE_E_final(data)