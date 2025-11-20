#!/bin/bash

# Get the directory where the script is located (not where it's called from)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define dataset directory relative to the script location
DATASET_DIR="$SCRIPT_DIR/../datasets"

# Check and create dataset directory
if [ ! -d "$DATASET_DIR" ]; then
    echo "Creating dataset directory: $DATASET_DIR"
    mkdir -p "$DATASET_DIR"
fi

# Define dataset URL associative array
declare -A dataset_urls
# dataset_urls["com-Friendster"]="https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
dataset_urls["Twitter-higgs"]="https://snap.stanford.edu/data/higgs-social_network.edgelist.gz"
dataset_urls["Facebook"]="https://snap.stanford.edu/data/facebook_combined.txt.gz"
dataset_urls["soc-Orkut-dir"]="https://nrvis.com/download/data/soc/soc-orkut-dir.zip"
# dataset_urls["soc-Sinaweibo"]="https://nrvis.com/download/data/soc/soc-sinaweibo.zip"
# dataset_urls["web-ClueWeb09-50m"]="https://nrvis.com/download/data/massive/web-ClueWeb09-50m.zip"
# dataset_urls["Wiki-link"]="https://nrvis.com/download/data/massive/web-wikipedia_link_en13-all.zip"
dataset_urls["Dogster"]="http://konect.cc/files/download.tsv.petster-dog-friend.tar.bz2"
dataset_urls["StackOverflow"]="http://konect.cc/files/download.tsv.sx-stackoverflow.tar.bz2"
# dataset_urls["Flickr"]="http://konect.cc/files/download.tsv.flickrEdges.tar.bz2"
# dataset_urls["com-Orkut"]="http://konect.cc/files/download.tsv.orkut-links.tar.bz2"
# dataset_urls["Wikipedia-link-en"]="http://konect.cc/files/download.tsv.wikipedia_link_en.tar.bz2"

# Process each dataset
for dataset_name in "${!dataset_urls[@]}"; do
    output_file="$DATASET_DIR/$dataset_name.txt"
    url=${dataset_urls[$dataset_name]}
    filename=$(basename "$url")
    temp_file="$DATASET_DIR/$filename"
    
    # Check if the file with dataset name exists
    if [ -f "$output_file" ]; then
        echo "$dataset_name: dataset exists"
    else
        # Clean up any existing temporary file first
        if [ -f "$temp_file" ]; then
            rm -f "$temp_file"
        fi
        
        # Download file to dataset directory
        cd "$DATASET_DIR" > /dev/null
        wget -q "$url" -O "$temp_file"
        
        if [ $? -eq 0 ]; then
            # Process extraction based on file extension
            case "$filename" in
                *.gz)
                    # For .gz files, use zcat to extract to dataset_name.txt
                    zcat "$temp_file" > "$output_file" 2>/dev/null
                    ;;
                *.zip)
                    # Extract to temporary directory and find main data file
                    temp_dir=$(mktemp -d)
                    unzip -q "$temp_file" -d "$temp_dir" > /dev/null
                    # Try to find the most relevant file
                    data_file=$(find "$temp_dir" -type f -exec ls -s {} \; | sort -nr | head -1 | awk '{print $2}')
                    if [ -n "$data_file" ]; then
                        mv "$data_file" "$output_file"
                    else
                        # Fallback to any text file or first found file
                        data_file=$(find "$temp_dir" -name "*.txt" -o -name "*.mtx" -o -name "*.edgelist" | head -1)
                        if [ -n "$data_file" ]; then
                            mv "$data_file" "$output_file"
                        else
                            data_file=$(find "$temp_dir" -type f | head -1)
                            if [ -n "$data_file" ]; then
                                mv "$data_file" "$output_file"
                            fi
                        fi
                    fi
                    rm -rf "$temp_dir"
                    ;;
                *.tar.bz2)
                    # Extract to temporary directory and find main data file
                    temp_dir=$(mktemp -d)
                    tar -xjf "$temp_file" -C "$temp_dir" > /dev/null
                    # Try to find the most relevant file
                    data_file=$(find "$temp_dir" -type f -exec ls -s {} \; | sort -nr | head -1 | awk '{print $2}')
                    if [ -n "$data_file" ]; then
                        mv "$data_file" "$output_file"
                    else
                        # Fallback to any text file or first found file
                        data_file=$(find "$temp_dir" -name "*.txt" -o -name "*.mtx" -o -name "*.edgelist" | head -1)
                        if [ -n "$data_file" ]; then
                            mv "$data_file" "$output_file"
                        else
                            data_file=$(find "$temp_dir" -type f | head -1)
                            if [ -n "$data_file" ]; then
                                mv "$data_file" "$output_file"
                            fi
                        fi
                    fi
                    rm -rf "$temp_dir"
                    ;;
                *)
                    # For other formats, just rename
                    mv "$temp_file" "$output_file"
                    ;;
            esac
            
            # Check if extraction was successful
            if [ -f "$output_file" ]; then
                # Clean up temporary file
                rm -f "$temp_file"
                echo "$dataset_name: downloaded"
            else
                # Clean up and report error
                rm -f "$temp_file"
                echo "$dataset_name: download or extraction failed"
            fi
        else
            # Clean up and report download error
            rm -f "$temp_file"
            echo "$dataset_name: download error"
        fi
        
        # Return to original directory
        cd - > /dev/null
    fi
done

echo ""
echo "Dataset check and download completed!"