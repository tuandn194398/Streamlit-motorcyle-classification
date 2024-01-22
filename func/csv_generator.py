import os
import csv

labels_folder = 'runs/classify/predict/labels'
labels_file = 'runs/detect/predict/labels/image0.txt'  # Replace with the correct path to your labels.txt

# Prepare the CSV file to write the merged data
def csv_generate():
    with open('annotations.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'class', 'x_center', 'y_center', 'width', 'height'])

        # Read labels.txt file
        with open(labels_file, 'r') as file:
            labels_data = [line.strip().split() for line in file.readlines()]

        print(f"Total labels: {len(labels_data)}")

        # Iterate over all the .txt files in the labels folder
        for file_name in os.listdir(labels_folder):
            if file_name.endswith('.txt') and file_name != 'labels.txt':
                # Extract the numeric part of the filename
                base_name = os.path.splitext(file_name)[0]
                numeric_part = ''.join(filter(str.isdigit, base_name))
                index = int(numeric_part)

                print(f"Processing {file_name}, index: {index}")

                # Construct the full path to the .txt file
                file_path = os.path.join(labels_folder, file_name)

                # Read the class from the .txt file (first row, second column)
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 1:
                        class_label = lines[0].strip().split()[1]

                        # Check if index is within the range of labels_data
                        if index < len(labels_data):
                            annotation = labels_data[index - 1]  # Adjust index to zero-based
                            csvwriter.writerow([f'{base_name}.jpg', class_label] + annotation[-4:])
                        else:
                            print(f"Index {index} out of range in labels.txt for file {file_name}")
