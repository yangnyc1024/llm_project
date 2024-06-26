import json
import csv
import os


json_file_path = 'llm_project/sample.json'

csv_file_path = 'llm_project/sample.csv'

headers = ['product_id', 'name', 'category', 'price', 'stock', 'average_rating', 'number_of_reviews']

# Open the CSV file for writing

# Step 1: Open the JSON file and load the data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
    print(data)

# Step 2: Open a new CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    # If your data is a list of dictionaries, each dict can represent a row
    # And the dict keys represent the column names
    fieldnames = data.keys()
    print(fieldnames)

    # Step 3: Create a DictWriter object, passing the fieldnames
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Step 4: Write the header (column names)
    csv_writer.writeheader()

    # Step 5: Write the JSON data to the CSV file
    # for row in data:
    #     csv_writer.writerow(row)
    csv_writer.writerow(data)
