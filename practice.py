import csv

# Example initial data
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Los Angeles', 'Chicago']
}

# Define the CSV file name
csv_file = 'output.csv'

# Get the headers from the dictionary keys
headers = data.keys()

# Write the title and headers once
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Title: Example CSV File'])  # Adding title
    writer.writerow(headers)  # Adding headers

# Simulate adding data for each epoch
num_epochs = 3  # For example, 3 epochs

for epoch in range(num_epochs):
    # Update data (simulate new data for each epoch)
    data = {
        'name': [f'Alice_{epoch}', f'Bob_{epoch}', f'Charlie_{epoch}'],
        'age': [25 + epoch, 30 + epoch, 35 + epoch],
        'city': [f'New York_{epoch}', f'Los Angeles_{epoch}', f'Chicago_{epoch}']
    }
    
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the data rows iteratively
        for i in range(len(data['name'])):
            row = [data[key][i] for key in headers]
            writer.writerow(row)
    
    print(f'Data for epoch {epoch + 1} written to {csv_file} successfully.')

print(f'All data written to {csv_file} successfully.')
