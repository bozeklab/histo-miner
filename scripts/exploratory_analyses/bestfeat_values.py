import os
import json
import pandas as pd
import yaml
from attrdictionary import AttrDict as attributedict


#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(config)
analyser_output = confighm.paths.folders.tissue_analyser_output
pathtosavefolder = confighm.paths.folders.visualizations


#############################################################
## Generate table of values 
#############################################################


# List to hold the data for each file
data_list = []

# Iterate over all files in the folder
for filename in os.listdir(analyser_output):
    if filename.endswith('.json'):
        file_path = os.path.join(analyser_output, filename)
        print(f"Processing file: {filename}")
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            try:
	            # Extract the required data using Option 3 formatting
	            areas_mean_inside = (
	                data['CalculationsMorphinsideTumor']['Morphology_of_cells_in_Tumor_Regions']
	                    ['Morphology_insideTumor']['Granulocyte']['areas_mean']
	            )

	            areas_median_inside = (
	                data['CalculationsMorphinsideTumor']['Morphology_of_cells_in_Tumor_Regions']
	                    ['Morphology_insideTumor']['Granulocyte']['areas_median']
	            )

	            areas_mean_vicinity = (
	                data['CalculationsMorphinsideTumor']['Morphology_of_cells_in_Tumor_Regions']
	                    ['Morphology_insideTumorVicinity']['Granulocyte']['areas_mean']
	            )

	            areas_median_vicinity = (
	                data['CalculationsMorphinsideTumor']['Morphology_of_cells_in_Tumor_Regions']
	                    ['Morphology_insideTumorVicinity']['Granulocyte']['areas_median']
	            )

	            lymphocytes_percentage = (
	                data['CalculationsforWSI']['Pourcentages_of_cell_types_in_WSI']
	                ['Lymphocytes_Pourcentage']
	            )
	            granulocytes_percentage = (
	                data['CalculationsforWSI']['Pourcentages_of_cell_types_in_WSI']
	                ['Granulocytes_Pourcentage']
	            )


	            # Simplify the filename (optional)
	            simplified_filename = os.path.splitext(filename)[0]

	            # Append the extracted data to the list
	            data_list.append({
	                'Filename': simplified_filename,
	                'Areas Mean Inside Tumor': areas_mean_inside,
	                'Areas Median Inside Tumor': areas_median_inside,
	                'Areas Mean Tumor Vicinity': areas_mean_vicinity,
	                'Areas Median Tumor Vicinity': areas_median_vicinity,
	                'Lymphocyte Percentage': lymphocytes_percentage,
	                'Granulocyte Percentage': granulocytes_percentage,	  
	            })
            except KeyError as e:
                print(f"KeyError in file {filename}: {e}")
                continue

# Create a DataFrame from the list
df = pd.DataFrame(data_list)

# Set 'Filename' as the index if desired
df.set_index('Filename', inplace=True)

# Save the DataFrame to a CSV file
output_file = pathtosavefolder + 'extracted_data.csv'
df.to_csv(output_file)

print(f"Data extraction complete. The results are saved in {output_file}.")

					
				



                    





