#%%

import json

import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData 
from lordcapulet.data_structures.databank import DataBank
# aiida profile load
aiida.load_profile()

# %%
material_name = "FeO"
json_filename = f"{material_name}_scan_data_extractor_redone.json"


databank = DataBank.from_json(json_filename)
# %%
databank.to_dataframe()


# %%
occ_matrix = databank.get_occ_data(0)

occ_matrix.get_occupation_matrix('1', 'up')


occ_matrix.get_trace('1', 'up')

#%%
# from already loaded databank, get the total magnetic moment for each entry
databank.get_electron_number()


# now reload the databank adding electron number and magnetic moment to each entry
databank = DataBank.from_json(json_filename, include_electron_number=True, include_moment=True)
# %%
databank.to_pytorch()
#%%
import numpy as np
from lordcapulet.data_structures.occupation_matrix import compute_occupation_distance
# get 2 different occupation matrices and compare their distance
occ_data1 = databank.get_occ_data(0)
occ_data2 = databank.get_occ_data(1)

zerodist = compute_occupation_distance(occ_data1, occ_data1)
dist = compute_occupation_distance(occ_data1, occ_data2)

with np.printoptions(precision=4, suppress=True):
    print(f"Distance between identical matrices: {zerodist:.4f}")
    print(f"Distance between different matrices: {dist:.4f}")


# %%
