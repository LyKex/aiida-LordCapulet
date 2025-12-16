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
