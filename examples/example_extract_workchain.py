#%%
import json
import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
# aiida profile load
aiida.load_profile()


#%%

# workchain_pk = 19232 # FeO
# workchain_pk = 24930 # CoO
# material_name = "CoO"
# workchain_pk = 25730 # CuO
# material_name = "CuO"
# workchain_pk = 36461 # VO
# material_name = "VO"
# workchain_pk = 59855 # UO2

# workchain_pk = 64716 # UO2
# material_name = "UO2"
material_name = "trial"
# workchain_pk = 74786 # UO2 second run with so(n) enabled
workchain_pk =  95201# UO2 second run with so(n) enabled
afm_chain = 64717
# feo_so_n_global_chain = 49395
# Create extractor with SO(N) decomposition enabled
extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

# Extract data from workchain
data = extractor.extract_from_workchain(workchain_pk)

# Save to JSON
extractor.save_to_json(data, f"{material_name}_scan_data_extractor_so_n.json")
# extractor.save_to_json(data, f"{material_name}_scan_data_extractor.json")

# Print summary
#%%
print(extractor.get_extraction_summary(data))

#%% 
#%%
import numpy as np
# Example: Analyze calculation 71782 and get regularization results
extractor = WorkchainExtractor(perform_so_n=True, sanity_check_reconstruct_rho=True, debug=True)
# calc_data = extractor.extract_single_calculation(71782)
calc_data = extractor.extract_single_calculation(64802)

# Get regularization summary and details
print(extractor.get_regularization_summary())
reg_details = extractor.get_regularization_details()
print(f"Regularization details: {reg_details}")


#manually check the data
#%%
occup_data = calc_data['output_atomic_occupations']['1']

recon_data = calc_data['so_n_decomposition']['atom_decompositions']['1']

down_matrix = np.array(recon_data['down_spin']['sanity_check']['reconstructed_density_matrix'])

with np.printoptions(precision=3, suppress=True):
    print("Original down spin occupation matrix:")
    print(np.array(occup_data['spin_data']['down']['occupation_matrix']))
    print("Reconstructed down spin occupation matrix:")
    print(down_matrix)
    print("Difference:")
    print(np.array(occup_data['spin_data']['down']['occupation_matrix']) - down_matrix)
#%%
