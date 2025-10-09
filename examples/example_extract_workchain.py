#!/usr/bin/env python3

"""
Simple example: Extract calculations from a workchain and save to JSON.

Usage: python example_extract_workchain.py
"""
#%%
import json
import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import gather_workchain_data
# aiida profile load
aiida.load_profile()


def extract_workchain_to_json(workchain_pk, output_file=None):
    """
    Extract all calculations from a workchain and save to JSON.
    
    Args:
        workchain_pk (int): Primary key of the workchain to process
        output_file (str, optional): Output JSON filename. If None, uses workchain_pk.json
        
    Returns:
        dict: Extracted data with calculations, metadata, and statistics
    """
    if output_file is None:
        output_file = f"workchain_{workchain_pk}_data.json"
    
    print(f"Extracting calculations from workchain {workchain_pk}...")
    
    # Extract data from the workchain
    data = gather_workchain_data(workchain_pk=workchain_pk, output_filename=output_file, debug=True)
    
    # Print summary
    print(f"\nExtraction complete!")
    
    # Debug: print available metadata keys
    print(f"Available metadata keys: {list(data['metadata'].keys())}")
    
    # Print workchain info if available
    if 'workchain_process_type' in data['metadata']:
        print(f"Workchain type: {data['metadata']['workchain_process_type']}")
    else:
        print(f"Workchain process type not available in metadata")
    
    print(f"Total calculations found: {data['metadata']['total_calculations_found']}")
    print(f"Extraction method: {data['metadata']['extraction_method']}")
    
    print(f"Source breakdown:")
    for source, count in data['statistics']['calculation_sources'].items():
        print(f"  {source}: {count}")
    print(f"Data saved to: {output_file}")
    
    return data
#%%

    # workchain_pk = 19232 # FeO
    # workchain_pk = 24930 # CoO
    # workchain_pk = 25730 # CuO
    # workchain_pk = 36461 # VO
    # workchain_pk = 59855 # UO2
# data = extract_workchain_to_json(49395)
# data = extract_workchain_to_json(24930, output_file = "CoO_scan_data.json")
# data = extract_workchain_to_json(19232, output_file = "FeO_scan_data.json")
data = extract_workchain_to_json(64716, output_file = "UO2_scan_data.json")
# %%
