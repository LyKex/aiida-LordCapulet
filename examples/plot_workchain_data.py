#%%
# Import required libraries
import json
import pandas as pd
import matplotlib.pyplot as plt


#%%
# Configuration - MODIFY THESE VALUES

# json_file = "workchain_24930_data.json"  # Replace with your JSON file
# material_name = "Material_24930"          # Replace with your material name
# material = "FeO"
material_name = "FeO"

json_file = f"{material_name}_scan_data.json"  # Replace with your JSON file
output_dir = "."                          # Directory to save plots and CSV

print(f"Analyzing workchain data from: {json_file}")
print(f"Material: {material_name}")
print("-" * 50)

#%%
# Load and extract magnetic moments and energy data
with open(json_file, 'r') as f:
    data = json.load(f)

results = []

# Iterate through all calculations
for calc_id, calc_data in data['calculations'].items():
    # Skip calculations without proper data
    if not isinstance(calc_data, dict):
        continue
        
    # Extract source tag (afm_workchain vs constrained_scan)
    source = calc_data.get('calculation_source', 'unknown')
    
    # Extract total energy from output_parameters
    total_energy = None
    if 'output_parameters' in calc_data and calc_data['output_parameters']:
        if isinstance(calc_data['output_parameters'], dict):
            total_energy = calc_data['output_parameters'].get('energy', None)
    
    # Extract magnetic moments from output_atomic_occupations
    magnetic_moments = []
    if 'output_atomic_occupations' in calc_data and calc_data['output_atomic_occupations']:
        atomic_occupations = calc_data['output_atomic_occupations']
        
        if isinstance(atomic_occupations, dict):
            for atom_id, atom_data in atomic_occupations.items():
                if isinstance(atom_data, dict) and 'magnetic_moment' in atom_data:
                    magnetic_moments.append(atom_data['magnetic_moment'])
    
    # Calculate average magnetic moment per atom if we have data
    avg_magnetic_moment = sum(magnetic_moments) / len(magnetic_moments) if magnetic_moments else None
    
    results.append({
        'calculation_id': calc_id,
        'source': source,
        'total_energy_eV': total_energy,
        'magnetic_moment_per_atom_Bohr': avg_magnetic_moment,
        'number_of_atoms': len(magnetic_moments) if magnetic_moments else 0,
        'individual_magnetic_moments': magnetic_moments
    })

print(f"Raw data extracted: {len(results)} calculations found")

#%%
# Create DataFrame and add total magnetic moment column
df = pd.DataFrame(results)

# Calculate total magnetic moment from individual moments
def calculate_total_magnetic_moment(magnetic_moments_list):
    if isinstance(magnetic_moments_list, list) and len(magnetic_moments_list) >= 2:
        return sum(magnetic_moments_list)
    elif isinstance(magnetic_moments_list, list) and len(magnetic_moments_list) == 1:
        return magnetic_moments_list[0]
    else:
        return 0

df['total_magnetic_moment_Bohr'] = df['individual_magnetic_moments'].apply(calculate_total_magnetic_moment)

# Filter out calculations without proper data
df_valid = df[df['total_energy_eV'].notna() & df['magnetic_moment_per_atom_Bohr'].notna()]

print(f"Valid calculations with energy and magnetic data: {len(df_valid)}")

#%%
# Display summary statistics
source_counts = df_valid['source'].value_counts()
print(f"Source breakdown:")
for source, count in source_counts.items():
    print(f"  {source}: {count}")

print(f"\nSummary Statistics:")
print(f"Energy range: {df_valid['total_energy_eV'].min():.3f} to {df_valid['total_energy_eV'].max():.3f} eV")
print(f"Energy spread: {df_valid['total_energy_eV'].max() - df_valid['total_energy_eV'].min():.3f} eV")
print(f"Magnetic moment per atom range: {df_valid['magnetic_moment_per_atom_Bohr'].min():.3f} to {df_valid['magnetic_moment_per_atom_Bohr'].max():.3f} Bohr magnetons")
print(f"Total magnetic moment range: {df_valid['total_magnetic_moment_Bohr'].min():.3f} to {df_valid['total_magnetic_moment_Bohr'].max():.3f} Bohr magnetons")
print(f"Average magnetic moment per atom: {df_valid['magnetic_moment_per_atom_Bohr'].mean():.3f} Bohr magnetons")

# Show first few rows
print(f"\nFirst 10 calculations:")
display_cols = ['calculation_id', 'source', 'total_energy_eV', 'magnetic_moment_per_atom_Bohr', 'total_magnetic_moment_Bohr']
print(df_valid[display_cols].head(10).to_string(index=False))

#%%
# Set up plotting parameters
plt.rcParams.update({'font.size': 16})

# Separate AFM and constrained calculations
df_afm = df_valid[df_valid['source'] == 'afm_workchain']
df_constrained = df_valid[df_valid['source'] == 'constrained_scan']

print(f"AFM calculations: {len(df_afm)}")
print(f"Constrained calculations: {len(df_constrained)}")

#%%
# Create energy landscape scatter plot
plt.figure(figsize=(12, 8))

# Plot constrained calculations in blue (smaller, more transparent)
if len(df_constrained) > 0:
    plt.scatter(df_constrained['total_magnetic_moment_Bohr'], 
               df_constrained['total_energy_eV'] - df_valid['total_energy_eV'].min(),
               c='blue', edgecolor='darkblue', s=40, alpha=0.5,
               label=f'Constrained scan ({len(df_constrained)} calcs)')

# Plot AFM calculations in red (larger, fully opaque)
if len(df_afm) > 0:
    plt.scatter(df_afm['total_magnetic_moment_Bohr'], 
               df_afm['total_energy_eV'] - df_valid['total_energy_eV'].min(),
               c='red', edgecolor='darkred', s=120, alpha=1.0,
               label=f'AFM workchain ({len(df_afm)} calcs)')

plt.title(f'{material_name} Energy Landscape', fontsize=24)
plt.xlabel('Total Magnetic Moment (Bohr magnetons)', fontsize=18)
plt.ylabel('Relative Energy (eV)', fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

# Save plot
landscape_filename = f'{output_dir}/{material_name}_magnetic_moment_vs_energy.png'
plt.savefig(landscape_filename, dpi=300, bbox_inches='tight')
print(f"Energy landscape plot saved to: {landscape_filename}")
plt.show()

#%%
# Create energy distribution histogram
plt.figure(figsize=(12, 8))

# Plot histogram with source separation
if len(df_afm) > 0 and len(df_constrained) > 0:
    afm_energies = df_afm['total_energy_eV'] - df_valid['total_energy_eV'].min()
    constrained_energies = df_constrained['total_energy_eV'] - df_valid['total_energy_eV'].min()
    
    plt.hist([afm_energies, constrained_energies], 
            bins=50, color=['red', 'blue'], alpha=0.7, 
            label=['AFM workchain', 'Constrained scan'],
            edgecolor='black', linewidth=0.5)
    plt.legend(fontsize=14)
else:
    # If only one type of calculation
    relative_energies = df_valid['total_energy_eV'] - df_valid['total_energy_eV'].min()
    plt.hist(relative_energies, bins=50, color='blue', 
            edgecolor='black', alpha=0.7)

plt.title(f'{material_name} Energy Distribution', fontsize=24)
plt.xlabel('Relative Energy (eV)', fontsize=18)  
plt.ylabel('Frequency', fontsize=18)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save plot
histogram_filename = f'{output_dir}/{material_name}_energy_histogram.png'
plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
print(f"Energy histogram saved to: {histogram_filename}")
plt.show()

#%%
# Final summary
print(f"\nAnalysis complete!")
print(f"Files created:")
print(f"  - {landscape_filename}")
print(f"  - {histogram_filename}")
print(f"\nDataFrame 'df_valid' contains {len(df_valid)} calculations and is ready for further analysis.")