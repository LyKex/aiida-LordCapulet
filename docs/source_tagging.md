# Workchain Data Extraction with Source Tagging

## Overview

Extracts calculations from `GlobalConstrainedSearchWorkChain` instances and tags them by source for separate plotting.

## Source Tags

- **`afm_workchain`**: Standard PW calculations from AFM scans (simple points)
- **`constrained_scan`**: ConstrainedPW calculations from OSCDFT scans (complex analysis)

## Usage

```python
from lordcapulet.utils.postprocessing.gather_workchain_data import (
    gather_workchain_data, 
    filter_calculations_by_source
)

# Extract all calculations
data = gather_workchain_data()

# Separate by source for plotting
afm_data = filter_calculations_by_source(data, 'afm_workchain')
constrained_data = filter_calculations_by_source(data, 'constrained_scan')

# Plot AFM as simple points, constrained as different markers
```

## Options

- `gather_workchain_data()` - Extract from all global workchains
- `gather_workchain_data(workchain_pk=12345)` - Extract from specific workchain
- `gather_workchain_data(max_results=10)` - Limit number processed
- `gather_workchain_data(output_filename="data.json")` - Save to file

## Output

Each calculation includes a `calculation_source` field for clear separation when plotting.