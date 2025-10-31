#!/usr/bin/env python3
"""
Simplified CDR Feature Extraction for Competition

Quick implementation focusing on the most critical CDR features
for antibody developability prediction.
"""

import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path

# Add workspace to path
sys.path.append('/workspaces/bitcore/workspace')
from bioinformatics.modules.cdr_extraction import CDRExtractor
from bioinformatics.feature_utils import (
    hydrophobicity_values,
    shannon_entropy,
    charge_totals,
    count_glycosylation_sites,
    FLEXIBLE_RESIDUES,
    AROMATIC_RESIDUES
)
from bioinformatics.provenance import record_provenance_event

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VH_PREFIXES = {'CDR1': 'cdr_h1', 'CDR2': 'cdr_h2', 'CDR3': 'cdr_h3'}
VL_PREFIXES = {'CDR1': 'cdr_l1', 'CDR2': 'cdr_l2', 'CDR3': 'cdr_l3'}

H3_CATEGORY_ENCODING = {
    'short': 0,
    'medium': 1,
    'long': 2,
    'ultralong': 3
}

def extract_basic_cdr_features(df):
    """Extract basic CDR features for all antibodies"""
    logger.info("Starting basic CDR feature extraction")
    
    extractor = CDRExtractor()
    results = []
    
    for idx, row in df.iterrows():
        antibody_id = row['antibody_id']
        vh_seq = row['vh_protein_sequence']
        vl_seq = row['vl_protein_sequence']
        
        try:
            # Extract CDR regions
            vh_cdrs = extractor.extract_cdr(vh_seq, 'H')
            vl_cdrs = extractor.extract_cdr(vl_seq, 'L')
            
            # Basic features
            features = {
                'antibody_id': antibody_id,
                # CDR lengths
                'cdr_h1_length': len(vh_cdrs['CDR1']),
                'cdr_h2_length': len(vh_cdrs['CDR2']),
                'cdr_h3_length': len(vh_cdrs['CDR3']),
                'cdr_l1_length': len(vl_cdrs['CDR1']),
                'cdr_l2_length': len(vl_cdrs['CDR2']),
                'cdr_l3_length': len(vl_cdrs['CDR3']),
                
                # CDR sequences (for analysis)
                'cdr_h1_seq': vh_cdrs['CDR1'],
                'cdr_h2_seq': vh_cdrs['CDR2'],
                'cdr_h3_seq': vh_cdrs['CDR3'],
                'cdr_l1_seq': vl_cdrs['CDR1'],
                'cdr_l2_seq': vl_cdrs['CDR2'],
                'cdr_l3_seq': vl_cdrs['CDR3'],
                
                # Total CDR length
                'total_cdr_length': sum(len(cdr) for cdr in list(vh_cdrs.values()) + list(vl_cdrs.values())),
                
                # CDR-H3 specific (most important)
                'h3_flexibility_score': calculate_h3_flexibility(vh_cdrs['CDR3']),
                'h3_category': categorize_h3_length(len(vh_cdrs['CDR3'])),
                
                'extraction_success': True
            }

            # Encode H3 category numerically to support modeling
            features['h3_category_code'] = H3_CATEGORY_ENCODING[features['h3_category']]
            
            # Detailed per-CDR composition metrics
            composition_metrics = derive_cdr_composition_metrics(vh_cdrs, vl_cdrs)
            features.update(composition_metrics)
            
            results.append(features)
            
        except Exception as e:
            logger.warning(f"Failed to extract CDRs for {antibody_id}: {str(e)}")
            results.append({
                'antibody_id': antibody_id,
                'extraction_success': False,
                'error': str(e)
            })
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} antibodies")
    
    return pd.DataFrame(results)

def calculate_h3_flexibility(h3_seq):
    """Simple H3 flexibility score based on amino acid properties"""
    if not h3_seq:
        return 0.0
    
    flexible_aa = set('GSTDNQ')  # Flexible amino acids
    rigid_aa = set('PWFY')       # Rigid amino acids
    
    flexible_count = sum(1 for aa in h3_seq if aa in flexible_aa)
    rigid_count = sum(1 for aa in h3_seq if aa in rigid_aa)
    
    # Simple scoring: +1 for flexible, -1 for rigid, 0 for others
    score = (flexible_count - rigid_count) / len(h3_seq)
    return score

def categorize_h3_length(length):
    """Categorize H3 length based on structural studies"""
    if length <= 8:
        return 'short'
    elif length <= 13:
        return 'medium'
    elif length <= 20:
        return 'long'
    else:
        return 'ultralong'

def derive_cdr_composition_metrics(vh_cdrs, vl_cdrs):
    """Compute detailed composition metrics for each CDR and aggregate statistics."""
    metrics = {}
    entropy_values = []
    hydrophobicity_means = []
    heavy_net_charge = 0.0
    light_net_charge = 0.0
    total_net_charge = 0.0
    total_glyco_sites = 0
    total_positive = 0.0
    total_negative = 0.0
    total_length = 0
    flexible_fraction_sum = 0.0
    proline_fraction_sum = 0.0
    cdr_count = 0

    # Process heavy chain CDRs
    for region, seq in vh_cdrs.items():
        prefix = VH_PREFIXES[region]
        region_metrics = compute_cdr_metrics(seq, prefix)
        metrics.update(region_metrics)
        entropy_values.append(region_metrics[f'{prefix}_sequence_entropy'])
        hydrophobicity_means.append(region_metrics[f'{prefix}_hydrophobicity_mean'])
        heavy_net_charge += region_metrics[f'{prefix}_net_charge']
        total_net_charge += region_metrics[f'{prefix}_net_charge']
        total_glyco_sites += region_metrics[f'{prefix}_glycosylation_sites']
        total_positive += region_metrics[f'{prefix}_positive_charge_count']
        total_negative += region_metrics[f'{prefix}_negative_charge_count']
        total_length += len(seq)
        flexible_fraction_sum += region_metrics[f'{prefix}_flexibility_fraction']
        proline_fraction_sum += region_metrics[f'{prefix}_proline_fraction']
        cdr_count += 1
    
    # Process light chain CDRs
    for region, seq in vl_cdrs.items():
        prefix = VL_PREFIXES[region]
        region_metrics = compute_cdr_metrics(seq, prefix)
        metrics.update(region_metrics)
        entropy_values.append(region_metrics[f'{prefix}_sequence_entropy'])
        hydrophobicity_means.append(region_metrics[f'{prefix}_hydrophobicity_mean'])
        light_net_charge += region_metrics[f'{prefix}_net_charge']
        total_net_charge += region_metrics[f'{prefix}_net_charge']
        total_glyco_sites += region_metrics[f'{prefix}_glycosylation_sites']
        total_positive += region_metrics[f'{prefix}_positive_charge_count']
        total_negative += region_metrics[f'{prefix}_negative_charge_count']
        total_length += len(seq)
        flexible_fraction_sum += region_metrics[f'{prefix}_flexibility_fraction']
        proline_fraction_sum += region_metrics[f'{prefix}_proline_fraction']
        cdr_count += 1

    # Aggregate statistics across all CDRs (protect against division by zero)
    metrics['cdr_total_net_charge'] = total_net_charge
    metrics['cdr_total_glycosylation_sites'] = total_glyco_sites
    metrics['cdr_positive_charge_total'] = total_positive
    metrics['cdr_negative_charge_total'] = total_negative
    metrics['cdr_heavy_vs_light_charge_gap'] = abs(heavy_net_charge - light_net_charge)
    symmetry_denominator = max(abs(total_net_charge), 1.0)
    metrics['cdr_charge_symmetry_ratio'] = abs(heavy_net_charge - light_net_charge) / symmetry_denominator
    metrics['cdr_average_entropy'] = float(np.mean(entropy_values)) if entropy_values else 0.0
    metrics['cdr_entropy_std'] = float(np.std(entropy_values)) if entropy_values else 0.0
    metrics['cdr_max_hydrophobicity_mean'] = float(np.max(hydrophobicity_means)) if hydrophobicity_means else 0.0
    metrics['cdr_min_hydrophobicity_mean'] = float(np.min(hydrophobicity_means)) if hydrophobicity_means else 0.0
    metrics['cdr_average_flexibility_fraction'] = (
        flexible_fraction_sum / cdr_count if cdr_count else 0.0
    )
    metrics['cdr_average_proline_fraction'] = (
        proline_fraction_sum / cdr_count if cdr_count else 0.0
    )
    metrics['cdr_global_charge_density'] = (
        total_net_charge / total_length if total_length else 0.0
    )
    metrics['h3_glycosylation_presence'] = int(metrics.get('cdr_h3_glycosylation_sites', 0) > 0)
    metrics['h3_positive_charge_density'] = metrics.get('cdr_h3_positive_charge_density', 0.0)
    metrics['h3_negative_charge_density'] = metrics.get('cdr_h3_negative_charge_density', 0.0)
    metrics['h3_entropy'] = metrics.get('cdr_h3_sequence_entropy', 0.0)

    return metrics

def compute_cdr_metrics(sequence, prefix):
    """Compute physicochemical metrics for a single CDR sequence."""
    metrics = {}
    if not sequence:
        metrics[f'{prefix}_hydrophobicity_mean'] = 0.0
        metrics[f'{prefix}_aromatic_fraction'] = 0.0
        metrics[f'{prefix}_positive_charge_density'] = 0.0
        metrics[f'{prefix}_negative_charge_density'] = 0.0
        metrics[f'{prefix}_net_charge'] = 0.0
        metrics[f'{prefix}_glycosylation_sites'] = 0
        metrics[f'{prefix}_flexibility_fraction'] = 0.0
        metrics[f'{prefix}_proline_fraction'] = 0.0
        metrics[f'{prefix}_sequence_entropy'] = 0.0
        metrics[f'{prefix}_positive_charge_count'] = 0.0
        metrics[f'{prefix}_negative_charge_count'] = 0.0
        return metrics

    length = len(sequence)
    hydrophobic_values = hydrophobicity_values(sequence)
    hydrophobicity_mean = float(np.mean(hydrophobic_values)) if hydrophobic_values else 0.0
    aromatic_fraction = sum(1 for aa in sequence if aa in AROMATIC_RESIDUES) / length
    flexible_fraction = sum(1 for aa in sequence if aa in FLEXIBLE_RESIDUES) / length
    proline_fraction = sequence.count('P') / length
    glyco_sites = count_glycosylation_sites(sequence)

    charges = charge_totals(sequence)
    positive_charge = charges['positive']
    negative_charge = charges['negative']
    net_charge = charges['net']
    positive_density = positive_charge / length
    negative_density = negative_charge / length

    # Shannon entropy of amino-acid composition
    entropy = shannon_entropy(sequence)

    metrics.update({
        f'{prefix}_hydrophobicity_mean': hydrophobicity_mean,
        f'{prefix}_aromatic_fraction': aromatic_fraction,
        f'{prefix}_positive_charge_density': positive_density,
        f'{prefix}_negative_charge_density': negative_density,
        f'{prefix}_net_charge': net_charge,
        f'{prefix}_glycosylation_sites': glyco_sites,
        f'{prefix}_flexibility_fraction': flexible_fraction,
        f'{prefix}_proline_fraction': proline_fraction,
        f'{prefix}_sequence_entropy': entropy,
        f'{prefix}_positive_charge_count': positive_charge,
        f'{prefix}_negative_charge_count': negative_charge
    })

    return metrics

def main():
    """Main execution"""
    logger.info("Starting simplified CDR feature extraction")
    
    # Load data
    input_path = Path("/workspaces/bitcore/workspace/data/sequences/GDPa1_v1.2_sequences.csv")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} antibodies")
    
    # Extract features
    features_df = extract_basic_cdr_features(df)
    
    # Save results
    output_path = Path("/workspaces/bitcore/workspace/data/features/cdr_features_basic.csv")
    output_path.parent.mkdir(exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    # Summary
    success_count = features_df['extraction_success'].sum()
    logger.info(f"CDR extraction complete: {success_count}/{len(df)} successful")
    logger.info(f"Features saved to {output_path}")
    
    # Quick analysis
    if success_count > 0:
        successful_df = features_df[features_df['extraction_success']]
        logger.info(f"H3 length range: {successful_df['cdr_h3_length'].min()} - {successful_df['cdr_h3_length'].max()}")
        logger.info(f"Average total CDR length: {successful_df['total_cdr_length'].mean():.1f}")
        
        h3_categories = successful_df['h3_category'].value_counts()
        logger.info(f"H3 categories: {dict(h3_categories)}")
    
    print(f"CDR feature extraction completed: {output_path}")
    provenance_path = record_provenance_event(
        event_name="cdr_features_basic",
        inputs=[input_path],
        outputs=[output_path],
        metadata={
            "antibodies": int(len(df)),
            "successful_extractions": int(features_df['extraction_success'].sum()),
            "feature_columns": int(len(features_df.columns)),
        },
    )
    logger.info(f"Provenance log saved to {provenance_path}")
    print(f"- Provenance: {provenance_path}")

    return features_df

if __name__ == "__main__":
    main()