#!/usr/bin/env python3
"""
=============================================================
Appian ML Pipeline - Complete End-to-End Training & Inference
=============================================================

One command to rule them all:
    python pipeline.py --all

Or run individual steps:
    python pipeline.py --generate     # Data generation
    python pipeline.py --train        # Train all models
    python pipeline.py --threshold    # Threshold analysis
    python pipeline.py --inference    # Test inference
    python pipeline.py --export       # Export to weights folder
"""

import argparse
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Pipeline stages
def run_data_generation(num_cases: int = 10000):
    """Stage 1: Generate training data with all variability factors"""
    print("\n" + "=" * 60)
    print("🏭 STAGE 1: DATA GENERATION")
    print("=" * 60)
    
    from data_generation import generate_training_data
    cases, events = generate_training_data(num_cases)
    
    return True


def run_training():
    """Stage 2: Train all 3 XGBoost models"""
    print("\n" + "=" * 60)
    print("🎓 STAGE 2: MODEL TRAINING")
    print("=" * 60)
    
    from train_all_models import ModelTrainingOrchestrator
    
    orchestrator = ModelTrainingOrchestrator()
    # Data already generated, load it
    import pandas as pd
    cases = pd.read_csv('cases_table.csv')
    events = pd.read_csv('events_log.csv')
    
    # Calculate remaining time if not present
    if 'will_breach_sla' not in events.columns:
        from data_generation import calculate_remaining_time
        events = calculate_remaining_time(events)
        events.to_csv('events_log.csv', index=False)
    
    # Train models
    orchestrator.train_duration_model(events, cases)
    orchestrator.train_routing_model(events, cases)
    orchestrator.train_sla_model(events, cases)
    orchestrator.print_training_summary()
    
    return True


def run_threshold_analysis():
    """Stage 3: Analyze optimal thresholds"""
    print("\n" + "=" * 60)
    print("📊 STAGE 3: THRESHOLD ANALYSIS")
    print("=" * 60)
    
    from threshold_analysis import run_threshold_analysis
    results = run_threshold_analysis()
    
    # Save recommended threshold
    with open('optimal_threshold.txt', 'w') as f:
        f.write(f"{results['recommended_threshold']:.4f}")
    
    print(f"\n💾 Saved optimal threshold: {results['recommended_threshold']:.2f}")
    return True


def run_inference_test():
    """Stage 4: Test inference with sample data"""
    print("\n" + "=" * 60)
    print("🔮 STAGE 4: INFERENCE TEST")
    print("=" * 60)
    
    import pandas as pd
    from multi_model_inference import MultiModelInferenceEngine
    
    # Load data
    events = pd.read_csv('events_log.csv')
    cases = pd.read_csv('cases_table.csv')
    
    # Initialize engine
    engine = MultiModelInferenceEngine()
    
    # Test with 5 sample events
    sample_events = events[events['status'] == 'ASSIGNED'].head(5)
    
    print("\n📋 Testing predictions on 5 sample events:")
    for idx, event in sample_events.iterrows():
        case_id = event['case_id']
        case_data = cases[cases['case_id'] == case_id].iloc[0] if len(cases[cases['case_id'] == case_id]) > 0 else None
        
        if case_data is not None:
            pred = engine.predict_comprehensive(event.to_dict(), case_data.to_dict())
            print(f"\n   Case: {pred.case_id}")
            print(f"   Activity: {pred.current_activity}")
            print(f"   Duration: {pred.predicted_duration_hours:.2f}h")
            print(f"   Next: {pred.next_activity}")
            print(f"   SLA Risk: {pred.sla_risk_level} ({pred.sla_breach_probability:.1%})")
    
    return True


def run_export(dest_dir: str = None):
    """Stage 5: Export models to weights folder"""
    print("\n" + "=" * 60)
    print("📦 STAGE 5: EXPORT MODELS")
    print("=" * 60)
    
    if dest_dir is None:
        dest_dir = os.path.abspath('../weights/xgboost/v1')
    
    os.makedirs(dest_dir, exist_ok=True)
    
    models = [
        ('duration_model.json', 'duration_model.json'),
        ('routing_model.json', 'routing_model.json'),
        ('sla_model.json', 'sla_model.json')
    ]
    
    for src, dst in models:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest_dir, dst))
            print(f"   ✅ Copied {src} → {dest_dir}/{dst}")
        else:
            print(f"   ⚠️ Not found: {src}")
    
    # Also copy threshold
    if os.path.exists('optimal_threshold.txt'):
        shutil.copy('optimal_threshold.txt', os.path.join(dest_dir, 'optimal_threshold.txt'))
    
    # Create metadata
    metadata = f"""# Model Metadata
version: v1
created: {datetime.now().isoformat()}
models:
  - duration_model.json (regression)
  - routing_model.json (multi-class)
  - sla_model.json (binary)
variability_factors:
  - volume_spikes: true
  - workforce_variability: true
  - case_complexity: true
  - concept_drift: true
"""
    with open(os.path.join(dest_dir, 'metadata.yaml'), 'w') as f:
        f.write(metadata)
    
    print(f"\n✅ Models exported to: {dest_dir}")
    return True


def run_full_pipeline(num_cases: int = 10000):
    """Run the complete pipeline end-to-end"""
    print("\n" + "=" * 60)
    print("🚀 APPIAN ML PIPELINE - FULL RUN")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    stages = [
        ("Data Generation", lambda: run_data_generation(num_cases)),
        ("Model Training", run_training),
        ("Threshold Analysis", run_threshold_analysis),
        ("Inference Test", run_inference_test),
        ("Export", run_export),
    ]
    
    for stage_name, stage_fn in tqdm(stages, desc="Overall Pipeline Progress"):
        try:
            success = stage_fn()
            if not success:
                print(f"❌ Stage failed: {stage_name}")
                return False
        except Exception as e:
            print(f"❌ Error in {stage_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n📁 Outputs:")
    print("   - cases_table.csv")
    print("   - events_log.csv")
    print("   - duration_model.json")
    print("   - routing_model.json")
    print("   - sla_model.json")
    print("   - optimal_threshold.txt")
    print("   - ../weights/xgboost/v1/ (exported)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Appian ML Pipeline - Training, Inference & Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --all           # Run complete pipeline
  python pipeline.py --generate      # Generate data only
  python pipeline.py --train         # Train models only
  python pipeline.py --threshold     # Analyze thresholds only
  python pipeline.py --inference     # Test inference only
  python pipeline.py --export        # Export to weights folder only
  
  python pipeline.py --all --cases 20000   # Full run with 20k cases
"""
    )
    
    # Stages
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--generate', action='store_true', help='Stage 1: Generate data')
    parser.add_argument('--train', action='store_true', help='Stage 2: Train models')
    parser.add_argument('--threshold', action='store_true', help='Stage 3: Threshold analysis')
    parser.add_argument('--inference', action='store_true', help='Stage 4: Test inference')
    parser.add_argument('--export', action='store_true', help='Stage 5: Export models')
    
    # Options
    parser.add_argument('--cases', type=int, default=10000, help='Number of cases (default: 10000)')
    parser.add_argument('--export-dir', type=str, help='Custom export directory')
    
    args = parser.parse_args()
    
    # If no stage specified, show help
    if not any([args.all, args.generate, args.train, args.threshold, args.inference, args.export]):
        parser.print_help()
        return
    
    # Run stages
    try:
        if args.all:
            success = run_full_pipeline(args.cases)
        else:
            if args.generate:
                run_data_generation(args.cases)
            if args.train:
                run_training()
            if args.threshold:
                run_threshold_analysis()
            if args.inference:
                run_inference_test()
            if args.export:
                run_export(args.export_dir)
        
        sys.exit(0)
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
