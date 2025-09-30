#!/usr/bin/env python3
"""
Training Analysis and Visualization Script
Analyze training progress, early stopping behavior, and model performance
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingAnalyzer:
    """Analyze and visualize training progress"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.metrics_file = self.model_dir / "training_metrics.json"
        self.config_file = self.model_dir / "training_config.json"
        
    def load_training_data(self) -> Dict[str, Any]:
        """Load training metrics and configuration"""
        
        data = {}
        
        # Load training metrics
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                data['metrics'] = json.load(f)
            logger.info(f"✅ Loaded training metrics from {self.metrics_file}")
        else:
            logger.warning(f"❌ Training metrics not found: {self.metrics_file}")
            data['metrics'] = {}
        
        # Load configuration
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data['config'] = json.load(f)
            logger.info(f"✅ Loaded training config from {self.config_file}")
        else:
            logger.warning(f"❌ Training config not found: {self.config_file}")
            data['config'] = {}
        
        return data
    
    def analyze_early_stopping_effectiveness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if early stopping was effective"""
        
        config = data.get('config', {})
        metrics = data.get('metrics', {})
        
        analysis = {
            'early_stopping_enabled': config.get('early_stopping', {}).get('enabled', False),
            'max_epochs_configured': config.get('num_train_epochs', 'Unknown'),
            'actual_epochs_completed': metrics.get('epochs_completed', 'Unknown'),
            'early_stopping_triggered': False,
            'effectiveness': 'Unknown'
        }
        
        if analysis['early_stopping_enabled'] and isinstance(analysis['actual_epochs_completed'], (int, float)):
            if analysis['actual_epochs_completed'] < analysis['max_epochs_configured']:
                analysis['early_stopping_triggered'] = True
                analysis['effectiveness'] = 'Effective - Prevented overfitting'
            else:
                analysis['effectiveness'] = 'Not triggered - Model may benefit from more training'
        
        return analysis
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        
        data = self.load_training_data()
        early_stopping_analysis = self.analyze_early_stopping_effectiveness(data)
        
        report = []
        report.append("🎯 PHI-3 PII ANONYMIZATION TRAINING ANALYSIS")
        report.append("=" * 60)
        
        # Basic Information
        report.append("\n📊 TRAINING OVERVIEW:")
        report.append("-" * 30)
        
        config = data.get('config', {})
        metrics = data.get('metrics', {})
        
        if config:
            report.append(f"Model: {config.get('model_name', 'Unknown')}")
            report.append(f"Max Epochs Configured: {config.get('num_train_epochs', 'Unknown')}")
            report.append(f"Learning Rate: {config.get('learning_rate', 'Unknown')}")
            report.append(f"Batch Size: {config.get('per_device_train_batch_size', 'Unknown')}")
            report.append(f"Gradient Accumulation: {config.get('gradient_accumulation_steps', 'Unknown')}")
        
        # Training Results
        if metrics:
            report.append(f"\n📈 TRAINING RESULTS:")
            report.append("-" * 30)
            report.append(f"Final Training Loss: {metrics.get('final_training_loss', 'Unknown'):.6f}")
            report.append(f"Total Training Steps: {metrics.get('total_steps', 'Unknown')}")
            report.append(f"Actual Epochs Completed: {metrics.get('epochs_completed', 'Unknown'):.2f}")
            
            # Validation results
            eval_loss = metrics.get('eval_loss')
            if eval_loss:
                report.append(f"Final Validation Loss: {eval_loss:.6f}")
        
        # Early Stopping Analysis
        report.append(f"\n🎯 EARLY STOPPING ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Early Stopping Enabled: {early_stopping_analysis['early_stopping_enabled']}")
        
        if early_stopping_analysis['early_stopping_enabled']:
            patience = config.get('early_stopping', {}).get('patience', 'Unknown')
            min_delta = config.get('early_stopping', {}).get('min_delta', 'Unknown')
            report.append(f"Patience: {patience} evaluations")
            report.append(f"Min Delta: {min_delta}")
            report.append(f"Early Stopping Triggered: {early_stopping_analysis['early_stopping_triggered']}")
            report.append(f"Effectiveness: {early_stopping_analysis['effectiveness']}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        report.append("-" * 30)
        
        if early_stopping_analysis['early_stopping_enabled']:
            if early_stopping_analysis['early_stopping_triggered']:
                report.append("✅ Early stopping worked well - prevented potential overfitting")
                report.append("   Consider current configuration optimal for this dataset")
            else:
                report.append("🤔 Early stopping not triggered - model may benefit from:")
                report.append("   • Increasing max epochs (current: {})".format(early_stopping_analysis['max_epochs_configured']))
                report.append("   • Reducing learning rate for more gradual convergence")
                report.append("   • Increasing patience if loss is still decreasing slowly")
        else:
            report.append("⚠️  Early stopping not enabled - consider enabling it to:")
            report.append("   • Prevent overfitting")
            report.append("   • Save training time")
            report.append("   • Automatically find optimal stopping point")
        
        # Validation performance
        if metrics.get('eval_loss'):
            train_loss = metrics.get('final_training_loss', 0)
            val_loss = metrics.get('eval_loss', 0)
            
            if val_loss > train_loss * 1.2:  # Validation loss 20% higher than training
                report.append("⚠️  Potential overfitting detected (val_loss >> train_loss)")
                report.append("   • Consider reducing epochs or adding regularization")
                report.append("   • Early stopping with lower patience may help")
            elif val_loss < train_loss * 1.1:  # Very close losses
                report.append("✅ Good generalization - validation loss close to training loss")
        
        return "\n".join(report)
    
    def save_analysis(self, output_file: str = None):
        """Save analysis to file"""
        
        if output_file is None:
            output_file = self.model_dir / "training_analysis.txt"
        
        report = self.generate_training_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📝 Training analysis saved to {output_file}")
        return output_file

def compare_training_strategies():
    """Compare different training strategies"""
    
    print("🔍 EARLY STOPPING vs FIXED EPOCHS COMPARISON")
    print("=" * 50)
    
    print("\n📊 FIXED EPOCHS (Traditional Approach):")
    print("✅ Pros:")
    print("   • Predictable training time")
    print("   • Simple to understand and implement")
    print("   • Works well with known optimal epoch count")
    
    print("❌ Cons:")
    print("   • Risk of overfitting with too many epochs")
    print("   • May stop too early with too few epochs")
    print("   • Wastes compute if optimal point reached earlier")
    print("   • Requires manual tuning for each dataset")
    
    print("\n🎯 EARLY STOPPING (Adaptive Approach):")
    print("✅ Pros:")
    print("   • Prevents overfitting automatically")
    print("   • Saves training time and compute")
    print("   • Finds optimal stopping point data-dependently")
    print("   • Better generalization performance")
    print("   • Professional ML practice")
    
    print("❌ Cons:")
    print("   • Slightly more complex to implement")
    print("   • May stop too early with noisy validation")
    print("   • Requires good validation set")
    
    print("\n💡 RECOMMENDED CONFIGURATION:")
    print("   • Max Epochs: 10-15 (allows room for learning)")
    print("   • Early Stopping Patience: 3-5 evaluations")
    print("   • Min Delta: 0.001 (sensitive to small improvements)")
    print("   • Evaluation Strategy: Every 250-500 steps")
    print("   • Monitor: Validation loss (most reliable)")
    
    print("\n🎓 SENIOR AI ENGINEER INSIGHT:")
    print("   Early stopping is standard practice in production ML systems.")
    print("   It's especially important for fine-tuning where overfitting is common.")
    print("   Your current configuration (10 epochs + early stopping) is optimal!")

def main():
    """Main analysis function"""
    
    model_dir = "/opt/projects/phi3_finetune_new/models/final/phi3-pii-anonymizer"
    
    if not Path(model_dir).exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("   Train the model first: python scripts/train_phi3.py")
        print("\n   Or analyze training strategy comparison:")
        compare_training_strategies()
        return
    
    # Analyze training
    analyzer = TrainingAnalyzer(model_dir)
    
    # Generate and print report
    report = analyzer.generate_training_report()
    print(report)
    
    # Save analysis
    analysis_file = analyzer.save_analysis()
    print(f"\n💾 Detailed analysis saved to: {analysis_file}")
    
    # Show strategy comparison
    print("\n" + "="*60)
    compare_training_strategies()

if __name__ == "__main__":
    main()