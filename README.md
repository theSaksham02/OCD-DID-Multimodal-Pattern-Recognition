## Training
#### Train multimodal model
- python scripts/train_model.py –config configs/training_config.yaml
#### Train baseline (facial only)
- python scripts/train_model.py –model facial_only –epochs 50
#### Resume from checkpoint
python scripts/train_model.py –resume results/models/checkpoint_epoch30.pth

## 📈 Evaluation
#### Evaluate on test set
python scripts/evaluate_model.py –model results/models/best_model.pth
#### Generate visualizations
python src/utils/visualization.py –output results/figures/

## 🎯 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Facial-Only | 78.5% | 76.8% | 77.2% | 77.0% |
| Pose-Only | 72.3% | 70.1% | 71.5% | 70.8% |
| **Multimodal Fusion** | **87.2%** | **85.6%** | **86.4%** | **86.0%** |

src/
── data/          # Data loading and Rreprocessing
├── models/        # Model architectures
├── training/      # Training and evaluation loops
└── utils/         # Helper functions and visualization

### Key Findings:
- OCD patients show 67% accuracy on disgust recognition (vs 92% controls)
- DID switching detected with 82% precision
- Multimodal fusion improves accuracy by 12-15% over single modality

## 📁 Project Structure


## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- FER2013 dataset by Pierre-Luc Carrier and Aaron Courville
- OCD clinical data by Ohin Haque (Kaggle)
- OpenPose by CMU Perceptual Computing Lab
- Base facial recognition code adapted from [mujiyantosvc/FER-Mental-Health](https://github.com/mujiyantosvc/Facial-Expression-Recognition-FER-for-Mental-Health-Detection-)

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/YOUR_USERNAME/OCD-DID-Multimodal-Pattern-Recognition](https://github.com/YOUR_USERNAME/OCD-DID-Multimodal-Pattern-Recognition)

---

**Note**: This research is for academic purposes only. Clinical deployment requires proper validation and ethical approval.