# S4L-COD - Camouflaged Object Detection Based on Self-upervised Semi-supervised Learning

## dependency and prerequisites
- `pip install -r requirement.txt`

## data preparation
- COD10K
- CPD1K
- CAMO
- CHAMELEON

## train on combined dataset
- `python source/main.py --cfg configure/w48.yaml`

## test based on best epoch
- `python source/test.py --cfg configure/w48.yaml`

## modify your own configure

## technical stuff
- HR-Net baseline for resolution presevering mask output
- Multiview non-local filter for enhanced high level feature
- Online contrastive representation for robust learning
- Fully unsupervised approach to simulating camouflaged object in wild context