# Support

Looking for help with CABiNet? Here's how to get support.

## Documentation

Before asking for help, check if your question is answered in:

- **[README.md](README.md)** - Installation, usage, and basic examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup and guidelines
- **[GitHub Issues](https://github.com/dronefreak/CABiNet/issues)** - Known bugs and feature requests
- **Research Paper** - Technical details and methodology
  - [ICRA 2021 Paper](https://doi.org/10.1109/ICRA48506.2021.9560977)
  - [ISPRS Journal Paper](https://doi.org/10.1016/j.isprsjprs.2021.06.006)

## Getting Help

### GitHub Issues (Recommended)

For bugs, feature requests, or technical questions:

1. **Search existing issues** to see if someone else has asked the same question
2. **Create a new issue** using one of our templates:
   - Bug Report
   - Feature Request
   - Question

[Open an Issue →](https://github.com/dronefreak/CABiNet/issues/new)

### Email Support

For private inquiries or security issues:

📧 **kumaar324@gmail.com**

Please note:

- Email responses may take longer than GitHub issues
- For technical questions, GitHub issues are preferred (helps others too!)
- Security vulnerabilities should ONLY be reported via email (see [SECURITY.md](SECURITY.md))

## Common Questions

### Installation Issues

**Q: Conda environment creation fails**

```bash
# Try creating with specific Python version
conda create -n cabinet python=3.8
conda activate cabinet
pip install -r requirements.txt  # if we create one
```

**Q: CUDA/GPU not detected**

```bash
# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Issues

**Q: Where do I download Cityscapes dataset?**

Visit [Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/) and download:

- `gtFine_trainvaltest.zip` (241MB)
- `leftImg8bit_trainvaltest.zip` (11GB)

**Q: UAVid dataset access?**

Download from [UAVid official website](https://uavid.nl/) under the Downloads section.

### Training Issues

**Q: Out of memory errors during training**

Reduce batch size in config files:

```json
{
  "batch_size": 4 // Try reducing this
}
```

**Q: Where are pre-trained weights?**

Currently working on uploading pre-trained weights. Check back soon or open an issue to track this.

### Inference Issues

**Q: How to run inference on custom images?**

```bash
python scripts/demo.py --config configs/train_citys.json --image path/to/image.jpg
```

**Q: Slow inference speed**

- Use FP16 precision
- Reduce input resolution
- Ensure GPU is being used
- Check for CPU bottlenecks in data loading

## Feature Requests

Have an idea for improving CABiNet? We'd love to hear it!

1. Check if someone else has suggested it in [Issues](https://github.com/dronefreak/CABiNet/issues)
2. If not, [create a new issue](https://github.com/dronefreak/CABiNet/issues/new) with:
   - Clear description of the feature
   - Use cases and benefits
   - Any implementation ideas

## Contributing

Want to contribute code or documentation? Check out our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

Areas where we especially need help:

- Adding pre-trained model weights
- Creating unit tests
- Dockerization
- Additional dataset support
- Performance optimization

## Response Times

We're a small team working on this in our spare time. Expected response times:

| Channel                  | Expected Response |
| ------------------------ | ----------------- |
| GitHub Issues (Bug)      | 3-5 days          |
| GitHub Issues (Question) | 5-7 days          |
| GitHub Issues (Feature)  | 1-2 weeks         |
| Email                    | 1-2 weeks         |
| Security Issues          | 1-3 days          |

Please be patient! If your issue is urgent and you haven't received a response, feel free to politely ping the thread.

## Community Guidelines

When seeking support:

✅ **Do:**

- Be respectful and patient
- Provide context and details
- Share relevant code, configs, and error messages
- Search before posting
- Follow up if you solve your own issue (helps others!)

❌ **Don't:**

- Demand immediate responses
- Post duplicate issues
- Share security vulnerabilities publicly
- Go off-topic in existing issues

## Resources

### Related Projects

- [PyTorch](https://pytorch.org/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [UAVid Dataset](https://uavid.nl/)

### Semantic Segmentation Resources

- [Papers with Code - Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation)
- [Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)

### Self-Driving & Robotics

- [ICRA Conference](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra)
- [Autonomous Driving Datasets](https://github.com/autonomousvision/awesome-autonomous-driving-datasets)

## No Response?

If you've waited longer than the expected response time:

1. **Double-check your issue** - Does it contain enough information?
2. **Bump the thread** - Add a polite comment asking if anyone has seen it
3. **Reach out via email** - Use kumaar324@gmail.com as a last resort

---

Thank you for using CABiNet! Your feedback helps make this project better for everyone.
