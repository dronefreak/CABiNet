# Security Policy

## Supported Versions

Currently, we support security updates for the following versions of CABiNet:

| Version              | Supported          |
| -------------------- | ------------------ |
| Latest (main branch) | :white_check_mark: |
| < Latest             | :x:                |

As this is a research project, we recommend always using the latest version from the main branch.

## Reporting a Vulnerability

We take the security of CABiNet seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by emailing:

**kumaar324@gmail.com**

Include the following information in your report:

- **Type of vulnerability** (e.g., code injection, privilege escalation, etc.)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### What to Expect

After you submit a vulnerability report:

1. **Acknowledgment**: We will acknowledge receipt of your report within 3 business days
2. **Investigation**: We will investigate and validate the vulnerability
3. **Updates**: We will keep you informed about our progress
4. **Resolution**: We will work on a fix and coordinate disclosure timing with you
5. **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Security Update Process

When a security issue is confirmed:

1. A security advisory will be created
2. A fix will be developed in a private repository
3. The fix will be tested thoroughly
4. A security release will be published
5. The vulnerability will be publicly disclosed with appropriate credit

## Security Best Practices

When using CABiNet in production environments:

### Data Security

- **Validate all input data** before passing to the model
- **Sanitize file paths** when loading datasets or models
- **Use secure data storage** for training datasets
- **Implement access controls** for model weights and configs

### Model Security

- **Verify model checksums** before loading pre-trained weights
- **Use trusted sources** for downloading pre-trained models
- **Implement input validation** to prevent adversarial attacks
- **Monitor model predictions** for anomalies

### Deployment Security

- **Run inference in isolated environments** (containers, sandboxes)
- **Limit resource access** (CPU, GPU, memory) to prevent DoS
- **Implement rate limiting** for API endpoints
- **Use HTTPS** for all network communications
- **Keep dependencies updated** to patch known vulnerabilities

### Dependencies

This project relies on external packages. Keep them updated:

```bash
# Regularly update dependencies
pip install --upgrade torch torchvision
pip list --outdated
```

Monitor security advisories for:

- PyTorch
- NumPy
- Pillow
- OpenCV
- Other dependencies listed in `cabinet_environment.yml`

## Known Security Considerations

### Model Inversion Attacks

Deep learning models can potentially leak training data through model inversion attacks. If using CABiNet with sensitive data:

- Implement differential privacy during training
- Use techniques like knowledge distillation for deployment
- Limit access to model weights

### Adversarial Examples

Semantic segmentation models can be fooled by adversarial perturbations:

- Implement input validation and sanitization
- Consider adversarial training if deploying in security-critical applications
- Monitor for unusual prediction patterns

### Resource Exhaustion

Large models can consume significant computational resources:

- Implement timeouts for inference
- Set memory limits
- Use batch size limits
- Monitor GPU/CPU usage

## Disclosure Policy

- Security vulnerabilities will be disclosed publicly after a fix is available
- We aim for a 90-day disclosure timeline from initial report to public disclosure
- Critical vulnerabilities may be disclosed sooner if actively exploited
- We will coordinate disclosure timing with the reporter

## Security Contacts

For security-related inquiries:

- **Email**: kumaar324@gmail.com
- **Response Time**: Within 3 business days

For general questions (non-security), please use GitHub Issues.

## Additional Resources

- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [PyTorch Security Guidelines](https://pytorch.org/docs/stable/notes/security.html)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

---

**Last Updated**: November 2025

Thank you for helping keep CABiNet and its users safe!
