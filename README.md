# Face-Structure-Aware DeepFake Localization

**Team:** Web Warriors

**Team Members**
- Rajnish (B24DS024)
- Shubham Kumar Jha (B24DS029)
- Sikharam Satya Sri Harsha Phani (B24DS031)

---

## 1. Project Overview

This project builds on [FakeShield](https://github.com/zhipeixu/FakeShield), an explainable framework for image-forgery detection and localization.

FakeShield combines multimodal explanations with image-level detection and pixel-level localization. Although detection performance is strong, accurately identifying the manipulated pixels remains challenging.

Our project investigates whether facial structure can provide useful spatial guidance for the localization stage.

The main idea is to create a soft facial prior from face metadata and facial regions mentioned in the model's explanation, then use this prior to refine the localization logits.

**Project focus:** improving MFLM pixel-level localization, rather than changing the image-level detection component.

## 2. Objectives

- Study the localization behaviour of the original FakeShield pipeline.
- Use facial landmarks and face information as spatial guidance.
- Extract facial-region cues from the model's explanation.
- Apply a prior to the original localization logits at inference time.
- Compare the refined masks with the baseline masks using paired metrics.
- Identify limitations and directions for further improvement.

## 3. Existing FakeShield Pipeline

The original FakeShield framework includes the following components:

1. **DTG:** generates textual descriptions and guidance.
2. **DTE-FDM:** performs forgery detection and produces explanations.
3. **MFLM:** generates pixel-level localization predictions.
4. **SAM:** supports mask generation/refinement in the original framework.

For details about the original architecture, checkpoints, and setup, refer to the [official FakeShield repository](https://github.com/zhipeixu/FakeShield) and the [research paper](https://arxiv.org/abs/2410.02761).

## 4. Proposed Method

Our extension adds a facial spatial prior to the MFLM localization output.

### 4.1 Prior Generation

The prior is generated using available face information, including the face bounding box and five facial landmarks.

The explanation text is also checked for facial-region cues. When region-related terms are found, they help guide the spatial prior.

A Gaussian-style spatial map is used to create a soft prior. A broader face-based fallback is used when specific facial-region cues are absent.

### 4.2 Logit Refinement

Let:

- `z(x,y)` be the original MFLM localization logit.
- `P(x,y)` be the facial spatial prior.
- `alpha` be the prior-strength parameter.

The refined logit is:

\[ z'(x,y) = z(x,y) + \\alpha(2P(x,y)-1)
\]

The adjusted logits are then used to obtain the refined localization prediction.

This operation is performed during inference. The reported experiment does **not** fine-tune the original model weights.

### 4.3 Processing Workflow

1. Provide an image to the FakeShield pipeline.
2. Run the detection and explanation component.
3. Obtain the MFLM localization logits.
4. Read the available face metadata.
5. Identify facial-region cues in the explanation.
6. Generate the soft facial prior.
7. Adjust the original localization logits.
8. Generate the refined mask.
9. Evaluate the baseline and refined masks against the ground truth.

## 5. Dataset and Experimental Setup

The experiments use the FaceApp portion of the DFFD data prepared for this project.

The DTE-FDM and MFLM components were run in separate Docker environments. The MFLM component required a separate runtime to handle its older software dependencies.

The full paired evaluation contains **6,248 images** for which both baseline and refined predictions were available.

A further 61 images were skipped by MFLM after DTE-FDM classified them as not tampered. These images are not part of the paired localization comparison.

The original model checkpoints and datasets are not included in this repository. Users should obtain them from their official sources and follow the setup instructions provided by FakeShield.

## 6. Evaluation Metrics

We compare the baseline localization masks with the masks produced after facial-prior refinement.

The reported metrics are:

- **Mean Intersection over Union (IoU):** measures the overlap between
the predicted manipulated region and the ground-truth mask.
- **Mean Pixel F1:** measures pixel-level precision and recall through
the F1 score.

Both metrics are calculated on the same paired set of 6,248 images.

## 7. Results

### 7.1 Aggregate Results

| Metric | Baseline | Facial-prior refinement |
|---|---:|---:|
| Mean IoU | 0.140032 | 0.139441 |
| Mean Pixel F1 | 0.219012 | 0.218175 |

### 7.2 Per-image Comparison

Across the paired evaluation:

- 3,637 images improved.
- 2,380 images declined.
- 231 images remained unchanged.

### 7.3 Interpretation

Although more individual images improved than declined, the aggregate mean IoU and mean Pixel F1 both decreased slightly.

Therefore, the current facial-prior refinement does **not** demonstrate an overall improvement over the baseline localization results.

The result suggests that a facial prior can help some images, but the current prior-generation and logit-adjustment strategy is not reliable enough to improve the aggregate metrics.

## 8. Limitations

- Facial-region cues depend on the wording of the generated explanation.
- Broad fallback priors may include pixels unrelated to manipulation.
- A facial prior may not be useful when manipulation occurs outside
the regions emphasized by the prior.
- The prior strength can affect the final mask and requires careful
validation.
- The current method applies a fixed inference-time adjustment rather
than learning how the prior should influence localization.
- The reported experiment does not include fine-tuning of MFLM.

## 9. Future Work

- Improve the extraction of facial-region cues from explanations.
- Evaluate different prior-generation strategies.
- Tune the prior strength using a separate validation set.
- Investigate adaptive weighting instead of a fixed prior strength.
- Study cases where the prior improves or harms localization.
- Explore learned spatial priors and model fine-tuning in future work.

## 10. Repository and Reproduction Notes

This repository documents the project extension and its evaluation.

For the original FakeShield implementation, installation steps, model components, and checkpoint information, use the [official FakeShield GitHub repository](https://github.com/zhipeixu/FakeShield).

Before reproducing the experiments, ensure that the required datasets, pretrained checkpoints, and compatible runtime environments are available.

The numerical results in this README refer to the paired evaluation described above. They should not be interpreted as a claim that the proposed refinement improves FakeShield overall.

## 11. References

1. Z. Xu, X. Zhang, R. Li, Z. Tang, Q. Huang, and J. Zhang, “FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models,” *International Conference on Learning Representations (ICLR)*, 2025.
- Paper: https://arxiv.org/abs/2410.02761
- Code: https://github.com/zhipeixu/FakeShield

2. A. Kirillov et al., “Segment Anything,” *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023.
- Paper: https://arxiv.org/abs/2304.02643

3. X. Dang et al., “On the Detection of Digital Face Manipulation,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2020.
- Dataset/resource page: https://cse.buffalo.edu/~siweilyu/

---

**Note:** This project is an experimental extension of FakeShield. The current evaluation reports a small decrease in aggregate localization metrics after facial-prior refinement.
