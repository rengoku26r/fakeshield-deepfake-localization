FakeShield DeepFake Localization — Web Warriors

A reproducibility study of FakeShield with a focus on DeepFake manipulation localization

This repository documents the Web Warriors project based on "FakeShield" (https://github.com/zhipeixu/FakeShield), an explainable image-forgery detection and localization framework. The project reproduces the baseline localization pipeline, evaluates predicted manipulation masks against ground-truth masks, and establishes a foundation for improving facial-region localization through explicit facial-structure information.

«Project status: Baseline inference and partial-subset localization evaluation have been completed. The reported 500-image evaluation contains 496 successfully evaluated images. The proposed facial-structure-aware refinement, LoRA fine-tuning, and ablation studies remain separate research objectives and are not claimed as completed.»

---

Table of Contents

- "1. Project Overview" (#1-project-overview)
- "2. Research Motivation" (#2-research-motivation)
- "3. Original FakeShield Architecture" (#3-original-fakeshield-architecture)
- "4. Project Objectives and Scope" (#4-project-objectives-and-scope)
- "5. Repository Structure" (#5-repository-structure)
- "6. Dataset" (#6-dataset)
- "7. Experimental Environment" (#7-experimental-environment)
- "8. Installation and Setup" (#8-installation-and-setup)
- "9. Inference Pipeline" (#9-inference-pipeline)
- "10. Evaluation Methodology" (#10-evaluation-methodology)
- "11. Experimental Results" (#11-experimental-results)
- "12. Engineering Findings" (#12-engineering-findings)
- "13. Proposed Research Extension" (#13-proposed-research-extension)
- "14. Limitations" (#14-limitations)
- "15. Reproducibility Checklist" (#15-reproducibility-checklist)
- "16. Future Work" (#16-future-work)
- "17. References and Acknowledgements" (#17-references-and-acknowledgements)
- "18. Citation" (#18-citation)

---

1. Project Overview

FakeShield is a multimodal framework for explainable image-forgery detection and localization. Unlike approaches that only return a real/fake classification, FakeShield aims to identify manipulated content, explain the evidence in natural language, and generate a pixel-level mask of the suspected forged region.

This project, developed by Web Warriors, focuses on reproducing and evaluating FakeShield's DeepFake localization capability under practical hardware constraints.

The project has three main objectives:

1. Reproduce the original DTE-FDM and MFLM inference workflow.
2. Evaluate predicted manipulation masks against ground-truth masks using Intersection over Union (IoU) and pixel-level F1.
3. Establish a reliable baseline for future experiments that incorporate facial structure into the localization stage.

The work prioritizes reproducibility, traceable experimental outputs, memory-aware inference, and a clear separation between the original implementation and project-specific modifications.

Research information

Item| Details
Project| Web Warriors
Base framework| FakeShield
Primary research task| DeepFake manipulation localization
Dataset subset| FaceAPP validation images
Baseline evaluation| 20-image and 500-image experiments
Main metrics| Mean IoU and mean pixel-level F1
Reference hardware| NVIDIA GeForce RTX 3060, 12 GB VRAM
Inference environment| Docker-based runtimes
Original paper| ICLR 2025

2. Research Motivation

FakeShield demonstrates strong DeepFake detection performance in the original paper, but its localization performance on the DeepFake domain is substantially weaker.

The paper reports the following reference values:

Metric| Original FakeShield
Detection accuracy| 0.98
Detection F1| 0.99
Localization IoU| 0.14
Localization pixel F1| 0.22

These values refer to the original paper's evaluation setting, not measurements obtained from the experiments in this repository.

The central research problem is the gap between recognizing a manipulated face and identifying the exact pixels affected by the manipulation.

DeepFake attribute editing may change facial properties such as skin texture, age, smile, or other appearance characteristics without creating a clear boundary around the altered region. Consequently, producing an accurate pixel-level mask can remain difficult even when the model generates a plausible textual explanation.

The original paper also investigates the effect of changing the textual descriptions supplied to the localization module. Its DeepFake-domain ablations indicate that improving the description alone does not resolve the localization weakness.

This motivates investigating whether explicit, image-specific facial geometry can help the localization module identify relevant spatial regions more precisely.

3. Original FakeShield Architecture

FakeShield consists of two primary inference modules, supported by a domain-tag generation component.

3.1 Domain Tag Generator (DTG)

DTG identifies the manipulation domain and provides domain information to the detection and explanation stage.

The original framework considers Photoshop manipulation, DeepFake manipulation, and AI-generated content editing.

3.2 Domain Tag-guided Explainable Forgery Detection Module (DTE-FDM)

DTE-FDM uses a multimodal language model to analyze an image and produce a manipulation judgment with a natural-language explanation.

Its output contains the information required by the localization stage, including the input image and generated textual analysis.

3.3 Multimodal Forgery Localization Module (MFLM)

MFLM combines information from the image and DTE-FDM's output to generate a manipulation mask. Its architecture includes a Tamper Comprehension Module (TCM) and a Segment Anything Model (SAM)-based segmentation component.

The high-level inference pipeline is:

                    Input Image
                         |
                         v
                  Domain Tag Generator
                         |
                         v
                       DTE-FDM
                         |
              Manipulation Judgment
                 and Explanation
                         |
                         v
                        MFLM
                         |
               TCM + SAM Localization
                         |
                         v
              Predicted Manipulation Mask
                         |
                         v
            Comparison with Ground Truth
                         |
                         v
                  IoU and Pixel F1

DTE-FDM and MFLM are separate stages in the original framework. This project preserves that separation for baseline inference.

4. Project Objectives and Scope

Completed baseline work

The baseline work documented in this repository includes:

- Setting up the original FakeShield implementation and model assets.
- Configuring separate Docker environments for DTE-FDM and MFLM.
- Running DTE-FDM inference with 4-bit quantization on the reference GPU.
- Addressing MFLM inference memory constraints through implementation-level device-placement changes.
- Generating localization outputs for experimental images.
- Evaluating predicted masks against matching ground-truth masks.
- Recording per-image metrics and aggregate evaluation results.

Research objectives beyond the baseline

The following are proposed extensions rather than completed baseline contributions:

- Constructing explicit facial-region spatial priors.
- Integrating facial bounding boxes and landmarks into localization.
- Applying coarse-to-fine mask refinement.
- Fine-tuning the localization module using LoRA.
- Conducting controlled ablation experiments.
- Comparing the modified method against the original baseline on identical evaluation samples.

The baseline results must not be interpreted as results from the proposed facial-structure-aware method.

5. Repository Structure

The repository is intended to organize project-specific scripts, reproducibility documentation, and evaluation artifacts separately from the upstream FakeShield implementation.

A logical layout for the project-specific files is:

fakeshield-deepfake-localization/
├── README.md
├── baseline/
│   ├── inference/
│   │   ├── run_dte_fdm.py
│   │   └── run_mflm.py
│   ├── evaluation/
│   │   └── calculate_metrics.py
│   └── dataset/
│       └── build_subset.py
├── configs/
│   └── README.md
├── experiments/
│   └── phase2/
│       ├── README.md
│       └── results/
├── docs/
│   └── reproducibility.md
├── requirements.txt
└── .gitignore

This is a logical organization for project-maintained files, not a claim that every listed file is already present in the public repository. Use the actual tracked files and paths when reproducing the experiments.

The original FakeShield implementation remains available at:

https://github.com/zhipeixu/FakeShield

The upstream repository contains the DTE-FDM and MFLM implementations, supporting scripts, and model-loading instructions.

Large and environment-specific assets

The following should be obtained or configured separately rather than assumed to be included in this project repository:

- Original FakeShield model checkpoints.
- SAM pretrained weights.
- Full training and validation datasets.
- Docker images and container layers.
- Generated intermediate outputs and large prediction collections.
- Local CLIP vision-tower assets, if required by the selected configuration.

Do not commit large model weights, private data, or machine-specific absolute paths unless there is an explicit requirement and permission to distribute them.

6. Dataset

6.1 Primary evaluation subset

The primary baseline evaluation uses manipulated FaceAPP validation images from the DeepFake dataset distributed with FakeShield.

The expected data organization follows the upstream convention:

dataset/
└── deepfake/
    ├── FaceAPP_Train/
    │   ├── image/
    │   └── mask/
    ├── FaceAPP_Val/
    │   ├── image/
    │   └── mask/
    ├── FFHQ_Train/
    │   └── image/
    └── FFHQ_Val/
        └── image/

- "FaceAPP_Train" contains manipulated training images and corresponding masks.
- "FaceAPP_Val" contains manipulated validation images and corresponding masks.
- "FFHQ_Train" and "FFHQ_Val" contain authentic face images used by the broader dataset and training/evaluation framework.

Actual directory names and capitalization must match the downloaded dataset.

6.2 Additional datasets

The broader FakeShield dataset collection includes other manipulation domains and supporting resources, such as:

- Photoshop manipulation datasets, including CASIAv2 and Coverage.
- AI-generated editing data, including SD-Inpaint.
- MMTD-Set multimodal image, mask, and description data.
- FFHQ authentic face images.

These resources belong to the broader framework. The results reported in this README concern the stated FaceAPP localization experiments, not a complete evaluation across all datasets.

6.3 Image-mask matching

For localization evaluation, each prediction must be matched to its corresponding ground-truth mask.

The evaluation implementation uses matching filename stems to associate images and masks. Before running a complete experiment, verify that:

- Every selected image has the intended ground-truth mask.
- Image and mask filenames are paired correctly.
- The masks correspond to the same manipulation instance.
- Missing predictions are recorded explicitly rather than treated as successful results.

The full dataset should be stored outside the Git repository when its size or redistribution conditions make that appropriate.

7. Experimental Environment

The reference inference environment used for the engineering work is summarized below.

Component| Reference configuration
Operating system| Ubuntu 24.04.5 LTS
CPU| Intel Core i7-12700
System RAM| Approximately 15 GiB
GPU| NVIDIA GeForce RTX 3060
GPU memory| 12 GB
NVIDIA driver| 535.288.01
Host CUDA environment| CUDA 12.2
DTE-FDM runtime| "zhipeixu/dte-fdm:v1.0"
MFLM runtime| "fakeshield-mflm-runtime:1.1"
MMCV for the custom MFLM runtime| 1.5.0

These values describe the reference setup and should not be treated as universal requirements.

Why separate runtimes were used

DTE-FDM and MFLM have different dependency requirements. In particular, MFLM relies on an older MMCV/MMDetection dependency combination.

The project therefore uses separate containers rather than forcing both components into one Python environment.

The DTE-FDM container uses the released runtime with 4-bit model loading. The MFLM container uses a project-built runtime with MMCV CUDA operations enabled.

For the custom MMCV build, limiting parallel compilation was important on the reference machine:

MMCV_WITH_OPS=1 MAX_JOBS=1 pip install .

The installed MMCV CUDA operations were checked during environment setup. This is a reference implementation detail and may require adjustment for different CUDA, PyTorch, compiler, or dependency versions.

8. Installation and Setup

8.1 Obtain the source code

Clone the project repository:

git clone https://github.com/rengoku26r/fakeshield-deepfake-localization.git
cd fakeshield-deepfake-localization

Obtain the original implementation separately:

git clone https://github.com/zhipeixu/FakeShield.git

Follow the upstream installation instructions for the appropriate model code and dependencies.

8.2 Download model checkpoints

FakeShield's pretrained weights are available from the official Hugging Face model repository:

https://huggingface.co/zhipeixu/fakeshield-v1-22b

The upstream README describes the required checkpoint organization, including DTE-FDM, MFLM, and DTG weights, along with the SAM checkpoint.

The expected layout is:

FakeShield/
└── weight/
    ├── fakeshield-v1-22b/
    │   ├── DTE-FDM/
    │   ├── MFLM/
    │   └── DTG.pth
    └── sam_vit_h_4b8939.pth

Download the weights from their official sources and verify that the files are present before inference.

8.3 Prepare Docker

Install Docker and configure NVIDIA GPU access according to the official installation instructions for your operating system and driver.

The upstream FakeShield repository provides the following reference images:

zhipeixu/dte-fdm:v1.0
zhipeixu/mflm:v1.0

The Web Warriors experiments used a custom MFLM runtime:

fakeshield-mflm-runtime:1.1

That custom image must be built from the corresponding Dockerfile and dependency configuration. It is not interchangeable with the upstream image unless the required modifications have been applied.

8.4 Configure local paths

Set the paths for:

- The FakeShield repository.
- Dataset images and ground-truth masks.
- DTE-FDM output files.
- MFLM output files.
- Model checkpoints.
- Model offloading storage.

Use environment variables, command-line arguments, or repository-relative paths where supported. Do not copy the reference workstation's absolute paths into a different environment.

9. Inference Pipeline

The baseline inference pipeline runs DTE-FDM first and then supplies its output to MFLM.

9.1 DTE-FDM inference

The following is the reference invocation used during setup. Replace the host paths and image path with the locations in your environment.

docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/ml_project/FakeShield/DTE-FDM \
  -v /path/to/ml_project:/workspace/ml_project \
  -w /workspace/ml_project/FakeShield \
  zhipeixu/dte-fdm:v1.0 \
  python -m llava.serve.cli \
    --model-path ./weight/fakeshield-v1-22b/DTE-FDM \
    --DTG-path ./weight/fakeshield-v1-22b/DTG.pth \
    --image-path /workspace/ml_project/dataset/aigc/SD_inpaint_val/image/12936.jpg \
    --output-path ./playground/DTE-FDM_12936.jsonl \
    --load-4bit \
    --max-new-tokens 512

The command processes the selected image and saves the DTE-FDM output for the localization stage.

The 4-bit option was important for running DTE-FDM within the available GPU memory.

9.2 MFLM inference

MFLM consumes the DTE-FDM output and generates a localization result.

The reference invocation used during setup was:

docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/ml_project/FakeShield/MFLM \
  -v /path/to/ml_project:/workspace/ml_project \
  -w /workspace/ml_project/FakeShield \
  fakeshield-mflm-runtime:1.1 \
  python MFLM/cli_demo.py \
    --DTE-FDM-output \
      /workspace/ml_project/FakeShield/playground/DTE-FDM_12936.jsonl \
    --MFLM-output \
      /workspace/ml_project/FakeShield/playground/MFLM_12936.jsonl

This invocation assumes the custom runtime is available and the required memory-management modifications have been applied to the code used by the container.

The working directory matters because the original MFLM script expects checkpoint paths relative to the FakeShield repository root.

9.3 Memory-aware MFLM inference

The original MFLM inference path attempts to move the full model onto the GPU. On the reference RTX 3060, this caused CUDA out-of-memory errors.

The engineering modifications investigated and applied during the project included:

- Device mapping and CPU offloading for selected model components.
- A dedicated offload directory for runtime storage.
- Explicit handling of CPU/GPU tensor placement in the grounding and mask-generation paths.
- Adjustments to the placement of the prompt encoder and mask decoder.
- Removal of relevant Accelerate hooks where necessary before moving selected modules.

These modifications target inference-time memory management. They do not, by themselves, change the underlying FakeShield localization algorithm.

The modified MFLM pipeline subsequently generated a mask in a Photoshop-domain test. That result established that mask generation was possible with the adjusted implementation, but it did not establish that the mask was accurate. The baseline evaluation results reported below are the relevant quantitative measurements for the DeepFake experiment.

Device placement and Accelerate hooks are sensitive to library versions. The implementation should therefore be validated against the installed versions of PyTorch, Transformers, and Accelerate before being ported to another machine.

10. Evaluation Methodology

The evaluation compares predicted manipulation masks with their matching ground-truth masks.

The implemented evaluation procedure:

1. Identifies the selected ground-truth mask and corresponding predicted mask.
2. Converts each mask to grayscale.
3. Binarizes the ground-truth and predicted masks using the configured thresholds.
4. Computes the pixel-level confusion counts.
5. Calculates IoU and pixel-level F1 for each successfully evaluated image.
6. Exports per-image metrics to CSV.
7. Calculates arithmetic means over the successfully evaluated images.

10.1 Intersection over Union

Intersection over Union measures the overlap between the predicted manipulated region and the ground-truth manipulated region.

[
\mathrm{IoU} = \frac{TP}{TP + FP + FN}
]

10.2 Pixel-level F1

Pixel-level F1 measures the balance between precision and recall for manipulated pixels.

[
\mathrm{Pixel\ F1} = \frac{2TP}{2TP + FP + FN}
]

Where:

- TP: Pixels correctly classified as manipulated.
- FP: Pixels predicted as manipulated that are not manipulated in the ground truth.
- FN: Ground-truth manipulated pixels missed by the prediction.
- TN: Pixels correctly classified as unmanipulated.

Higher IoU and pixel-level F1 values indicate better mask overlap.

10.3 Mask preprocessing

The current evaluation implementation uses these thresholds:

Mask| Binarization rule
Ground-truth mask| Grayscale intensity greater than 25.5
Predicted mask| Grayscale intensity greater than 127

The reported mean values are arithmetic averages of the per-image IoU and pixel-level F1 scores.

These preprocessing choices are part of the current implementation. Exact equivalence with the original paper's full evaluation protocol has not been independently established.

10.4 Evaluation output

Per-image metrics are recorded in CSV files. The reported evaluation records include fields such as:

- Image filename.
- TP, FP, FN, and TN.
- Ground-truth foreground pixel count.
- Predicted foreground pixel count.
- IoU.
- Pixel-level F1.

Timestamped output filenames are preferable because they preserve previous experimental results and make different runs easier to audit.

Important: Images with missing predictions must be counted and reported separately. A missing prediction is not a successful localization result.

11. Experimental Results

The baseline experiments were conducted on a small subset first and then expanded to a 500-image test bed.

11.1 Quantitative comparison

Evaluation metric| Original paper| 20-image experiment| 500-image experiment
Test images| Approximately 1,000 fake images in the paper's setting| 20| 500
Successfully evaluated images| Not directly comparable| 20| 496
Missing predictions| Not reported here| 0| 4
Mean IoU| 0.14| 0.147683| 0.155511
Mean pixel-level F1| 0.22| 0.223832| 0.242786

The original paper's figures are reference values from its DeepFake-domain evaluation. The two experimental columns contain results from the Web Warriors implementation and should be interpreted in the context of their own subsets and evaluation configuration.

11.2 Observations

20-image experiment

The initial experiment achieved:

- Mean IoU: 0.147683.
- Mean pixel-level F1: 0.223832.
- Successfully evaluated images: 20 out of 20.

This experiment provided a small-scale check of the inference and evaluation workflow.

500-image experiment

The expanded experiment achieved:

- Mean IoU: 0.155511.
- Mean pixel-level F1: 0.242786.
- Successfully evaluated images: 496 out of 500.
- Missing predictions: 4.

The larger experiment provides a broader sample of the selected FaceAPP validation subset. However, four images did not have predictions at the time of evaluation.

11.3 Interpretation

The experimental localization scores are broadly comparable to the original paper's reported DeepFake-domain results. The measured means are numerically higher in these experiments.

This does not demonstrate that the Web Warriors implementation outperforms the original FakeShield method. Differences in evaluation subsets, successfully evaluated image counts, mask preprocessing, inference configuration, and evaluation protocols can affect the results.

The reported 500-image metrics apply to the 496 successfully evaluated images, not all 500 selected images.

These results establish a useful baseline for subsequent experiments. A stronger reproduction claim requires verification of the evaluation protocol and completion of the missing predictions.

12. Engineering Findings

Several practical findings emerged from the reproduction work.

12.1 DTE-FDM inference under constrained GPU memory

DTE-FDM successfully processed an input image on the reference RTX 3060 using 4-bit model loading. The generated output included a manipulation judgment and textual analysis.

12.2 Separate Docker environments

Maintaining separate DTE-FDM and MFLM runtimes avoided repeatedly mixing incompatible dependency versions. This was especially useful for the older MMCV/MMDetection stack required by MFLM.

12.3 MFLM memory management

The original MFLM inference path exceeded available GPU memory when attempting to place the entire model on the GPU. Selected device-placement and CPU-offloading modifications enabled subsequent mask generation in a test case.

The exact memory requirements remain dependent on model configuration, input processing, dependency versions, and the runtime environment.

12.4 Incremental inference

Long-running MFLM sessions exhibited intermittent errors and stalls during experimentation. Restarting the MFLM process after a limited batch of new inference attempts was adopted as a practical recovery strategy.

Skipping outputs that already exist also allows interrupted experiments to resume without recomputing completed images.

These observations describe the behavior seen during the experiments. They do not establish a definitive cause for every intermittent inference failure.

12.5 Traceable experimental outputs

Separating model outputs, predicted masks, and metric CSV files improves experiment organization and makes it easier to associate results with specific runs.

13. Proposed Research Extension

The broader research project is titled:

Face-Structure-Aware Localization Refinement for Deepfake Forgery Detection: Extending FakeShield's MFLM

The proposed extension targets the spatial-localization weakness of FakeShield on facial manipulations.

The original detection and explanation components are intended to remain frozen, while changes focus on the MFLM localization stage.

13.1 Facial-Region Prior Constructor

The proposed component uses two sources of information:

1. Facial-region mentions in DTE-FDM's generated explanation.
2. Face bounding boxes and five-point facial landmarks provided with the DFFD dataset.

A rule-based text analysis step identifies facial regions such as the eyes, nose, and mouth. The corresponding facial metadata is then used to derive image-specific regions associated with those mentions.

This avoids requiring a separate facial-landmark model if the relevant metadata is available and validated for the selected images.

The landmark ordering, coordinate conventions, and accuracy on manipulated images must be verified before using this information in the localization pipeline.

13.2 Coarse-to-fine localization

The proposed method introduces a refinement stage:

Input Image
     |
     v
Original DTE-FDM
     |
     v
Manipulation Explanation
     |
     v
Facial-Region Prior Constructor
     |
     v
Original MFLM / SAM
     |
     v
Coarse Localization Mask
     |
     v
Facial-Region Refinement
     |
     v
Refined Full-Image Mask

The intent is to provide explicit spatial guidance to the segmentation process and refine the predicted mask around relevant facial regions.

The prior should act as soft guidance rather than a hard restriction. Some manipulations affect broad facial areas, and restricting the prediction to a single named facial part could exclude genuinely manipulated pixels.

13.3 LoRA fine-tuning and ablations

The proposed training and evaluation plan includes:

1. Original baseline: Evaluate the released FakeShield localization pipeline.
2. Training-only comparison: Measure the effect of additional localization training without the facial-structure prior.
3. Full proposed method: Evaluate the facial-region prior and refinement strategy with the corresponding training configuration.

All methods should use the same evaluation split, ground-truth masks, and metric implementation.

The primary outcome measures are changes in mean IoU and mean pixel-level F1. Qualitative comparisons between ground-truth masks, baseline predictions, and refined predictions should also be included.

These are planned experiments. The baseline scores in this README must not be presented as evidence that the proposed extension has already improved localization.

14. Limitations

The current work has several limitations.

1. Partial evaluation: Four predictions were missing from the 500-image experiment, leaving 496 successfully evaluated images.

2. Restricted dataset scope: The reported localization results concern manipulated FaceAPP images with ground-truth masks. They do not constitute a complete evaluation of all FakeShield manipulation domains or a balanced real-versus-fake detection benchmark.

3. Evaluation compatibility: The equivalence of the current mask binarization and metric implementation to the original paper's full evaluation protocol has not been completely verified.

4. Hardware constraints: The reference GPU has 12 GB VRAM. MFLM required implementation-level memory management to generate masks in the constrained environment.

5. Runtime sensitivity: The older MMCV/MMDetection dependency stack and Accelerate device hooks may behave differently across software versions.

6. No completed extension evaluation: The proposed facial-region prior, coarse-to-fine refinement, and associated LoRA fine-tuning are not represented by the reported baseline metrics.

7. Reproduction versus improvement: Numerically higher scores on a different or partially evaluated subset do not establish that the reproduced implementation is better than the original method.

These limitations should be considered when interpreting the baseline and planning subsequent research.

15. Reproducibility Checklist

A reproduction should document the following information.

Environment

- [ ] Operating system and version.
- [ ] GPU model and available VRAM.
- [ ] NVIDIA driver and CUDA configuration.
- [ ] Python and PyTorch versions.
- [ ] Transformers and Accelerate versions.
- [ ] MMCV/MMDetection versions.
- [ ] Docker image names and versions.

Model and data

- [ ] FakeShield repository commit.
- [ ] Model checkpoint source and configuration.
- [ ] Dataset source and selected split.
- [ ] Number of selected images.
- [ ] Image-mask filename pairing.
- [ ] Inference precision and device-placement configuration.
- [ ] Output and offload directories.

Evaluation

- [ ] Ground-truth and prediction mask thresholds.
- [ ] Missing prediction count.
- [ ] Number of successfully evaluated images.
- [ ] Per-image metric CSV.
- [ ] Aggregate mean IoU and pixel-level F1.
- [ ] Evaluation logs and error records.
- [ ] Qualitative comparison samples.

Reporting

- [ ] Clearly distinguish original paper values from locally measured results.
- [ ] Identify the exact subset used for each reported metric.
- [ ] Record any deviations from the upstream implementation.
- [ ] Avoid treating missing predictions as successful evaluations.
- [ ] Report proposed-method results separately from baseline results.

16. Future Work

The next steps for this project are:

1. Complete the four missing predictions from the 500-image experiment.
2. Verify the evaluation implementation against the original protocol wherever possible.
3. Record the exact software versions, model configurations, and dataset subset used for the reported results.
4. Perform qualitative comparisons of predictions against ground-truth masks.
5. Validate DFFD facial bounding boxes and landmarks on authentic and manipulated images.
6. Implement the Facial-Region Prior Constructor within MFLM.
7. Integrate coarse-to-fine localization refinement.
8. Fine-tune the relevant localization components using the planned LoRA configuration.
9. Conduct controlled ablation studies.
10. Compare the baseline and proposed method on identical evaluation samples.
11. Expand the evaluation when compute resources and verified predictions are available.

17. References and Acknowledgements

Original research

Xu, Z., Zhang, X., Li, R., Tang, Z., Huang, Q., and Zhang, J. FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models. International Conference on Learning Representations (ICLR), 2025.

- Paper: https://arxiv.org/abs/2410.02761
- Project website: https://zhipeixu.github.io/projects/FakeShield/

Official implementation

- Original FakeShield repository: https://github.com/zhipeixu/FakeShield
- Pretrained model weights: https://huggingface.co/zhipeixu/fakeshield-v1-22b

Related work

- LLaVA: https://github.com/haotian-liu/LLaVA
- Segment Anything Model: https://github.com/facebookresearch/segment-anything
- LISA: https://github.com/JIA-Lab-research/LISA

The project builds on the publicly released FakeShield framework and its associated research. Please consult the upstream repository for the original code, dependency requirements, model assets, and license information.

18. Citation

If you use the original FakeShield method, please cite the original paper:

@inproceedings{xu2024fakeshield,
  title={FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models},
  author={Xu, Zhipei and Zhang, Xuanyu and Li, Runyi and Tang, Zecheng and Huang, Qing and Zhang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2025}
}

For work derived from this repository, please also acknowledge the Web Warriors project and identify the specific baseline implementation, evaluation configuration, and any modifications used.

---

Web Warriors — FakeShield Reproduction and DeepFake Localization

This repository documents the engineering work, baseline measurements, and reproducibility considerations for evaluating FakeShield's DeepFake localization capability. Its purpose is to provide a traceable baseline for future experiments in facial-structure-aware manipulation localization, without conflating completed results with proposed research contributions.
