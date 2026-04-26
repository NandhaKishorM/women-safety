# ShoeGuard: Edge AI Chip for Women Safety

ShoeGuard is a shoe-mounted edge AI chip designed for women's safety. It fires a panic alarm and sends location alerts (via a NEO-6M GPS module and SIM800L GSM modem) to relatives when the wearer performs a deliberate repeated kick pattern.

## Quick Links

- **Hugging Face Model & Dataset:** [convaiinnovations/shoeguard-safety-slm-q4_k_m](https://huggingface.co/convaiinnovations/shoeguard-safety-slm-q4_k_m)
- **Colab Fine-Tuning Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1c8wo9xHxf0vMDwssOPaWS1uWyFAuS0yv#scrollTo=Ct3VIZ-RH0zq)

## Overview

The whole pipeline lives on the smallest edge AI chip powered by an RP2040/RP2350 (7x7mm QFN) class device acting as the brain. An MPU6050 inertial measurement unit (IMU) feeds 50 Hz, six-axis samples into a two-second ring buffer. A 14-dimensional summary feature vector is extracted and fed into a fine-tuned **SmolLM2-135M Instruct** model running as an 88 MB Q4_K_M GGUF file using `llama.cpp`. A trigger engine debounces the model's output labels over a six-second history to determine if the alarm should be triggered. The system is powered by an impact-regenerative battery (powered by kicks) and embedded in a tiny custom PCB fixed by a magnet.

## Features

- **Hands-Free & Deniable Trigger:** The wearer can stomp out a clear repeated kick sequence to summon help without needing their hands.
- **On-Chip SLM:** Utilizes a Small Language Model (SmolLM2-135M) for action classification directly on the edge.
- **Physics-Simulated Dataset:** Because gathering real-world data of attacks is impractical and unethical, we built a physics-simulated MPU6050 dataset using a three-link kinematic chain of the leg and biomechanical joint angle trajectories.
- **GPS & GSM Integration:** When triggered, the buzzer sounds and the chip sends an SMS containing the wearer's current GPS coordinates to a predefined list of contacts.
- **Privacy First:** All processing happens locally on the device. No audio or video is captured. Data only leaves the device when an alert is triggered.

## Repository Contents

- `code/`: Contains the physics simulator, data generation scripts, evaluation code, feature extraction, and the trigger engine.
  - `generate_dataset.py` & `physics_sim.py`: Physics-simulated MPU6050 dataset generation.
  - `trigger.py` & `feature_extractor.py`: On-chip inference wrapper and trigger engine.
  - `safety_slm_finetune.ipynb`: Q-LoRA fine-tuning notebook for the SmolLM2 model. ([Open in Colab](https://colab.research.google.com/drive/1c8wo9xHxf0vMDwssOPaWS1uWyFAuS0yv#scrollTo=Ct3VIZ-RH0zq))
  - `make_plots.py`: Generates the figures used in the paper.

## Authors

- Nandakishor M
- Sruthi K
- Nisha M
- Rajitna B

*Convai Innovations Research Lab*
