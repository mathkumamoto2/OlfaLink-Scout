# Verification — Windows Desktop release (2026-10-30)

## What was tested here

- `python -m pytest -q tests`: **12 passed** (pure numerical functions, URDF structure, mocked Genesis/Gymnasium API integration, command argument checks).
- `python -m compileall -q scout_genesis tests`: syntax check passed.
- `python -m scout_genesis --help`: CLI arguments and entrypoint displayed correctly.
- `report.measured_data()` rejected missing measured output as intended.
- Japanese PDF generator was tested with **temporary explicitly synthetic fixture data**, and a one-page PDF was rendered and visually inspected. The synthetic PDF and fixture data are **not included** in this project or represented as real results.
- The project archive is checked by reading every ZIP member and running ZIP CRC integrity check.

## NOT validated on this machine

- `genesis-world==1.4.0` dependency cannot be installed here because the execution container has no network access and Genesis is not preinstalled.
- Native Windows / NVIDIA driver / OpenGL environment is not available.
- True Genesis wheel-ground contact, joints, obstacle contact, 3-D MP4 capture, and PPO training/convergence are **unverified in the actual engine**.
- Success rate / reward improvements are **not measured**. Evaluation must be run on the actual desktop.

## Desktop acceptance gate

1. Setup and doctor passes.
2. `smoke` reports forward progress over 0.01 m and yaw change over 0.03 rad.
3. `train` saves `outputs/ppo_genesis.zip`.
4. `evaluate` saves genuine metrics and reports PPO vs random, even if PPO does not outperform random.
5. `demo --policy ppo --record` writes a nonempty MP4 with the **trained** model.
6. Visually inspect MP4 and report prior to submission; remove personal information.

Never report tests 3-5 as passed until you have the outputs from that machine.
