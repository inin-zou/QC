# Coding challenge — Dataset Augmentation for LeRobot Datasets

## Task

Build a dataset augmentation tool for a [LeRobot v3](https://github.com/huggingface/lerobot) dataset. 


Your tool should **automatically upload the augmented dataset to Hugging Face Hub** as part of its run, and print a direct visualizer link, e.g.:
  ```
  https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F<your-username>%2F<dataset-name>%2Fepisode_0
  ```

You have **up to 6 hours**. Spend the time however you see fit — you can go deep on one idea or ship several smaller tools. There is no single right answer.


## What to deliver

Before the 6 hours are up, email us the link to your **public GitHub repo** containing at least:

- Working code (script, CLI, notebook, or library — your choice)
- A short README explaining what it does and how to run it

## Hints

- A LeRobot v3 dataset is a Hugging Face dataset with episodes stored as `.parquet` files plus a `meta/` folder containing `info.json`, `tasks.json`, and optionally `stats.json`. Videos live under `videos/`.
- You can inspect any public dataset with the visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset
- A good starting dataset to work with: [`lerobot/aloha_static_cups_open`](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Flerobot%2Faloha_static_cups_open%2Fepisode_0%3Ft%3D2) or any dataset on the [LeRobot hub](https://huggingface.co/lerobot).
- It could for example be tools to improve, filter, multiply, increase variation, change background etc etc of a dataset, the sky is the limit.