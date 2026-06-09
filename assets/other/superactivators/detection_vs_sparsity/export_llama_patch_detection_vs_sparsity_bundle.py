import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "Experiments"
OUTPUT_PATH = Path(__file__).resolve().parent / "llama_patch_detection_vs_sparsity.json"

PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
DATASETS = [
    "Coco",
    "Broden-OpenSurfaces",
    "Broden-Pascal",
    "iSarcasm",
    "GoEmotions",
]

DATASET_DISPLAY_NAMES = {
    "Coco": "COCO",
    "Broden-OpenSurfaces": "OpenSurfaces",
    "Broden-Pascal": "Pascal",
    "iSarcasm": "iSarcasm",
    "GoEmotions": "GoEmotions",
}

YMAX_OVERRIDES = {
    "Coco": 0.9,
    "Broden-OpenSurfaces": 0.6,
    "iSarcasm": 1.0,
    "GoEmotions": 0.5,
}

SERIES_STYLES = [
    {"key": "avg", "label": "Avg", "color": "#0072B2", "borderDash": []},
    {"key": "linsep", "label": "LinSep", "color": "#E69F00", "borderDash": []},
    {"key": "kmeans", "label": "KMeans", "color": "#009E73", "borderDash": []},
    {
        "key": "linsep_kmeans",
        "label": "LinSep KMeans",
        "color": "#CC79A7",
        "borderDash": [],
    },
    {
        "key": "prompt",
        "label": "Prompt",
        "color": "#8B4513",
        "borderDash": [10, 4, 2, 4],
        "isBaseline": True,
    },
    {
        "key": "random",
        "label": "Random",
        "color": "#808080",
        "borderDash": [10, 4, 2, 4],
        "isBaseline": True,
    },
]

DATASET_TO_CONCEPTS = {
    "Coco": [
        "accessory",
        "animal",
        "appliance",
        "bench",
        "book",
        "bottle",
        "bowl",
        "bus",
        "car",
        "chair",
        "couch",
        "cup",
        "dining table",
        "electronic",
        "food",
        "furniture",
        "indoor",
        "kitchen",
        "motorcycle",
        "outdoor",
        "person",
        "pizza",
        "potted plant",
        "sports",
        "train",
        "truck",
        "tv",
        "umbrella",
        "vehicle",
    ],
    "Broden-Pascal": [
        "object::airplane",
        "object::bicycle",
        "object::bird",
        "object::boat",
        "object::body",
        "object::book",
        "object::building",
        "object::bus",
        "object::cap",
        "object::car",
        "object::cat",
        "object::cup",
        "object::dog",
        "object::door",
        "object::ear",
        "object::engine",
        "object::grass",
        "object::hair",
        "object::horse",
        "object::leg",
        "object::mirror",
        "object::motorbike",
        "object::mountain",
        "object::painting",
        "object::person",
        "object::pottedplant",
        "object::saddle",
        "object::screen",
        "object::sky",
        "object::sofa",
        "object::table",
        "object::track",
        "object::train",
        "object::tvmonitor",
        "object::wheel",
        "object::wood",
        "part::arm",
        "part::bag",
        "part::beak",
        "part::bottle",
        "part::box",
        "part::cabinet",
        "part::ceiling",
        "part::chain wheel",
        "part::chair",
        "part::coach",
        "part::curtain",
        "part::eye",
        "part::eyebrow",
        "part::fabric",
        "part::fence",
        "part::floor",
        "part::foot",
        "part::ground",
        "part::hand",
        "part::handle bar",
        "part::head",
        "part::headlight",
        "part::light",
        "part::mouth",
        "part::muzzle",
        "part::neck",
        "part::nose",
        "part::paw",
        "part::plant",
        "part::plate",
        "part::plaything",
        "part::pole",
        "part::pot",
        "part::road",
        "part::rock",
        "part::rope",
        "part::shelves",
        "part::sidewalk",
        "part::signboard",
        "part::stern",
        "part::tail",
        "part::torso",
        "part::tree",
        "part::wall",
        "part::water",
        "part::windowpane",
        "part::wing",
    ],
    "Broden-OpenSurfaces": [
        "material::brick",
        "material::cardboard",
        "material::carpet",
        "material::ceramic",
        "material::concrete",
        "material::fabric",
        "material::food",
        "material::fur",
        "material::glass",
        "material::granite",
        "material::hair",
        "material::laminate",
        "material::leather",
        "material::metal",
        "material::mirror",
        "material::painted",
        "material::paper",
        "material::plastic-clear",
        "material::plastic-opaque",
        "material::rock",
        "material::rubber",
        "material::skin",
        "material::tile",
        "material::wallpaper",
        "material::wicker",
        "material::wood",
    ],
    "iSarcasm": ["sarcastic"],
    "GoEmotions": [
        "confusion",
        "joy",
        "sadness",
        "anger",
        "love",
        "caring",
        "optimism",
        "amusement",
        "curiosity",
        "disapproval",
        "approval",
        "annoyance",
        "gratitude",
        "admiration",
    ],
}


def is_text_dataset(dataset_name: str) -> bool:
    return (
        dataset_name in {"Stanford-Tree-Bank", "Sarcasm", "iSarcasm", "GoEmotions", "HateXplain"}
        or "Sarcasm" in dataset_name
        or "Emotion" in dataset_name
    )


def filter_concept_dict(concept_dict: Dict[str, object], dataset_name: str) -> Dict[str, object]:
    allowed = set(DATASET_TO_CONCEPTS.get(dataset_name, []))
    if not allowed:
        return concept_dict
    return {key: value for key, value in concept_dict.items() if key in allowed}


def torch_load_compat(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def get_gt_path(dataset_name: str, model_name: str, split: str) -> Path:
    if is_text_dataset(dataset_name):
        if model_name == "Llama":
            return EXPERIMENTS_ROOT / "GT_Samples" / dataset_name / f"gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        if model_name == "Gemma":
            return EXPERIMENTS_ROOT / "GT_Samples" / dataset_name / f"gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        if model_name == "Qwen":
            return EXPERIMENTS_ROOT / "GT_Samples" / dataset_name / f"gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    if model_name == "CLIP":
        return EXPERIMENTS_ROOT / "GT_Samples" / dataset_name / f"gt_samples_per_concept_{split}_inputsize_(224, 224).pt"
    if model_name == "Llama":
        return EXPERIMENTS_ROOT / "GT_Samples" / dataset_name / f"gt_samples_per_concept_{split}_inputsize_(560, 560).pt"
    raise ValueError(f"Unsupported model: {model_name}")


def load_gt_samples(dataset_name: str, model_name: str, split: str) -> Dict[str, object]:
    gt_samples = torch_load_compat(get_gt_path(dataset_name, model_name, split))
    return filter_concept_dict(gt_samples, dataset_name)


def get_per_concept_prompt_scores(dataset_name: str, model_name: str, metric: str) -> Dict[str, float]:
    candidate_dirs = [
        EXPERIMENTS_ROOT / "Quant_Results" / dataset_name,
        EXPERIMENTS_ROOT / "Quant_Results_with_CI" / dataset_name,
        EXPERIMENTS_ROOT / "prompt_results" / dataset_name,
    ]
    search_roots = [root for root in candidate_dirs if root.is_dir()]

    candidate_paths: List[Path] = []
    for root in search_roots:
        for pattern in ("*prompt*.csv", "*Prompt*.csv", "*f1_scores*.csv"):
            candidate_paths.extend(sorted(root.glob(pattern)))

    if not candidate_paths:
        return {}

    model_key = model_name.lower()
    dataset_key = dataset_name.lower()

    filtered = [path for path in candidate_paths if model_key in path.name.lower()]
    if filtered:
        candidate_paths = filtered

    filtered = [path for path in candidate_paths if dataset_key in path.name.lower()]
    if filtered:
        candidate_paths = filtered

    csv_path = candidate_paths[0]
    df = pd.read_csv(csv_path)

    if metric == "precision" and metric not in df.columns and {"tp", "fp"} <= set(df.columns):
        df["precision"] = df.apply(
            lambda row: row["tp"] / (row["tp"] + row["fp"]) if (row["tp"] + row["fp"]) > 0 else 0.0,
            axis=1,
        )

    if metric == "recall" and metric not in df.columns and {"tp", "fn"} <= set(df.columns):
        df["recall"] = df.apply(
            lambda row: row["tp"] / (row["tp"] + row["fn"]) if (row["tp"] + row["fn"]) > 0 else 0.0,
            axis=1,
        )

    if metric not in df.columns:
        return {}

    return filter_concept_dict(dict(zip(df["concept"], df[metric])), dataset_name)


def get_weighted_prompt_score(dataset_name: str, model_name: str, metric: str, split: str) -> float:
    per_concept = get_per_concept_prompt_scores(dataset_name, model_name, metric)
    gt_samples = load_gt_samples(dataset_name, model_name, split)

    if metric in {"fp", "fn", "tp", "tn"}:
        return float(sum(value for concept, value in per_concept.items() if concept in gt_samples))

    weighted_sum = 0.0
    total_samples = 0
    for concept, score in per_concept.items():
        if concept in gt_samples:
            count = len(gt_samples[concept])
            weighted_sum += float(score) * count
            total_samples += count

    return weighted_sum / total_samples if total_samples else 0.0


def filter_unsupervised_detection_metrics(
    detection_metrics: pd.DataFrame,
    best_clusters_per_concept: Dict[str, object],
) -> pd.DataFrame:
    filtered_rows = []
    for concept in best_clusters_per_concept:
        cluster_info = best_clusters_per_concept[concept]
        if isinstance(cluster_info, dict):
            cluster_id = cluster_info.get("best_cluster")
        else:
            cluster_id = cluster_info
        if cluster_id is None:
            continue
        row = detection_metrics[detection_metrics["concept"] == f"('{concept}', '{cluster_id}')"]
        if row.empty:
            continue
        simplified = row.iloc[0].copy()
        simplified["concept"] = concept
        filtered_rows.append(simplified)

    return pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame()


def compute_weighted_score(
    detection_metrics: pd.DataFrame,
    gt_samples_per_concept: Dict[str, object],
    metric: str,
) -> float:
    if detection_metrics.empty:
        return 0.0

    detection_metrics = detection_metrics[detection_metrics["concept"].isin(gt_samples_per_concept)]
    total = sum(len(gt_samples_per_concept[concept]) for concept in detection_metrics["concept"])
    if total <= 0:
        return 0.0

    score = sum(
        float(row[metric]) * len(gt_samples_per_concept[row["concept"]])
        for _, row in detection_metrics.iterrows()
    )
    return score / total


def build_con_labels(model_name: str, sample_type: str, percentthrumodel: int) -> Dict[str, str]:
    n_clusters = 1000 if sample_type == "patch" else 50
    return {
        "avg": f"{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}",
        "linsep": f"{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}",
        "kmeans": f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}",
        "linsep_kmeans": f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}",
    }


def load_concept_series(
    dataset_name: str,
    gt_samples_per_concept: Dict[str, object],
    metric: str,
    concept_key: str,
    con_label: str,
) -> List[float]:
    scores: List[float] = []
    for percentile in PERCENTILES:
        if concept_key in {"kmeans", "linsep_kmeans"}:
            csv_path = (
                EXPERIMENTS_ROOT
                / "Quant_Results"
                / dataset_name
                / f"detectionmetrics_allpairs_per_{percentile}_{con_label}.csv"
            )
            best_clusters_path = (
                EXPERIMENTS_ROOT
                / "Unsupervised_Matches"
                / dataset_name
                / f"bestdetects_{con_label}.pt"
            )
            if not csv_path.exists() or not best_clusters_path.exists():
                scores.append(0.0)
                continue
            detection_metrics = pd.read_csv(csv_path)
            best_clusters = torch_load_compat(best_clusters_path)
            detection_metrics = filter_unsupervised_detection_metrics(detection_metrics, best_clusters)
        else:
            pt_path = (
                EXPERIMENTS_ROOT
                / "Quant_Results"
                / dataset_name
                / f"detectionmetrics_per_{percentile}_{con_label}.pt"
            )
            if not pt_path.exists():
                scores.append(0.0)
                continue
            detection_metrics = torch_load_compat(pt_path)

        scores.append(round(compute_weighted_score(detection_metrics, gt_samples_per_concept, metric), 6))

    return scores


def load_random_baseline(
    dataset_name: str,
    gt_samples_per_concept: Dict[str, object],
    metric: str,
) -> float:
    baseline_path = (
        EXPERIMENTS_ROOT / "Quant_Results" / dataset_name / "random_Llama_cls_baseline.csv"
    )
    if not baseline_path.exists():
        return 0.0
    df = pd.read_csv(baseline_path)
    df = df[df["concept"].isin(gt_samples_per_concept)]
    return round(compute_weighted_score(df, gt_samples_per_concept, metric), 6)


def nice_ymax(value: float) -> float:
    if value <= 0.5:
        step = 0.05
    else:
        step = 0.1
    scaled = max(step, value + step * 0.35)
    steps = int(scaled / step)
    if steps * step < scaled:
        steps += 1
    return round(steps * step, 2)


def export_bundle() -> Dict[str, object]:
    model_name = "Llama"
    sample_type = "patch"
    metric = "f1"
    split = "test"
    percentthrumodel = 100
    con_labels = build_con_labels(model_name, sample_type, percentthrumodel)

    charts = []
    for dataset_name in DATASETS:
        gt_samples = load_gt_samples(dataset_name, model_name, split)

        series = {
            "avg": load_concept_series(dataset_name, gt_samples, metric, "avg", con_labels["avg"]),
            "linsep": load_concept_series(dataset_name, gt_samples, metric, "linsep", con_labels["linsep"]),
            "kmeans": load_concept_series(dataset_name, gt_samples, metric, "kmeans", con_labels["kmeans"]),
            "linsep_kmeans": load_concept_series(
                dataset_name,
                gt_samples,
                metric,
                "linsep_kmeans",
                con_labels["linsep_kmeans"],
            ),
            "prompt": round(get_weighted_prompt_score(dataset_name, model_name, metric, split), 6),
            "random": load_random_baseline(dataset_name, gt_samples, metric),
        }

        all_values = []
        for key, value in series.items():
            if isinstance(value, list):
                all_values.extend(value)
            else:
                all_values.append(value)
        ymax = YMAX_OVERRIDES.get(dataset_name, nice_ymax(max(all_values) if all_values else 0.1))

        chart_series = []
        for style in SERIES_STYLES:
            key = style["key"]
            raw = series[key]
            if style.get("isBaseline"):
                baseline_value = float(raw)
                points = [{"x": 0.0, "y": baseline_value}, {"x": 1.0, "y": baseline_value}]
            else:
                points = [
                    {"x": percentile, "y": float(score)}
                    for percentile, score in zip(PERCENTILES, raw)
                ]
            chart_series.append(
                {
                    "key": key,
                    "label": style["label"],
                    "color": style["color"],
                    "borderDash": style["borderDash"],
                    "isBaseline": bool(style.get("isBaseline")),
                    "points": points,
                }
            )

        display_name = DATASET_DISPLAY_NAMES[dataset_name]
        charts.append(
            {
                "datasetKey": dataset_name,
                "displayName": display_name,
                "title": f"{display_name} Detection vs \u03b4",
                "yMin": 0.0,
                "yMax": ymax,
                "paperFigure": f"Figs/Paper_Figs/patch_{dataset_name}_Llama_dataset_detection.png",
                "series": chart_series,
            }
        )

    return {
        "modelName": model_name,
        "sampleType": sample_type,
        "metric": metric,
        "split": split,
        "percentthrumodel": percentthrumodel,
        "percentiles": PERCENTILES,
        "charts": charts,
    }


def main() -> None:
    bundle = export_bundle()
    OUTPUT_PATH.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH}")
    for chart in bundle["charts"]:
        print(
            f"{chart['datasetKey']}: yMax={chart['yMax']}, "
            f"series={', '.join(series['label'] for series in chart['series'])}"
        )


if __name__ == "__main__":
    main()
