from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROUTES = {
    "asia_europe": {"origin": "Shanghai", "destination": "Rotterdam", "transit_days": 25, "weather_exposure": 0.65},
    "asia_west_coast": {"origin": "Singapore", "destination": "Los Angeles", "transit_days": 19, "weather_exposure": 0.8},
    "europe_east_coast": {"origin": "Hamburg", "destination": "New York", "transit_days": 13, "weather_exposure": 0.55},
    "asia_mediterranean": {"origin": "Busan", "destination": "Valencia", "transit_days": 29, "weather_exposure": 0.7},
}

LATE_THRESHOLD_HOURS = 24.0
SCHEDULED_PORT_PROCESSING_HOURS = 12.0
SCHEDULED_CUSTOMS_HOURS = 6.0


def generate_shipments(count: int = 240, seed: int = 42) -> list[dict[str, Any]]:
    if count < 30:
        raise ValueError("Generate at least 30 shipments for a useful train/test split.")

    rng = random.Random(seed)
    shipments = []
    start_date = datetime(2025, 1, 1, 8)

    for index in range(count):
        route_name = rng.choice(list(ROUTES))
        route = ROUTES[route_name]
        scheduled_departure = start_date + timedelta(days=rng.randint(0, 330), hours=rng.randint(0, 23))
        congestion = rng.betavariate(2.0, 2.2)
        weather = rng.betavariate(1.7, 3.2)
        departure_delay = max(0.0, rng.gauss(1.5 + congestion * 2.5, 3.0))
        sailing_delay = max(0.0, rng.gauss(weather * route["weather_exposure"] * 13, 3.5))
        port_wait = max(0.0, rng.gauss(2.0 + congestion * 22, 5.0))
        processing_hours = max(4.0, rng.gauss(10.0 + congestion * 8, 3.0))
        customs_delay = max(0.0, rng.gauss(1.0 + congestion * 3.5, 2.5))
        processing_delay = processing_hours - SCHEDULED_PORT_PROCESSING_HOURS
        checkpoint_delay = departure_delay + sailing_delay + port_wait + processing_delay + customs_delay
        downstream_delay = (
            -1.5
            + congestion * 3.0
            + weather * route["weather_exposure"] * 5.0
            + congestion * processing_hours * 0.045
            + rng.gauss(0, 2.5)
        )
        final_delay = checkpoint_delay + downstream_delay

        scheduled_port_arrival = scheduled_departure + timedelta(days=route["transit_days"])
        scheduled_customs_clearance = scheduled_port_arrival + timedelta(
            hours=SCHEDULED_PORT_PROCESSING_HOURS + SCHEDULED_CUSTOMS_HOURS
        )
        scheduled_arrival = scheduled_customs_clearance + timedelta(days=1)
        actual_departure = scheduled_departure + timedelta(hours=departure_delay)
        actual_port_arrival = scheduled_port_arrival + timedelta(hours=departure_delay + sailing_delay)
        actual_customs_clearance = scheduled_customs_clearance + timedelta(hours=checkpoint_delay)
        actual_arrival = scheduled_arrival + timedelta(hours=final_delay)

        shipments.append(
            {
                "shipment_id": f"S{index + 1:04d}",
                "route": route_name,
                "origin": route["origin"],
                "destination": route["destination"],
                "scheduled_departure": scheduled_departure.isoformat(timespec="minutes"),
                "actual_departure": actual_departure.isoformat(timespec="minutes"),
                "scheduled_port_arrival": scheduled_port_arrival.isoformat(timespec="minutes"),
                "actual_port_arrival": actual_port_arrival.isoformat(timespec="minutes"),
                "scheduled_customs_clearance": scheduled_customs_clearance.isoformat(timespec="minutes"),
                "actual_customs_clearance": actual_customs_clearance.isoformat(timespec="minutes"),
                "scheduled_arrival": scheduled_arrival.isoformat(timespec="minutes"),
                "actual_arrival": actual_arrival.isoformat(timespec="minutes"),
                "departure_delay_hours": round(departure_delay, 2),
                "sailing_delay_hours": round(sailing_delay, 2),
                "port_wait_hours": round(port_wait, 2),
                "port_processing_hours": round(processing_hours, 2),
                "customs_delay_hours": round(customs_delay, 2),
                "congestion_index": round(congestion, 3),
                "weather_severity": round(weather, 3),
                "checkpoint_delay_hours": round(checkpoint_delay, 2),
                "final_delay_hours": round(final_delay, 2),
            }
        )

    return shipments


def feature_row(shipment: dict[str, Any]) -> dict[str, float | str]:
    route = ROUTES[shipment["route"]]
    departure_delay = float(shipment["departure_delay_hours"])
    sailing_delay = float(shipment["sailing_delay_hours"])
    port_wait = float(shipment["port_wait_hours"])
    processing_hours = float(shipment["port_processing_hours"])
    customs_delay = float(shipment["customs_delay_hours"])
    congestion = float(shipment["congestion_index"])
    weather = float(shipment["weather_severity"])
    checkpoint_delay = float(shipment["checkpoint_delay_hours"])
    scheduled_departure = datetime.fromisoformat(shipment["scheduled_departure"])

    return {
        "route": shipment["route"],
        "departure_delay_hours": departure_delay,
        "sailing_delay_hours": sailing_delay,
        "port_wait_hours": port_wait,
        "port_processing_hours": processing_hours,
        "customs_delay_hours": customs_delay,
        "checkpoint_delay_hours": checkpoint_delay,
        "voyage_delay_growth_hours": sailing_delay,
        "port_pressure_hours": port_wait + processing_hours,
        "congestion_index": congestion,
        "weather_severity": weather,
        "congestion_x_processing": congestion * processing_hours,
        "weather_x_route_exposure": weather * route["weather_exposure"],
        "departure_weekday": float(scheduled_departure.weekday()),
        "departure_month": float(scheduled_departure.month),
    }


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Training data does not contain enough feature variation.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _standardize(
    matrix: list[list[float]], means: list[float] | None = None, scales: list[float] | None = None
) -> tuple[list[list[float]], list[float], list[float]]:
    if means is None:
        means = [sum(row[column] for row in matrix) / len(matrix) for column in range(len(matrix[0]))]
    if scales is None:
        scales = [
            max(1e-9, math.sqrt(sum((row[column] - means[column]) ** 2 for row in matrix) / len(matrix)))
            for column in range(len(matrix[0]))
        ]
    standardized = [
        [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]
        for row in matrix
    ]
    return standardized, means, scales


def _logistic(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _fit_linear(matrix: list[list[float]], targets: list[float], penalty: float = 1.0) -> list[float]:
    width = len(matrix[0]) + 1
    normal = [[0.0] * width for _ in range(width)]
    products = [0.0] * width
    for row, target in zip(matrix, targets):
        values = [1.0, *row]
        for first in range(width):
            products[first] += values[first] * target
            for second in range(width):
                normal[first][second] += values[first] * values[second]
    for index in range(1, width):
        normal[index][index] += penalty
    return _solve(normal, products)


def _fit_logistic(matrix: list[list[float]], labels: list[int]) -> list[float]:
    weights = [0.0] * (len(matrix[0]) + 1)
    sample_count = len(matrix)
    for _ in range(1800):
        gradients = [0.0] * len(weights)
        for row, label in zip(matrix, labels):
            values = [1.0, *row]
            error = _logistic(sum(weight * value for weight, value in zip(weights, values))) - label
            for index, value in enumerate(values):
                gradients[index] += error * value
        for index in range(len(weights)):
            regularization = 0.02 * weights[index] if index else 0.0
            weights[index] -= 0.15 * (gradients[index] / sample_count + regularization)
    return weights


def fit_and_evaluate(shipments: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    features = [feature_row(shipment) for shipment in shipments]
    feature_names = sorted(
        {f"route={value}" if name == "route" else name for row in features for name, value in row.items()}
    )
    numeric_names = [name for name in feature_names if not name.startswith("route=")]
    feature_names = numeric_names + [name for name in feature_names if name.startswith("route=")]
    matrix = [
        [float(row[name]) for name in numeric_names]
        + [1.0 if name == f"route={row['route']}" else 0.0 for name in feature_names[len(numeric_names):]]
        for row in features
    ]
    delays = [float(shipment["final_delay_hours"]) for shipment in shipments]
    labels = [int(delay > LATE_THRESHOLD_HOURS) for delay in delays]
    indices = list(range(len(shipments)))
    random.Random(42).shuffle(indices)
    test_count = max(1, round(len(indices) * 0.25))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    train_matrix, means, scales = _standardize([matrix[index] for index in train_indices])
    test_matrix, _, _ = _standardize([matrix[index] for index in test_indices], means, scales)
    train_delays = [delays[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    delay_weights = _fit_linear(train_matrix, train_delays)
    late_weights = _fit_logistic(train_matrix, train_labels)
    delay_predictions = [delay_weights[0] + sum(w * x for w, x in zip(delay_weights[1:], row)) for row in test_matrix]
    late_probabilities = [
        _logistic(late_weights[0] + sum(w * x for w, x in zip(late_weights[1:], row)))
        for row in test_matrix
    ]
    late_predictions = [int(probability >= 0.5) for probability in late_probabilities]
    test_labels = [labels[index] for index in test_indices]
    true_positives = sum(actual == 1 and predicted == 1 for actual, predicted in zip(test_labels, late_predictions))
    false_positives = sum(actual == 0 and predicted == 1 for actual, predicted in zip(test_labels, late_predictions))
    false_negatives = sum(actual == 1 and predicted == 0 for actual, predicted in zip(test_labels, late_predictions))
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    accuracy = sum(actual == predicted for actual, predicted in zip(test_labels, late_predictions)) / len(test_labels)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    training_mean_delay = sum(train_delays) / len(train_delays)
    baseline_mae = sum(
        abs(delays[index] - training_mean_delay) for index in test_indices
    ) / len(test_indices)
    majority_label = int(sum(train_labels) * 2 >= len(train_labels))
    baseline_accuracy = sum(label == majority_label for label in test_labels) / len(test_labels)
    metrics = {
        "test_shipments": len(test_indices),
        "mae_hours": sum(abs(actual - predicted) for actual, predicted in zip(
            [delays[index] for index in test_indices], delay_predictions
        )) / len(test_indices),
        "baseline_mae_hours": baseline_mae,
        "late_threshold_hours": LATE_THRESHOLD_HOURS,
        "late_accuracy": accuracy,
        "late_baseline_accuracy": baseline_accuracy,
        "late_precision": precision,
        "late_recall": recall,
        "late_f1": f1,
    }
    models = {
        "feature_names": feature_names,
        "means": means,
        "scales": scales,
        "delay_weights": delay_weights,
        "late_weights": late_weights,
        "test_indices": test_indices,
    }
    return models, metrics


def predict_shipment(models: dict[str, Any], shipment: dict[str, Any]) -> dict[str, Any]:
    features = feature_row(shipment)
    values = [
        1.0 if name == f"route={features['route']}" else 0.0
        if name.startswith("route=")
        else float(features[name])
        for name in models["feature_names"]
    ]
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(values, models["means"], models["scales"])
    ]
    delay_weights = models["delay_weights"]
    late_weights = models["late_weights"]
    delay_hours = delay_weights[0] + sum(weight * value for weight, value in zip(delay_weights[1:], standardized))
    late_probability = _logistic(
        late_weights[0] + sum(weight * value for weight, value in zip(late_weights[1:], standardized))
    )
    contributions = [weight * value for weight, value in zip(delay_weights[1:], standardized)]
    explanations = [
        {"feature": models["feature_names"][index], "hours": contribution}
        for index, contribution in enumerate(contributions)
        if abs(contribution) >= 0.1
    ]
    explanations.sort(key=lambda item: abs(item["hours"]), reverse=True)
    eta = datetime.fromisoformat(shipment["scheduled_arrival"]) + timedelta(hours=delay_hours)
    return {
        "predicted_delay_hours": delay_hours,
        "late_probability": late_probability,
        "predicted_eta": eta,
        "contributions": explanations[:6],
    }


def write_dataset(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    shipments = generate_shipments(count, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=shipments[0].keys())
        writer.writeheader()
        writer.writerows(shipments)
    return shipments


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate the synthetic shipment ETA model.")
    parser.add_argument("--records", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/shipments.csv"))
    args = parser.parse_args()

    shipments = write_dataset(args.output, args.records, args.seed)
    models, metrics = fit_and_evaluate(shipments)
    example = predict_shipment(models, shipments[0])
    print(f"Dataset: {args.output} ({len(shipments)} shipments)")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    print(f"Example predicted ETA: {example['predicted_eta'].isoformat(timespec='minutes')}")
    print(f"Example late probability: {example['late_probability']:.1%}")
    print("Largest delay contributions:")
    for contribution in example["contributions"][:4]:
        print(f"  {contribution['feature']}: {contribution['hours']:+.1f} h")


if __name__ == "__main__":
    main()