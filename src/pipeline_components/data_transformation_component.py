from kfp.dsl import OutputPath, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.3.2",
        "datasets==4.0.0",
        "gcsfs",
    ],
)
def data_transformation_component(
    raw_dataset_uri: str,
    train_test_split_ratio: float,
    train_dataset: OutputPath("Dataset"),  # type: ignore
    test_dataset: OutputPath("Dataset"),  # type: ignore
) -> None:
    """Formatte et divise le dataset Pirate (simplifié) pour le fine-tuning de Phi-3."""
    import logging

    import pandas as pd
    from datasets import Dataset

    def format_dataset_to_phi_messages(dataset: Dataset) -> Dataset:
        """Formate le dataset à la structure de messages de Phi."""

        def format_dataset(examples):
            """Formate un exemple unique à la structure de messages de Phi."""
            converted_sample = [
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["completion"]},
            ]
            return {"messages": converted_sample}

        return (
            dataset.rename_column("instruction", "prompt")
            .rename_column("response", "completion")
            .map(format_dataset)
            .remove_columns(["prompt", "completion"])
        )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Démarrage du processus de transformation des données...")

    logger.info(f"Lecture depuis {raw_dataset_uri}")
    dataset = Dataset.from_pandas(pd.read_csv(raw_dataset_uri))

    logger.info("Formatage et division du dataset...")
    formatted_dataset = format_dataset_to_phi_messages(dataset)
    split_dataset = formatted_dataset.train_test_split(test_size=train_test_split_ratio)

    logger.info(f"Écriture du dataset d'entraînement dans {train_dataset}...")
    split_dataset["train"].to_csv(train_dataset, index=False)

    logger.info(f"Écriture du dataset de test dans {test_dataset}...")
    split_dataset["test"].to_csv(test_dataset, index=False)

    logger.info("Transformation des données terminée.")
