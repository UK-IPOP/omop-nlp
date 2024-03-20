from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union
import pandas as pd
import warnings
from rich import print, pretty
from rich.console import Console
from rich.progress import track
import spacy
from spacy.language import Language
from negspacy.negation import Negex
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking_utils import Entity
from spacy.tokens import Doc

# formatted printing
pretty.install()

# logger
console = Console()

# don't show warnings
warnings.filterwarnings("ignore")


def build_models() -> tuple[Language, EntityLinker]:
    """Builds the spacy NLP pipeline and scispacy linker object.

    Returns:
        tuple[Language, EntityLinker]: The spacy NLP pipeline and scispacy linker object
    """
    console.log("[cyan]Setting up...")
    # load scispacy large biomedical NER model
    # disable everything except the NER model
    console.log("[yellow]Loading spacy model...")
    nlp = spacy.load(
        "en_core_sci_lg",
        # remove things not needed for NER, increases perf
        exclude=["tagger", "lemmatizer", "textcat"],
    )
    # load spacy pipeline components
    console.log("[yellow]Loading spacy pipeline components...")
    # use negex
    nlp.add_pipe("negex")
    nlp.add_pipe("abbreviation_detector", config={"make_serializable": True})
    nlp.add_pipe(
        "scispacy_linker",
        config={
            # use abbreviations
            "resolve_abbreviations": True,
            # only give us the best corresponding entity
            "max_entities_per_mention": 1,
            # default is 30
            "k": 30,
            # default is 0.7
            "threshold": 0.9,
            # this is the big one! limits to only entities with definitions in knowledge base
            "filter_for_definitions": True,
            # use UMLS knowledge base
            "linker_name": "umls",
        },
    )
    # return nlp object and scispacy linker object
    linker: EntityLinker = nlp.get_pipe("scispacy_linker")  # type: ignore
    return nlp, linker


def load_notes(omop_dir: Path) -> pd.DataFrame:
    """Loads the NOTE.csv table from the OMOP directory.

    Args:
        omop_dir (Path): The OMOP directory

    Returns:
        pd.DataFrame: The NOTE.csv table
    """
    console.log("[cyan]Loading dataset...")
    return pd.read_csv(
        omop_dir / "NOTE.csv",
        low_memory=False,
        dtype_backend="pyarrow",
        usecols=["note_id", "note_text"],
        dtype={"note_id": "int32", "note_text": "string"},
    )


def convert_to_text_tuples(df: pd.DataFrame) -> list[tuple[str, dict[str, int]]]:
    """Converts the NOTE.csv table into a list of tuples for NLP pipeline.

    Args:
        df (pd.DataFrame): The NOTE.csv table

    Returns:
        list[tuple[str, dict[str, int]]]: The list of tuples for NLP pipeline
    """
    data: list[tuple[str, dict[str, int]]] = []
    for _, row in df.iterrows():
        note_id: int = int(row["note_id"])
        note_text: str = str(row["note_text"])
        data.append((note_text, {"note_id": note_id}))
    return data


def build_nlp_pipe(
    nlp: Language,
    text_tuples: list[tuple[str, dict[str, int]]],
    n_processes: int,
    batch_size: int,
) -> Iterator[tuple[Doc, dict[str, int]]]:
    """Builds the NLP pipeline.

    Args:
        nlp (Language): The spacy NLP pipeline
        text_tuples (list[tuple[str, dict[str, int]]]): The list of tuples for NLP pipeline
        n_processes (int): The number of processes to use for multi-processing pipeline
        batch_size (int): The batch size to use for multi-processing pipeline

    Returns:
        Iterator[tuple[Doc, dict[str, int]]]: The NLP pipeline docs
    """
    console.log("[cyan]Calling NLP pipeline...")
    doc_tuples = nlp.pipe(
        texts=text_tuples,
        batch_size=batch_size,
        n_process=n_processes,
        as_tuples=True,
    )
    return doc_tuples


def consume_nlp_pipe(
    pipe: Iterator[tuple[Doc, dict[str, int]]],
    lookup: dict[str, Entity],
    n_items: int,
) -> pd.DataFrame:
    """Consumes the NLP pipeline and processes the results.

    Args:
        pipe (Iterator[tuple[Doc, dict[str, int]]]): The NLP pipeline docs
        lookup (dict[str, Entity]): The scispacy linker lookup table
        n_items (int): The number of items in the NLP pipeline

    Returns:
        pd.DataFrame: The processed results as a dataframe
    """
    output: list[dict[str, Union[str, bool, float, datetime]]] = []
    timestamp = datetime.now()
    for doc, context in track(
        pipe, total=n_items, description="Processing documents..."
    ):
        # process the results
        for ent in doc.ents:
            for kb_ent in ent._.kb_ents:
                # get first/best match
                concept = lookup[kb_ent[0]]
                results: dict[str, Union[str, bool, float, datetime]] = {
                    "note_id": context["note_id"],
                    "cui": concept.concept_id.strip(),
                    "name": concept.canonical_name.strip(),
                    "entity": ent.text.strip(),
                    "negated": ent._.negex,
                    "score": kb_ent[1],
                    "nlp_datetime": timestamp,
                }
                output.append(results)
    # make into a dataframe
    return pd.DataFrame(output)


def convert_to_nlp_table(results: pd.DataFrame) -> pd.DataFrame:
    """Converts the processed results into the NOTE_NLP.csv table.

    Args:
        results (pd.DataFrame): The processed results as a dataframe

    Returns:
        pd.DataFrame: The NOTE_NLP.csv table
    """
    results["note_nlp_id"] = list(range(1, len(results) + 1))
    results["nlp_system"] = f"scispacy {scispacy.__version__}"
    results["nlp_date"] = results["nlp_datetime"].dt.date
    results["term_modifiers"] = results["negated"].apply(
        lambda x: "Negated=True" if x else "Negated=False"
    )
    results.drop(columns=["name", "negated", "score"], inplace=True)
    results.rename(
        columns={
            "entity": "lexical_variant",
            "cui": "note_nlp_source_concept_id",
        },
        inplace=True,
    )
    # return this slice as it allows us to order the columns
    output_cols = [
        "note_nlp_id",
        "note_id",
        "lexical_variant",
        "nlp_system",
        "note_nlp_source_concept_id",
        "nlp_date",
        "nlp_datetime",
        "term_modifiers",
    ]
    sliced: pd.DataFrame = results[output_cols].copy()  # type: ignore
    return sliced


def run(
    omop_directory: Path,
    n_processes: int,
    batch_size: int,
) -> None:
    """Runs the NLP-NER pipeline on the NOTE.csv table in the OMOP directory.

    Args:
        omop_directory (Path): The OMOP directory
        n_processes (int): The number of processes to use for multi-processing pipeline
        batch_size (int): The batch size to use for multi-processing pipeline

    Returns:
        None
    """
    dataset = load_notes(omop_dir=omop_directory)

    nlp_model, kb_linker = build_models()
    lookup_table = kb_linker.kb.cui_to_entity

    text_tuples = convert_to_text_tuples(df=dataset)

    nlp_pipe = build_nlp_pipe(
        nlp=nlp_model,
        text_tuples=text_tuples,
        n_processes=n_processes,
        batch_size=batch_size,
    )

    results = consume_nlp_pipe(
        pipe=nlp_pipe,
        lookup=lookup_table,
        n_items=len(text_tuples),
    )

    nlp_table = convert_to_nlp_table(results=results)

    nlp_table.to_csv(omop_directory / "NOTE_NLP.csv", index=False)


def main():
    """Main program entry point.

    document the below argparse arguments
    Argparse arguments:

    - dir: The OMOP source directory
    - processes: The number of processes to use for multi-processing pipeline
    - batch-size: The batch size to use for multi-processing pipeline
    """
    parser = ArgumentParser(
        prog="run_nlp.py",
        description="""
        Runs NLP-NER on the NOTE.csv table in the provided OMOP directory. 
        Outputs the NOTE_NLP.csv table in the same directory.
        """,
    )
    parser.add_argument(
        "dir",
        type=Path,
        help="The OMOP source directory",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=4,
        required=False,
        help="The number of processes to use for multi-processing pipeline: USE WITH CAUTION",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1_000,
        required=False,
        help="The batch size to use for multi-processing pipeline",
    )
    args = parser.parse_args()

    if args.processes < 1:
        raise ValueError("The number of processes must be greater than 0")

    if args.batch_size < 1:
        raise ValueError("The batch size must be greater than 0")

    if args.dir.exists() is False:
        raise ValueError("The OMOP directory does not exist")

    if args.dir.is_dir() is False:
        raise ValueError("The OMOP directory is not a directory")

    run(
        omop_directory=args.dir,
        n_processes=args.processes,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
