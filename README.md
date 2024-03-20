# OMOP Note -> Note_NLP

This is a simple script that will take your [OMOP](https://ohdsi.github.io/CommonDataModel/index.html) compliant [Note](https://ohdsi.github.io/CommonDataModel/cdm54.html#note) table and convert it into an OMOP [Note_NLP](https://ohdsi.github.io/CommonDataModel/cdm54.html#note_nlp) table. 

The goal of this project is to: "Enable NLP analysis for **semi**-technical analysts and programmers"

## Installation

Clone this repository using git or the [GitHub CLI](https://cli.github.com)
```bash
git clone https://github.com/UK-IPOP/omop-nlp.git
# or
gh repo clone UK-IPOP/omop-nlp
```

Setup python and install dependencies. I recommend using [conda](https://docs.conda.io/en/latest/) to create a python virtual environment.

> NOTE: `scispacy` requires python3.9

Run the following _inside_ the repository you cloned:
```bash
# -y accept defaults
conda create -n omop-nlp -y
conda activate omop-nlp
# install requirements (rich for logging, pandas for file reading, and scispacy -- which includes spacy itself)
pip install -r requirements.txt
# install scispacy model
# NOTE: this model is a few GB
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

## Usage

Once you have created the relevant python environment, usage is simple, again inside the cloned directory:
```bash
python run_nlp.py <OMOP-DIR>
```

Replace `<OMOP-DIR>` with the directory of your OMOP files. For example:
```bash
python run_nlp.py ~/Documents/OMOP/
```

More CLI options can be discovered by running the help option:
```bash
python run_nlp.py --help
```

## Things to Know

This project is intentionally minimal and only accomplishes the following:

- Utilize _large_ `scispacy` model for named entity recognition of biomedical concepts
- Utilize `scispacy` `EntityLinker` to extract [UMLS](https://uts.nlm.nih.gov/uts/umls/home)https://uts.nlm.nih.gov/uts/umls/home) concepts
- Utilizze `negspacy` for concept negation
- Convert relevant `spacy` outputs to their OMOP fields:
  - `note_nlp_id`: a unique identifier
  - `note_id`: a linked note id from the Note table
  - `lexical_variant`: the entity we extracted
  - `note_nlp_source_concept_id`: the CUI of the UMLS concept we linked to
  - `nlp_system`: "scispacy"
  -  `nlp_datetime`: datetime script was run
  -  `nlp_date`: date script was run
  -  `term_modifiers`: "Negation=True/False" based on negspacy

This encompasses **all** of the **required** Note_NLP fields and **some** optional fields. This is intentional to limit potentially irrelevant information to a user and to decrease the surface area of code that must be maintained long-term. The following exercises are left to the user (with pull-requests/branches for their feature implemenations welcomed):

- Linking UMLS CUIs to OMOP concept_ids (`note_nlp_concept_id`)
- Extracting supplemental text information (`offset` or `term_temporal`)
- Transitioning from directory/file based storage to alternative storage solutions. Potential options include:
  - S3 data stores
  - Databases/warehouse
  - Note that becuase we use `pandas`, reading your Note table (and thus writing the Note_NLP table) are limited only by the data sources `pandas` can read from and thus the above can be implemented with ease
- Dockerizing this process and publishing it as a service
  - Note here that the models use a _significant_ amount of RAM so provision your infrastructure and container accordingly

## Resources

- [scispacy](https://github.com/allenai/scispacy)
- [negspacy](https://github.com/jenojp/negspacy)
- [spacy](https://spacy.io)
- [pandas](https://pandas.pydata.org)
- [UMLS](https://uts.nlm.nih.gov/uts/umls/home)

## Contributions

Contributions are welcomed. Please feel free to either create your own fork or submit pull-requests with corrections or feature additions. 
