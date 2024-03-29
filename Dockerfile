FROM python:3.8.15-slim-bullseye

# install base packages
RUN apt-get clean \
    && apt-get update --fix-missing \
    && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    build-essential \
    wget \
    awscli

WORKDIR /work

# install python packages
COPY requirements.in .

# copy over the data file
COPY data/s2and_name_tuples.txt ./data/
COPY data/s2and_name_tuples_filtered.txt ./data/
COPY data/path_config.json ./data/
COPY data/first_k_letter_counts_from_orcid.json ./data/

# add the code as the final step so that when we modify the code
# we don't bust the cached layers holding the dependencies and
# system packages.
COPY s2and/ s2and/
COPY scripts/ scripts/
COPY tests/ tests/

RUN pip install -r requirements.in

COPY setup.py .
RUN python setup.py develop

RUN aws s3 cp --no-sign-request s3://ai2-s2-research-public/s2and-release/lid.176.bin data/
RUN aws s3 cp --no-sign-request s3://ai2-s2-research-public/s2and-release/name_counts.pickle data/

CMD [ "/bin/bash" ]
