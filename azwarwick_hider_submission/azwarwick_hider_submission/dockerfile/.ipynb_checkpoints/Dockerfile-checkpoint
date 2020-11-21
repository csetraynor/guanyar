# Dockerfile example 2.

# This Dockerfile will set up a minimal set of dependencies needed for running submissions.
# Note that tensorflow 1.15 or keras 2.3.1 are not included.
# Can be a useful starting point if solution needs, e.g. tensorflow 2.

FROM python:3.6

RUN apt-get update -qy
RUN apt-get install -qy python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install scipy==1.4.1 numpy==1.18.2 pandas==1.0.3 tqdm==4.36.1 scikit-learn==0.21.3 tensorflow==2.3.0
