# Use the Ubuntu base image
FROM ubuntu:latest

# Update the package lists
RUN apt-get update && apt-get -y upgrade

# Install Python and pip
RUN apt-get install -y python3 python3-pip

COPY dataset/ /Kmeans/dataset/
COPY features/ /Kmeans/features/
COPY main.py README.md requirements.txt /Kmeans/
WORKDIR /Kmeans

RUN pip install -r requirements.txt
CMD python main.py