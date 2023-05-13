FROM python
COPY dataset/ /Kmeans/dataset
COPY features/ /Kmeans/features
COPY main.py README.md requirements.txt /Kmeans
WORKDIR /Kmeans
RUN apt-get update && apt-get -y upgrade
RUN pip install -r requirements.txt
CMD python main.py