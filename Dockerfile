FROM python
COPY /dataset/ /features/ /Kmeans/dataset
COPY main.py requirements.txt /Kmeans
WORKDIR /Kmeans
RUN apt-get update && apt-get -y upgrade
RUN pip install -r requirements.txt
CMD python main.py