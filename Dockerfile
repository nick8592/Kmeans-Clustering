FROM python
COPY main.py requirements.txt /main/
WORKDIR /main
RUN apt-get update && apt-get -y upgrade
RUN pip install -r requirements.txt
CMD python main.py