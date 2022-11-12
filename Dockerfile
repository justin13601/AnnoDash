FROM coady/pylucene:8.11

COPY requirements.txt ./requirements.txt
RUN pip install pip==20.0.2
RUN pip install -r requirements.txt

COPY . ./

WORKDIR .
EXPOSE 8080
ENTRYPOINT gunicorn -b 0.0.0.0:8080 main:server
#ENTRYPOINT gunicorn -b 0.0.0.0:80 main:server