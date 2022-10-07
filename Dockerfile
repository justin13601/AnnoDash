FROM coady/pylucene:8.11

COPY requirements.txt ./requirements.txt
RUN pip install pip==20.0.2
RUN pip install -r requirements.txt

COPY . ./

WORKDIR .
ENTRYPOINT gunicorn -b 0.0.0.0:8080 app:server