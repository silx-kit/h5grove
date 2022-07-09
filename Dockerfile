FROM python:slim-bullseye

COPY h5grove ./h5grove
COPY example ./example
COPY setup.cfg ./setup.cfg
COPY setup.py ./setup.py
COPY tasks.py  ./tasks.py 

ADD http://www.silx.org/pub/h5web/water_224.h5 /data/
ADD http://www.silx.org/pub/h5web/grove.h5 /data/

RUN pip install -e .[flask]
RUN pip install gunicorn

ENV PYTHONUNBUFFERED=1
EXPOSE 8888
ENTRYPOINT ["gunicorn"]
CMD ["-b", "0.0.0.0:8888", "--chdir", "/example", "flask_app:main(basedir='/data')"]
