FROM python:3.11

ADD app.py .

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV NAME configy.env

CMD ["python", "app.py"]