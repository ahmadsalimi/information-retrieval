FROM joyzoursky/python-chromedriver

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
